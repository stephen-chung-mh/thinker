# distutils: language = c++
import numpy as np
import gym
import torch
#from thinker import util
import thinker.util as util
from thinker.net import ModelNetOut

import cython
from libcpp cimport bool
from libcpp.vector cimport vector
from cpython.ref cimport PyObject, Py_INCREF, Py_DECREF
from libc.math cimport sqrt
from libc.stdlib cimport malloc, free

# util function

@cython.cdivision(True)
cdef float average(vector[float]& arr):
    cdef int n = arr.size()    
    if n == 0: return 0.
    cdef float sum = 0
    cdef int i    
    for i in range(n): sum += arr[i]
    return sum / n    

cdef float maximum(vector[float]& arr):
    cdef int n = arr.size()    
    if n == 0: return 0.
    cdef float max_val = arr[0]
    cdef int i
    for i in range(1, n): 
        if arr[i] > max_val: 
            max_val = arr[i]
    return max_val       

# Node-related function (we use structure instead of class to minimize Python code)

cdef struct Node:
    int action # action
    float r # reward
    float v # value
    int t # time step when last expanded
    bool done # whether done or not
    float logit # logit    
    vector[Node*]* ppchildren # children node list
    Node* pparent # parent node
    float trail_r # trailing reward
    float trail_discount # trailing discount
    float rollout_q # trailing rollout q    
    bool visited # visited?
    vector[float]* prollout_qs # all rollout return
    vector[vector[Node*]*]* ppaths # node path corresponding to the rollout return in prollout_qs
    int rollout_n # number of rollout
    float max_q # maximum of all v    
    PyObject* encoded # all python object    
    int rec_t # number of planning step
    int num_actions # number of actions
    float discounting # discount rate
    bool remember_path # whethere to remember path for the rollout

cdef Node* node_new(Node* pparent, int action, float logit, int num_actions, float discounting, int rec_t, bool remember_path):
    cdef Node* pnode = <Node*> malloc(sizeof(Node))
    cdef vector[Node*]* ppchildren =  new vector[Node*]()
    cdef vector[float]* prollout_qs = new vector[float]()
    cdef vector[vector[Node*]*]* ppaths
    if remember_path:
        ppaths = new vector[vector[Node*]*]()
    else:
        ppaths = NULL
    pnode[0] = Node(action=action, r=0., v=0., t=0, done=False, logit=logit, ppchildren=ppchildren, pparent=pparent, trail_r=0., trail_discount=1., rollout_q=0,
        visited=False, prollout_qs=prollout_qs, ppaths=ppaths, rollout_n=0, max_q=0., encoded=NULL, rec_t=rec_t, num_actions=num_actions, discounting=discounting, remember_path=remember_path)
    return pnode

cdef bool node_expanded(Node* pnode, int t):
    """
    Whether the node is expanded after time step t
    """
    return pnode[0].ppchildren[0].size() > 0 and t <= pnode[0].t

cdef node_expand(Node* pnode, float r, float v, int t, bool done, float[:] logits, PyObject* encoded, bool override):
    """
    First time arriving a node and so we expand it
    """
    cdef int a    
    cdef Node* pnode_
    if override and not node_expanded(pnode, -1):
        override = False # no override if not yet expanded

    if not override: 
        assert not node_expanded(pnode, -1), "node should not be expanded"
    else:        
        pnode[0].prollout_qs[0][0] = r + v * pnode[0].discounting
        for a in range(1, int(pnode[0].prollout_qs[0].size())):
            pnode[0].prollout_qs[0][a] = pnode[0].prollout_qs[0][a] - pnode[0].r + r
        if pnode[0].pparent != NULL and pnode[0].remember_path:
            node_refresh(pnode[0].pparent, pnode, r - pnode[0].r, v - pnode[0].v, pnode[0].discounting, 1)
    pnode[0].r = r
    pnode[0].v = v
    pnode[0].t = t
    if pnode[0].encoded != NULL: 
        Py_DECREF(<object>pnode[0].encoded)    
    pnode[0].encoded = encoded
    pnode[0].done = done
    Py_INCREF(<object>encoded)
    for a in range(pnode[0].num_actions):
        if not override:
            pnode[0].ppchildren[0].push_back(node_new(pparent=pnode, action=a, logit=logits[a], 
                num_actions = pnode[0].num_actions, discounting = pnode[0].discounting, rec_t = pnode[0].rec_t,
                remember_path = pnode[0].remember_path))
        else:
            pnode[0].ppchildren[0][a][0].logit = logits[a]    

cdef node_refresh(Node* pnode, Node* pnode_to_refresh, float r_diff, float v_diff, float discounting, int depth):
    """
    Refresh the r and v in the rollout_qs that contains pnode[0]; only available when remember_path is enabled
    """
    cdef int i, j, k
    cdef Node* pnode_check
    for i in range(int(pnode[0].ppaths[0].size())):
        j = int(pnode[0].ppaths[0][i][0].size())
        k = j - 1 - depth
        if k < 0: continue
        pnode_check = pnode[0].ppaths[0][i][0][k]
        if pnode_check == pnode_to_refresh:
            pnode[0].prollout_qs[0][i] += discounting * r_diff
            if k == 0:
                pnode[0].prollout_qs[0][i] += discounting * pnode[0].discounting * v_diff
    if pnode[0].pparent != NULL: node_refresh(pnode[0].pparent, pnode_to_refresh, r_diff, v_diff, discounting * pnode[0].discounting, depth+1)

cdef node_visit(Node* pnode):
    cdef vector[Node*]* ppath    
    pnode[0].trail_r = 0.
    pnode[0].trail_discount = 1.    
    if not pnode[0].visited and pnode[0].remember_path:
        ppath = new vector[Node*]()
    else:
        ppath = NULL
    node_propagate(pnode=pnode, r=pnode[0].r, v=pnode[0].v, new_rollout=not pnode[0].visited, ppath=ppath)
    pnode[0].visited = True

cdef void node_propagate(Node* pnode, float r, float v, bool new_rollout, vector[Node*]* ppath):
    cdef int i
    cdef vector[Node*]* ppath_
    pnode[0].trail_r = pnode[0].trail_r + pnode[0].trail_discount * r
    pnode[0].trail_discount = pnode[0].trail_discount * pnode[0].discounting
    pnode[0].rollout_q = pnode[0].trail_r + pnode[0].trail_discount * v    
    if new_rollout:
        if pnode[0].remember_path:
            ppath_ = new vector[Node*]()
            for i in range(int(ppath[0].size())):                
                ppath_.push_back(ppath[0][i])
            ppath_.push_back(pnode)
            pnode[0].ppaths[0].push_back(ppath_)
        else:
            ppath_ = NULL
        pnode[0].prollout_qs[0].push_back(pnode[0].rollout_q)        
        pnode[0].rollout_n = pnode[0].rollout_n + 1        
    if pnode[0].pparent != NULL: 
        node_propagate(pnode[0].pparent, r, v, new_rollout, ppath=ppath_)

#@cython.cdivision(True)
cdef float[:] node_stat(Node* pnode, bool detailed, int mask_type):
    cdef float[:] result = np.zeros((pnode[0].num_actions*5+5) if detailed else (pnode[0].num_actions*5+2), dtype=np.float32) 
    cdef int i
    result[pnode[0].action] = 1. # action
    result[pnode[0].num_actions] = pnode[0].r # reward
    if not mask_type in [2]: 
        result[pnode[0].num_actions+1] = pnode[0].v # value
    for i in range(int(pnode[0].ppchildren[0].size())):
        child = pnode[0].ppchildren[0][i][0]
        if not mask_type in [2]: 
            result[pnode[0].num_actions+2+i] = child.logit # child_logits
        if not mask_type in [1, 2]: 
            result[pnode[0].num_actions*2+2+i] = average(child.prollout_qs[0]) # child_rollout_qs_mean
            result[pnode[0].num_actions*3+2+i] = maximum(child.prollout_qs[0]) # child_rollout_qs_max
            result[pnode[0].num_actions*4+2+i] = child.rollout_n / <float>pnode[0].rec_t # child_rollout_ns_enc
    if detailed and not mask_type in [1, 2]:
        pnode[0].max_q = (maximum(pnode[0].prollout_qs[0]) - pnode[0].r) / pnode[0].discounting
        result[pnode[0].num_actions*5+2] = pnode[0].trail_r / pnode[0].discounting
        result[pnode[0].num_actions*5+3] = pnode[0].rollout_q / pnode[0].discounting
        result[pnode[0].num_actions*5+4] = pnode[0].max_q
    return result

cdef node_del(Node* pnode, int except_idx):
    cdef int i
    del pnode[0].prollout_qs

    if pnode[0].ppaths != NULL:
        for i in range(int(pnode[0].ppaths[0].size())):
            del pnode[0].ppaths[0][i]
        del pnode[0].ppaths

    for i in range(int(pnode[0].ppchildren[0].size())):
        if i != except_idx:
            node_del(pnode[0].ppchildren[0][i], -1)
        else:
            pnode[0].ppchildren[0][i][0].pparent = NULL
    del pnode[0].ppchildren
    if pnode[0].encoded != NULL:
        Py_DECREF(<object>pnode[0].encoded)
    free(pnode)

cdef class cVecFullModelWrapper():
    """Wrap the gym environment with a model; output for each 
    step is (out, reward, done, info), where out is a tuple 
    of (gym_env_out, model_out, model_encodes) that corresponds to underlying 
    environment frame, output from the model wrapper, and encoding from the model
    Assume a learned dynamic model.
    """
    # setting
    cdef int rec_t
    cdef int rep_rec_t
    cdef float discounting
    cdef int max_allow_depth
    cdef bool perfect_model
    cdef bool tree_carry  
    cdef int actor_see_type
    cdef bool actor_see_double_encode 
    cdef bool pred_done
    cdef int num_actions
    cdef int obs_n    
    cdef int env_n
    cdef bool im_enable
    cdef bool time 
    cdef int stat_mask_type
    cdef bool debug

    # python object
    cdef object device
    cdef object env
    cdef object timings
    cdef readonly baseline_max_q
    cdef readonly baseline_mean_q    
    cdef readonly object model_out_shape
    cdef readonly object gym_env_out_shape
    cdef readonly object xs

    # tree statistic
    cdef vector[Node*] cur_nodes
    cdef vector[Node*] root_nodes    
    cdef float[:] root_nodes_qmax
    cdef float[:] root_nodes_qmax_
    cdef int[:] rollout_depth
    cdef int[:] max_rollout_depth
    cdef int[:] cur_t

    # internal variables only used in step function
    cdef float[:] depth_delta
    cdef int[:] max_rollout_depth_
    cdef float[:] mean_q
    cdef float[:] max_q
    cdef int[:] status
    cdef vector[Node*] cur_nodes_
    cdef float[:] par_logits
    cdef float[:, :] full_reward
    cdef bool[:] full_done
    cdef bool[:] full_real_done
    cdef bool[:] full_truncated_done
    cdef int[:] total_step    

    def __init__(self, env, env_n, flags, device=None, time=False, debug=False):
        assert not flags.perfect_model, "this class only supports imperfect model"
        self.device = torch.device("cpu") if device is None else device
        self.env = env     
        if flags.test_rec_t > 0:
            self.rec_t = flags.test_rec_t
            self.rep_rec_t = flags.rec_t         
        else:
            self.rec_t = flags.rec_t           
            self.rep_rec_t = flags.rec_t         
        self.discounting = flags.discounting
        self.max_allow_depth = flags.max_depth
        self.perfect_model = flags.perfect_model
        self.tree_carry = flags.tree_carry
        self.num_actions = env.action_space[0].n
        self.im_enable = flags.im_cost > 0.
        self.actor_see_type = flags.actor_see_type  
        self.actor_see_double_encode = False
        self.pred_done = flags.model_done_loss_cost > 0.
        self.stat_mask_type = flags.stat_mask_type
        self.env_n = env_n
        self.obs_n = 9 + self.num_actions * 10 + self.rep_rec_t
        self.model_out_shape = (self.obs_n, 1, 1)
        self.gym_env_out_shape = env.observation_space.shape[1:]

        self.baseline_max_q = torch.zeros(self.env_n, dtype=torch.float32, device=self.device)
        self.baseline_mean_q = torch.zeros(self.env_n, dtype=torch.float32, device=self.device)        
        self.time = time        
        self.timings = util.Timings()
        self.debug = debug

        # internal variable init.
        self.depth_delta = np.zeros(self.env_n, dtype=np.float32)
        self.max_rollout_depth_ = np.zeros(self.env_n, dtype=np.intc)
        self.mean_q =  np.zeros(self.env_n, dtype=np.float32)
        self.max_q = np.zeros(self.env_n, dtype=np.float32)
        self.status = np.zeros(self.env_n, dtype=np.intc)
        self.par_logits = np.zeros(self.num_actions, dtype=np.float32)
        self.full_reward = np.zeros((self.env_n, 1 + int(self.im_enable)), 
            dtype=np.float32)
        self.full_done = np.zeros(self.env_n, dtype=np.bool)
        self.full_real_done = np.zeros(self.env_n, dtype=np.bool)
        self.full_truncated_done = np.zeros(self.env_n, dtype=np.bool)
        self.total_step = np.zeros(self.env_n, dtype=np.intc)
        
    def reset(self, model_net):
        """reset the environment; should only be called in the initial"""
        cdef int i
        cdef Node* root_node
        cdef Node* cur_node
        cdef float[:,:] model_out        

        with torch.no_grad():
            # some init.
            self.root_nodes_qmax = np.zeros(self.env_n, dtype=np.float32)
            self.root_nodes_qmax_ = np.zeros(self.env_n, dtype=np.float32)
            self.rollout_depth = np.zeros(self.env_n, dtype=np.intc)
            self.max_rollout_depth = np.zeros(self.env_n, dtype=np.intc)
            self.cur_t = np.zeros(self.env_n, dtype=np.intc)

            # reset obs
            obs = self.env.reset()

            # obtain output from model
            obs_py = torch.tensor(obs, dtype=torch.uint8, device=self.device)
            pass_action = torch.zeros(self.env_n, dtype=torch.long)
            model_net_out = model_net(obs_py, 
                                      pass_action.unsqueeze(0).to(self.device), 
                                      one_hot=False)  
            vs = model_net_out.vs.cpu()
            logits = model_net_out.logits.cpu()      

            # compute and update root node and current node
            for i in range(self.env_n):
                root_node = node_new(pparent=NULL, action=pass_action[i].item(), logit=0., num_actions=self.num_actions, 
                    discounting=self.discounting, rec_t=self.rec_t, remember_path=True)                
                encoded = {"gym_env_out": obs_py[i], 
                           "model_ys": model_net_out.ys[-1,i] if self.actor_see_type >= 0 else None,
                           "model_states": dict({sk:sv[[i]] for sk, sv in model_net_out.state.items()})
                          }  
                node_expand(pnode=root_node, r=0., v=vs[-1, i].item(), t=self.total_step[i], done=False,
                    logits=logits[-1, i].numpy(), encoded=<PyObject*>encoded, override=False)
                node_visit(pnode=root_node)
                self.root_nodes.push_back(root_node)
                self.cur_nodes.push_back(root_node)
            
            # compute model_out
            model_out = self.compute_model_out(None, None)

            gym_env_out = []
            for i in range(self.env_n):
                encoded = <dict>self.cur_nodes[i][0].encoded
                if encoded["gym_env_out"] is not None:
                    gym_env_out.append(encoded["gym_env_out"].unsqueeze(0))
            if len(gym_env_out) > 0:
                gym_env_out = torch.concat(gym_env_out)
            else:
                gym_env_out = None

            if self.actor_see_type >= 0:
                model_encodes = self.compute_model_encodes(self.cur_nodes)
                if self.actor_see_double_encode:
                    model_encodes = torch.concat([model_encodes, model_encodes], dim=1)
            else:
                model_encodes = None

            if self.debug:
                self.xs =  self.compute_model_xs(self.cur_nodes)

            # record initial root_nodes_qmax 
            for i in range(self.env_n):
                self.root_nodes_qmax[i] = self.root_nodes[i][0].max_q
            
            return torch.tensor(model_out, dtype=torch.float32, device=self.device), gym_env_out, model_encodes

    def step(self, action, model_net):  
        # action is tensor of shape (env_n, 3)
        # which corresponds to real_action, im_action, reset, term
        
        cdef int i, j, k, l
        cdef int[:] re_action
        cdef int[:] im_action
        cdef int[:] reset

        cdef Node* root_node
        cdef Node* cur_node
        cdef Node* next_node
        cdef vector[Node*] cur_nodes_
        cdef vector[Node*] root_nodes_    
        cdef float[:,:] model_out        

        cdef vector[int] pass_inds_restore
        cdef vector[int] pass_inds_step
        cdef vector[int] pass_inds_reset
        cdef vector[int] pass_inds_reset_
        cdef vector[int] pass_action
        cdef vector[int] pass_model_action

        cdef float[:] vs_1
        cdef float[:,:] logits_1

        cdef float[:] rs_4
        cdef float[:] vs_4
        cdef float[:,:] logits_4

        if self.time: self.timings.reset()
        action = action.cpu().int().numpy()
        re_action, im_action, reset = action[:, 0], action[:, 1], action[:, 2]

        pass_model_states, pass_re_model_states = [], []

        for i in range(self.env_n):            
            # compute the mask of real / imagination step                             
            self.max_rollout_depth_[i] = self.max_rollout_depth[i]
            self.depth_delta[i] = 1.
            if self.cur_t[i] < self.rec_t - 1: # imagaination step
                self.cur_t[i] += 1
                self.rollout_depth[i] += 1
                self.max_rollout_depth[i] = max(self.max_rollout_depth[i], self.rollout_depth[i])
                next_node = self.cur_nodes[i][0].ppchildren[0][im_action[i]]
                if node_expanded(next_node, self.total_step[i]):
                    self.status[i] = 2
                elif self.cur_nodes[i][0].done:
                    self.status[i] = 3
                else:
                    encoded = <dict> self.cur_nodes[i][0].encoded
                    pass_model_states.append(encoded["model_states"])
                    pass_model_action.push_back(im_action[i])
                    self.status[i] = 4  
            else: # real step
                self.cur_t[i] = 0
                self.rollout_depth[i] = 0          
                self.max_rollout_depth[i] = 0
                self.total_step[i] = self.total_step[i] + 1
                # record baseline before moving on
                self.baseline_mean_q[i] = average(self.root_nodes[i][0].prollout_qs[0]) / self.discounting
                self.baseline_max_q[i] = maximum(self.root_nodes[i][0].prollout_qs[0]) / self.discounting
                encoded = <dict> self.root_nodes[i][0].encoded
                pass_re_model_states.append(encoded["model_states"])
                pass_inds_restore.push_back(i)
                pass_action.push_back(re_action[i])                
                pass_inds_step.push_back(i)
                self.status[i] = 1                              
        if self.time: self.timings.time("misc_1")

        # one step of env
        if pass_inds_step.size() > 0:
            obs, reward, done, info = self.env.step(pass_action, inds=pass_inds_step) 
            real_done = [m["real_done"] if "real_done" in m else done[n] for n, m in enumerate(info)]
            truncated_done = [m["truncated_done"] if "truncated_done" in m else False for n, m in enumerate(info)]
        if self.time: self.timings.time("step_state")

        # reset needed?
        for i, j in enumerate(pass_inds_step):
            if done[i]:
                pass_inds_reset.push_back(j)
                pass_inds_reset_.push_back(i) # index within pass_inds_step

        # reset
        if pass_inds_reset.size() > 0:
            obs_reset = self.env.reset(inds=pass_inds_reset) 
            for i, j in enumerate(pass_inds_reset_):
                obs[j] = obs_reset[i]
                pass_action[j] = 0        

        # use model for status 1 transition (real transition)
        if pass_inds_step.size() > 0:
            with torch.no_grad():
                obs_py = torch.tensor(obs, dtype=torch.uint8, device=self.device)
                model_net_out_1 = model_net(obs_py, 
                        torch.tensor(pass_action, dtype=long, device=self.device).unsqueeze(0), 
                        one_hot=False,
                        ret_zs=False)                      
                    
            vs_1 = model_net_out_1.vs[-1].float().cpu().numpy()
            logits_1 = model_net_out_1.logits[-1].float().cpu().numpy()
                
        if self.time: self.timings.time("misc_2")
        # use model for status 4 transition (imagination transition)
        if pass_model_action.size() > 0:
            with torch.no_grad():
                pass_model_states = dict({msd: torch.concat([ms[msd] for ms in pass_model_states], dim=0)
                        for msd in pass_model_states[0].keys()})
                    # all batch index are in the first index 
                model_net_out_4 = model_net.forward_single(
                    state = pass_model_states,
                    action = torch.tensor(pass_model_action, dtype=long, device=self.device),                     
                    one_hot = False,)  
            rs_4 = model_net_out_4.rs[-1].float().cpu().numpy()
            vs_4 = model_net_out_4.vs[-1].float().cpu().numpy()
            logits_4 = model_net_out_4.logits[-1].float().cpu().numpy()
            if self.pred_done:
                done_4 = model_net_out_4.dones[-1].bool().cpu().numpy()

        # compute the current and root nodes
        j = 0 # counter for status 1 transition
        l = 0 # counter for status 4 transition

        for i in range(self.env_n):
            if self.status[i] == 1:
                # real transition
                new_root = (not self.tree_carry or 
                    not node_expanded(self.root_nodes[i][0].ppchildren[0][re_action[i]], -1) or done[j])
                encoded = {"gym_env_out": obs_py[j], 
                           "model_ys": model_net_out_1.ys[-1,j] if self.actor_see_type >= 0 else None,
                           "model_states": dict({sk:sv[[j]] for sk, sv in model_net_out_1.state.items()})
                          }         
                if new_root:
                    root_node = node_new(pparent=NULL, action=pass_action[j], logit=0., num_actions=self.num_actions, 
                        discounting=self.discounting, rec_t=self.rec_t, remember_path=True)                    
                    node_expand(pnode=root_node, r=0., v=vs_1[j], t=self.total_step[i], done=False,
                        logits=logits_1[j], encoded=<PyObject*>encoded, override=False)
                    node_del(self.root_nodes[i], except_idx=-1)
                    node_visit(root_node)
                else:
                    root_node = self.root_nodes[i][0].ppchildren[0][re_action[i]]
                    node_expand(pnode=root_node, r=0., v=vs_1[j], t=self.total_step[i], done=False,
                        logits=logits_1[j], encoded=<PyObject*>encoded, override=True)                        
                    node_del(self.root_nodes[i], except_idx=re_action[i])
                    node_visit(root_node)
                    
                j += 1
                root_nodes_.push_back(root_node)
                cur_nodes_.push_back(root_node)

            elif self.status[i] == 2:
                # expanded already
                cur_node = self.cur_nodes[i][0].ppchildren[0][im_action[i]]
                node_visit(cur_node)
                root_nodes_.push_back(self.root_nodes[i])
                cur_nodes_.push_back(cur_node)      

            elif self.status[i] == 3:
                # done already
                for k in range(self.num_actions):
                    self.par_logits[k] = self.cur_nodes[i].ppchildren[0][k][0].logit
                cur_node = self.cur_nodes[i][0].ppchildren[0][im_action[i]]
                node_expand(pnode=cur_node, r=0., v=0., t=self.total_step[i], done=True,
                        logits=self.par_logits, encoded=self.cur_nodes[i][0].encoded, override=True)
                node_visit(cur_node)
                root_nodes_.push_back(self.root_nodes[i])
                cur_nodes_.push_back(cur_node)        
            
            elif self.status[i] == 4:
                # need expand
                encoded = {"gym_env_out": None, 
                           "model_ys": model_net_out_4.ys[-1,l] if self.actor_see_type >= 0 else None,
                           "model_states": dict({sk:sv[[l]] for sk, sv in model_net_out_4.state.items()})
                           }
                cur_node = self.cur_nodes[i][0].ppchildren[0][im_action[i]]
                if self.pred_done:
                    v_in = vs_4[l] if not done_4[l] else 0.
                    done_in = done_4[l]
                else:
                    v_in = vs_4[l]
                    done_in = False

                node_expand(pnode=cur_node, r=rs_4[l], 
                        v=v_in, t=self.total_step[i], done=done_in,
                        logits=logits_4[l], encoded=<PyObject*>encoded, override=True)
                node_visit(cur_node)
                root_nodes_.push_back(self.root_nodes[i])
                cur_nodes_.push_back(cur_node)   
                l += 1             
        self.root_nodes = root_nodes_
        self.cur_nodes = cur_nodes_
        if self.time: self.timings.time("compute_root_cur_nodes")

        # reset if serach depth exceeds max depth
        if self.max_allow_depth > 0:
            for i in range(self.env_n):
                if self.rollout_depth[i] >= self.max_allow_depth:
                    action[i, 2] = 1
                    reset[i] = 1

        # compute model_out        
        model_out = self.compute_model_out(action, self.status)
        gym_env_out = []
        for i in range(self.env_n):
            encoded = <dict>self.cur_nodes[i][0].encoded
            if encoded["gym_env_out"] is not None:
                gym_env_out.append(encoded["gym_env_out"].unsqueeze(0))
        if len(gym_env_out) > 0:
            gym_env_out = torch.concat(gym_env_out)
        else:
            gym_env_out = None

        if self.actor_see_type >= 0:
            model_encodes = self.compute_model_encodes(self.cur_nodes)
            if self.actor_see_double_encode:
                model_encodes_ = self.compute_model_encodes(self.root_nodes)
                model_encodes = torch.concat([model_encodes, model_encodes_], dim=1)
        else:
            model_encodes = None
        if self.debug:
            self.xs =  self.compute_model_xs(self.cur_nodes)
        if self.time: self.timings.time("compute_model_out")
        # compute reward
        j = 0
        for i in range(self.env_n):
            # real reward
            if self.status[i] == 1:
                self.full_reward[i][0] = reward[j]
            else:
                self.full_reward[i][0] = 0.
            # planning reward
            if self.im_enable:                        
                self.root_nodes_qmax_[i] = self.root_nodes[i][0].max_q
                if self.status[i] != 1:                
                    self.full_reward[i][1] = (self.root_nodes_qmax_[i] - self.root_nodes_qmax[i])*self.depth_delta[i]
                    if self.full_reward[i][1] < 0: self.full_reward[i][1] = 0
                else:
                    self.full_reward[i][1] = 0.
                self.root_nodes_qmax[i] = self.root_nodes_qmax_[i]
            if self.status[i] == 1:
                j += 1
        if self.time: self.timings.time("compute_reward")
        # compute done & full_real_done
        j = 0
        for i in range(self.env_n):
            if self.status[i] == 1:
                self.full_done[i] = done[j]
                self.full_real_done[i] = real_done[j]
                self.full_truncated_done[i] = truncated_done[j]
            else:
                self.full_done[i] = False
                self.full_real_done[i] = False
                self.full_truncated_done[i] = False
            if self.status[i] == 1:
                j += 1
        # compute reset
        for i in range(self.env_n):
            if reset[i]:
                self.rollout_depth[i] = 0
                self.cur_nodes[i] = self.root_nodes[i]
                node_visit(self.cur_nodes[i])
                self.status[i] = 5 
        # some extra info
        info = {"cur_t": torch.tensor(self.cur_t, dtype=torch.long, device=self.device),
                "max_rollout_depth":  torch.tensor(self.max_rollout_depth_, dtype=torch.long, device=self.device),
                "real_done": torch.tensor(self.full_real_done, dtype=torch.bool, device=self.device),
                "truncated_done": torch.tensor(self.full_truncated_done, dtype=torch.bool, device=self.device),}
        if self.time: self.timings.time("end")

        return ((torch.tensor(model_out, dtype=torch.float32, device=self.device), gym_env_out, model_encodes), 
                torch.tensor(self.full_reward, dtype=torch.float32, device=self.device), 
                torch.tensor(self.full_done, dtype=torch.bool, device=self.device), 
                info)
    
    cdef float[:, :] compute_model_out(self, int[:, :]& action, int[:]& status):
        cdef int i
        cdef int idx1 = self.num_actions*5+5
        cdef int idx2 = self.num_actions*10+7

        result_np = np.zeros((self.env_n, self.obs_n), dtype=np.float32)
        cdef float[:, :] result = result_np        
        for i in range(self.env_n):
            result[i, :idx1] = node_stat(self.root_nodes[i], detailed=True, mask_type=self.stat_mask_type)
            result[i, idx1:idx2] = node_stat(self.cur_nodes[i], detailed=False, mask_type=self.stat_mask_type)    
            # reset
            if action is None or status[i] == 1:
                result[i, idx2] = 1.
            else:
                result[i, idx2] = action[i, 2]
            # time
            if self.cur_t[i] < self.rep_rec_t:
                result[i, idx2+1+self.cur_t[i]] = 1.
            # deprec
            result[i, idx2+self.rep_rec_t+1] = (self.discounting ** (self.rollout_depth[i]))           
        return result

    cdef compute_model_encodes(self, vector[Node*] nodes):
        model_encodes = []
        for i in range(self.env_n):
            encoded = <dict>nodes[i][0].encoded
            model_encodes.append((encoded["model_ys"] if not nodes[i][0].done 
                else torch.zeros_like(encoded["model_ys"])).unsqueeze(0))
        model_encodes = torch.concat(model_encodes)
        return model_encodes
    
    cdef compute_model_xs(self, vector[Node*] nodes):
        xs = []
        for i in range(self.env_n):
            encoded = <dict>nodes[i][0].encoded
            if "pred_xs" not in encoded["model_states"]: return None
            xs.append((encoded["model_states"]["pred_xs"] if not nodes[i][0].done 
                else torch.zeros_like(encoded["model_states"]["pred_xs"])).unsqueeze(0))
        xs = torch.concat(xs)
        return xs

    def close(self):
        cdef int i
        if hasattr(self, "root_nodes"):
            for i in range(self.env_n):
                node_del(self.root_nodes[i], except_idx=-1)
        self.env.close()

    def seed(self, x):
        self.env.seed(x)

    def print_time(self):
        print(self.timings.summary())

    def clone_state(self):
        return self.env.clone_state()

    def restore_state(self, state):
        self.env.restore_state(state)

    def get_action_meanings(self):
        return self.env.get_action_meanings()       

cdef class cVecModelWrapper():
    """Wrap the gym environment with a model; output for each 
    step is (out, reward, done, info), where out is a tuple 
    of (gym_env_out, model_out, model_encodes) that corresponds to underlying 
    environment frame, output from the model wrapper, and encoding from the model
    Assume a perfect dynamic model.
    """
    # setting
    cdef int rec_t
    cdef int rep_rec_t
    cdef float discounting
    cdef int max_allow_depth
    cdef bool perfect_model
    cdef bool tree_carry
    cdef bool im_enable
    cdef int actor_see_type
    cdef bool actor_see_double_encode
    cdef int num_actions
    cdef int obs_n    
    cdef int env_n
    cdef bool time 
    cdef int stat_mask_type
    cdef bool debug

    # python object
    cdef object device
    cdef object env
    cdef object timings
    cdef readonly baseline_max_q
    cdef readonly baseline_mean_q    
    cdef readonly object model_out_shape
    cdef readonly object gym_env_out_shape

    # tree statistic
    cdef vector[Node*] cur_nodes
    cdef vector[Node*] root_nodes    
    cdef float[:] root_nodes_qmax
    cdef float[:] root_nodes_qmax_
    cdef int[:] rollout_depth
    cdef int[:] max_rollout_depth
    cdef int[:] cur_t

    # internal variables only used in step function
    cdef float[:] depth_delta
    cdef int[:] max_rollout_depth_
    cdef float[:] mean_q
    cdef float[:] max_q
    cdef int[:] status
    cdef vector[Node*] cur_nodes_
    cdef float[:] par_logits
    cdef float[:, :] full_reward
    cdef bool[:] full_done
    cdef bool[:] full_real_done
    cdef bool[:] full_truncated_done
    cdef int[:] total_step

    def __init__(self, env, env_n, flags, device=None, time=False, debug=False):
        assert flags.perfect_model, "this class only supports perfect model"
        self.device = torch.device("cpu") if device is None else device
        self.env = env     
        if flags.test_rec_t > 0:
            self.rec_t = flags.test_rec_t
            self.rep_rec_t = flags.rec_t         
        else:
            self.rec_t = flags.rec_t  
            self.rep_rec_t = flags.rec_t   
        self.discounting = flags.discounting
        self.max_allow_depth = flags.max_depth
        self.perfect_model = flags.perfect_model
        self.tree_carry = flags.tree_carry
        self.num_actions = env.action_space[0].n
        self.im_enable = flags.im_cost > 0.
        self.actor_see_type = flags.actor_see_type      
        self.actor_see_double_encode = False
        self.stat_mask_type = flags.stat_mask_type
        self.env_n = env_n
        self.obs_n = 9 + self.num_actions * 10 + self.rep_rec_t
        self.model_out_shape = (self.obs_n, 1, 1)
        self.gym_env_out_shape = env.observation_space.shape[1:]

        self.baseline_max_q = torch.zeros(self.env_n, dtype=torch.float32, device=self.device)
        self.baseline_mean_q = torch.zeros(self.env_n, dtype=torch.float32, device=self.device)        
        self.time = time
        self.timings = util.Timings()
        self.debug = debug

        # internal variable init.
        self.depth_delta = np.zeros(self.env_n, dtype=np.float32)
        self.max_rollout_depth_ = np.zeros(self.env_n, dtype=np.intc)
        self.mean_q =  np.zeros(self.env_n, dtype=np.float32)
        self.max_q = np.zeros(self.env_n, dtype=np.float32)
        self.status = np.zeros(self.env_n, dtype=np.intc)
        self.par_logits = np.zeros(self.num_actions, dtype=np.float32)
        self.full_reward = np.zeros((self.env_n, 1 + int(self.im_enable)), dtype=np.float32)
        self.full_done = np.zeros(self.env_n, dtype=np.bool)
        self.full_real_done = np.zeros(self.env_n, dtype=np.bool)
        self.full_truncated_done = np.zeros(self.env_n, dtype=np.bool)
        self.total_step = np.zeros(self.env_n, dtype=np.intc)
        
    def reset(self, model_net):
        """reset the environment; should only be called in the initial"""
        cdef int i
        cdef Node* root_node
        cdef Node* cur_node
        cdef float[:,:] model_out        

        with torch.no_grad():
            # some init.
            self.root_nodes_qmax = np.zeros(self.env_n, dtype=np.float32)
            self.root_nodes_qmax_ = np.zeros(self.env_n, dtype=np.float32)
            self.rollout_depth = np.zeros(self.env_n, dtype=np.intc)
            self.max_rollout_depth = np.zeros(self.env_n, dtype=np.intc)
            self.cur_t = np.zeros(self.env_n, dtype=np.intc)

            # reset obs
            obs = self.env.reset()

            # obtain output from model
            obs_py = torch.tensor(obs, dtype=torch.uint8, device=self.device)
            pass_action = torch.zeros(self.env_n, dtype=torch.long)
            model_net_out = model_net(obs_py, 
                                      pass_action.unsqueeze(0).to(self.device), 
                                      one_hot=False)  
            vs = model_net_out.vs.cpu()
            logits = model_net_out.logits.cpu()
            env_state = self.env.clone_state(inds=np.arange(self.env_n))

            # compute and update root node and current node
            for i in range(self.env_n):
                root_node = node_new(pparent=NULL, action=pass_action[i].item(), logit=0., num_actions=self.num_actions, 
                    discounting=self.discounting, rec_t=self.rec_t, remember_path=False)                
                if self.actor_see_type <= 0:
                    encoded = {"env_state": env_state[i], "gym_env_out": obs_py[i]}
                else:
                    encoded = {"env_state": env_state[i], 
                               "gym_env_out": obs_py[i], 
                               "model_ys": model_net_out.ys[-1,i]}
                node_expand(pnode=root_node, r=0., v=vs[-1, i].item(), t=self.total_step[i], done=False,
                    logits=logits[-1, i].numpy(), encoded=<PyObject*>encoded, override=False)
                node_visit(pnode=root_node)
                self.root_nodes.push_back(root_node)
                self.cur_nodes.push_back(root_node)
            
            # compute model_out
            model_out = self.compute_model_out(None, None)

            gym_env_out = []
            for i in range(self.env_n):
                encoded = <dict>self.cur_nodes[i][0].encoded
                gym_env_out.append(encoded["gym_env_out"].unsqueeze(0))
            gym_env_out = torch.concat(gym_env_out)

            if not self.actor_see_type == 0:
                model_encodes = self.compute_model_encodes(self.cur_nodes)
                if self.actor_see_double_encode:
                    model_encodes = torch.concat([model_encodes, model_encodes], dim=1)
            else:
                model_encodes = None

            # record initial root_nodes_qmax 
            for i in range(self.env_n):
                self.root_nodes_qmax[i] = self.root_nodes[i][0].max_q
            
            return torch.tensor(model_out, dtype=torch.float32, device=self.device), gym_env_out, model_encodes

    def step(self, action, model_net):  
        # action is tensor of shape (env_n, 3)
        # which corresponds to real_action, im_action, reset, term

        cdef int i, j, k
        cdef int[:] re_action
        cdef int[:] im_action
        cdef int[:] reset

        cdef Node* root_node
        cdef Node* cur_node
        cdef Node* next_node
        cdef vector[Node*] cur_nodes_
        cdef vector[Node*] root_nodes_    
        cdef float[:,:] model_out        

        cdef vector[int] pass_inds_restore
        cdef vector[int] pass_inds_step
        cdef vector[int] pass_inds_reset
        cdef vector[int] pass_inds_reset_
        cdef vector[int] pass_action

        cdef float[:] vs
        cdef float[:,:] logits

        if self.time: self.timings.reset()
        action = action.cpu().int().numpy()
        re_action, im_action, reset = action[:, 0], action[:, 1], action[:, 2]

        pass_env_states = []

        for i in range(self.env_n):            
            # compute the mask of real / imagination step                             
            self.max_rollout_depth_[i] = self.max_rollout_depth[i]
            self.depth_delta[i] = 1.
            if self.cur_t[i] < self.rec_t - 1: # imagaination step
                self.cur_t[i] += 1
                self.rollout_depth[i] += 1
                self.max_rollout_depth[i] = max(self.max_rollout_depth[i], self.rollout_depth[i])
                next_node = self.cur_nodes[i][0].ppchildren[0][im_action[i]]
                if node_expanded(next_node, -1):
                    self.status[i] = 2
                elif self.cur_nodes[i][0].done:
                    self.status[i] = 3
                else:
                    if self.status[i] != 0 or self.status[i] != 4: # no need restore if last step is real or just expanded
                        encoded = <dict> self.cur_nodes[i][0].encoded
                        pass_env_states.append(encoded["env_state"])
                        pass_inds_restore.push_back(i)
                        pass_action.push_back(im_action[i])
                        pass_inds_step.push_back(i)
                    self.status[i] = 4  
            else: # real step
                self.cur_t[i] = 0
                self.rollout_depth[i] = 0          
                self.max_rollout_depth[i] = 0
                self.total_step[i] = self.total_step[i] + 1
                # record baseline before moving on
                self.baseline_mean_q[i] = average(self.root_nodes[i][0].prollout_qs[0]) / self.discounting
                self.baseline_max_q[i] = maximum(self.root_nodes[i][0].prollout_qs[0]) / self.discounting
                encoded = <dict> self.root_nodes[i][0].encoded
                pass_env_states.append(encoded["env_state"])
                pass_inds_restore.push_back(i)
                pass_action.push_back(re_action[i])
                pass_inds_step.push_back(i)
                self.status[i] = 1                              
        if self.time: self.timings.time("misc_1")

        # restore env      
        if pass_inds_restore.size() > 0:
            self.env.restore_state(pass_env_states, inds=pass_inds_restore)

        # one step of env
        if pass_inds_step.size() > 0:
            obs, reward, done, info = self.env.step(pass_action, inds=pass_inds_step) 
            real_done = [m["real_done"] if "real_done" in m else done[n] for n, m in enumerate(info)]
            truncated_done = [m["truncated_done"] if "truncated_done" in m else False for n, m in enumerate(info)]
        if self.time: self.timings.time("step_state")

        # reset needed?
        for i, j in enumerate(pass_inds_step):
            if self.status[j] == 1 and done[i]:
                pass_inds_reset.push_back(j)
                pass_inds_reset_.push_back(i) # index within pass_inds_step
        # reset
        if pass_inds_reset.size() > 0:
            obs_reset = self.env.reset(inds=pass_inds_reset) 
            for i, j in enumerate(pass_inds_reset_):
                obs[j] = obs_reset[i]
                pass_action[j] = 0            
        if self.time: self.timings.time("misc_2")

        # use model
        if pass_inds_step.size() > 0:
            with torch.no_grad():
                obs_py = torch.tensor(obs, dtype=torch.uint8, device=self.device)
                model_net_out = model_net(obs_py, 
                        torch.tensor(pass_action, dtype=long, device=self.device).unsqueeze(0), 
                        one_hot=False)  
                model_ys = model_net_out.ys
            vs = model_net_out.vs[-1].float().cpu().numpy()
            logits = model_net_out.logits[-1].float().cpu().numpy()
            if self.time: self.timings.time("model")
            env_state = self.env.clone_state(inds=pass_inds_step)   
            if self.time: self.timings.time("clone_state")

        # compute the current and root nodes
        j = 0
        for i in range(self.env_n):
            if self.status[i] == 1:
                # real transition
                new_root = (not self.tree_carry or 
                    not node_expanded(self.root_nodes[i][0].ppchildren[0][re_action[i]], -1) or done[j])
                if new_root:
                    root_node = node_new(pparent=NULL, action=pass_action[j], logit=0., num_actions=self.num_actions, 
                        discounting=self.discounting, rec_t=self.rec_t, remember_path=False)
                    if self.actor_see_type <= 0:
                        encoded = {"env_state": env_state[j], "gym_env_out": obs_py[j]}
                    else:
                        encoded = {"env_state": env_state[j], "gym_env_out": obs_py[j], "model_ys": model_ys[-1,j]}
                    node_expand(pnode=root_node, r=0., v=vs[j], t=self.total_step[i], done=False,
                        logits=logits[j], encoded=<PyObject*>encoded, override=False)
                    node_del(self.root_nodes[i], except_idx=-1)
                    node_visit(root_node)
                else:
                    root_node = self.root_nodes[i][0].ppchildren[0][re_action[i]]
                    if self.actor_see_type <= 0:
                        encoded = {"env_state": env_state[j], "gym_env_out": obs_py[j]}
                    else:
                        encoded = {"env_state": env_state[j], "gym_env_out": obs_py[j], "model_ys": model_ys[-1,j]}
                    node_expand(pnode=root_node, r=0., v=vs[j], t=self.total_step[i], done=False,
                        logits=logits[j], encoded=<PyObject*>encoded, override=True)                        
                    node_del(self.root_nodes[i], except_idx=re_action[i])
                    node_visit(root_node)
                    
                j += 1
                root_nodes_.push_back(root_node)
                cur_nodes_.push_back(root_node)

            elif self.status[i] == 2:
                # expanded already
                cur_node = self.cur_nodes[i][0].ppchildren[0][im_action[i]]
                node_visit(cur_node)
                root_nodes_.push_back(self.root_nodes[i])
                cur_nodes_.push_back(cur_node)    

            elif self.status[i] == 3:
                # done already
                for k in range(self.num_actions):
                    self.par_logits[k] = self.cur_nodes[i].ppchildren[0][k][0].logit
                cur_node = self.cur_nodes[i][0].ppchildren[0][im_action[i]]
                node_expand(pnode=cur_node, r=0., v=0., t=self.total_step[i], done=True,
                        logits=self.par_logits, encoded=self.cur_nodes[i][0].encoded, override=False)
                node_visit(cur_node)
                root_nodes_.push_back(self.root_nodes[i])
                cur_nodes_.push_back(cur_node)              
            
            elif self.status[i] == 4:
                # need expand
                if self.actor_see_type <= 0:
                    encoded = {"env_state": env_state[j], "gym_env_out": obs_py[j]}
                else:
                    encoded = {"env_state": env_state[j], "gym_env_out": obs_py[j], "model_ys": model_ys[-1,j]}
                cur_node = self.cur_nodes[i][0].ppchildren[0][im_action[i]]
                node_expand(pnode=cur_node, r=reward[j], v=vs[j] if not done[j] else 0., t=self.total_step[i], done=done[j],
                        logits=logits[j], encoded=<PyObject*>encoded, override=False)
                node_visit(cur_node)
                root_nodes_.push_back(self.root_nodes[i])
                cur_nodes_.push_back(cur_node)   
                j += 1                            

        self.root_nodes = root_nodes_
        self.cur_nodes = cur_nodes_
        if self.time: self.timings.time("compute_root_cur_nodes")

        # reset if serach depth exceeds max depth
        if self.max_allow_depth > 0:
            for i in range(self.env_n):
                if self.rollout_depth[i] >= self.max_allow_depth:
                    action[i, 2] = 1
                    reset[i] = 1

        # compute model_out        
        model_out = self.compute_model_out(action, self.status)

        gym_env_out = []
        for i in range(self.env_n):
            encoded = <dict>self.cur_nodes[i][0].encoded
            gym_env_out.append(encoded["gym_env_out"].unsqueeze(0))
        gym_env_out = torch.concat(gym_env_out)

        if self.actor_see_type > 0:
            model_encodes = self.compute_model_encodes(self.cur_nodes)
            if self.actor_see_double_encode:
                model_encodes_ = self.compute_model_encodes(self.root_nodes)
                model_encodes = torch.concat([model_encodes, model_encodes_], dim=1)
        else:
            model_encodes = None

        if self.time: self.timings.time("compute_model_out")

        # compute reward
        j = 0
        for i in range(self.env_n):
            if self.status[i] == 1:
                self.full_reward[i][0] = reward[j]
            else:
                self.full_reward[i][0] = 0.
            if self.im_enable:                        
                self.root_nodes_qmax_[i] = self.root_nodes[i][0].max_q
                if self.status[i] != 1:                
                    self.full_reward[i][1] = (self.root_nodes_qmax_[i] - self.root_nodes_qmax[i])*self.depth_delta[i]
                else:
                    self.full_reward[i][1] = 0.
                self.root_nodes_qmax[i] = self.root_nodes_qmax_[i]
            if self.status[i] == 1 or self.status[i] == 4:
                j += 1
        if self.time: self.timings.time("compute_reward")

        # compute done & full_real_done
        j = 0
        for i in range(self.env_n):
            if self.status[i] == 1:
                self.full_done[i] = done[j]
                self.full_real_done[i] = real_done[j]
                self.full_truncated_done[i] = truncated_done[j]
            else:
                self.full_done[i] = False
                self.full_real_done[i] = False
                self.full_truncated_done[i] = False
            if self.status[i] == 1 or self.status[i] == 4:
                j += 1

        # compute reset
        for i in range(self.env_n):
            if reset[i]:
                self.rollout_depth[i] = 0
                self.cur_nodes[i] = self.root_nodes[i]
                node_visit(self.cur_nodes[i])
                self.status[i] = 5 # need to restore state on the next transition, so we need to alter the status from 4
        
        # some extra info
        info = {"cur_t": torch.tensor(self.cur_t, dtype=torch.long, device=self.device),
                "max_rollout_depth":  torch.tensor(self.max_rollout_depth_, dtype=torch.long, device=self.device),
                "real_done": torch.tensor(self.full_real_done, dtype=torch.bool, device=self.device),
                "truncated_done": torch.tensor(self.full_truncated_done, dtype=torch.bool, device=self.device)}
        if self.time: self.timings.time("end")

        return ((torch.tensor(model_out, dtype=torch.float32, device=self.device), gym_env_out, model_encodes), 
                torch.tensor(self.full_reward, dtype=torch.float32, device=self.device), 
                torch.tensor(self.full_done, dtype=torch.bool, device=self.device), 
                info)

    
    cdef float[:, :] compute_model_out(self, int[:, :]& action, int[:]& status):
        cdef int i
        cdef int idx1 = self.num_actions*5+5
        cdef int idx2 = self.num_actions*10+7

        result_np = np.zeros((self.env_n, self.obs_n), dtype=np.float32)
        cdef float[:, :] result = result_np        
        for i in range(self.env_n):
            result[i, :idx1] = node_stat(self.root_nodes[i], detailed=True, mask_type=self.stat_mask_type)
            result[i, idx1:idx2] = node_stat(self.cur_nodes[i], detailed=False, mask_type=self.stat_mask_type)    
            # reset
            if action is None or status[i] == 1:
                result[i, idx2] = 1.
            else:
                result[i, idx2] = action[i, 2]
            # time
            if self.cur_t[i] < self.rep_rec_t:
                result[i, idx2+1+self.cur_t[i]] = 1.
            # deprec
            result[i, idx2+self.rec_t+1] = (self.discounting ** (self.rollout_depth[i]))           
        return result

    cdef compute_model_encodes(self, vector[Node*] nodes):
        model_encodes = []
        for i in range(self.env_n):
            encoded = <dict>nodes[i][0].encoded
            model_encodes.append(encoded["model_ys"].unsqueeze(0))
        model_encodes = torch.concat(model_encodes)
        return model_encodes

    def close(self):
        cdef int i
        if hasattr(self, "root_nodes"):
            for i in range(self.env_n):
                node_del(self.root_nodes[i], except_idx=-1)
        self.env.close()

    def seed(self, x):
        self.env.seed(x)

    def print_time(self):
        print(self.timings.summary())

    def clone_state(self):
        return self.env.clone_state()

    def restore_state(self, state):
        self.env.restore_state(state)

    def get_action_meanings(self):
        return self.env.get_action_meanings()       
