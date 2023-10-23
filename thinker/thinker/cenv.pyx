# distutils: language = c++
import numpy as np
import gym
from gym import spaces
import torch
import thinker.util as util

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
cdef float[:] node_stat(Node* pnode, bool detailed, int enc_type, int mask_type):
    cdef float[:] result = np.zeros((pnode[0].num_actions*5+5) if detailed else (pnode[0].num_actions*5+2), dtype=np.float32) 
    cdef int i
    result[pnode[0].action] = 1. # action
    f = lambda x:x if enc_type == 0 else enc
    result[pnode[0].num_actions] = f(pnode[0].r) # reward
    if not mask_type in [2]: 
        result[pnode[0].num_actions+1] = f(pnode[0].v) # value
    for i in range(int(pnode[0].ppchildren[0].size())):
        child = pnode[0].ppchildren[0][i][0]
        if not mask_type in [2]: 
            result[pnode[0].num_actions+2+i] = child.logit # child_logits
        if not mask_type in [1, 2]: 
            result[pnode[0].num_actions*2+2+i] = f(average(child.prollout_qs[0])) # child_rollout_qs_mean
            result[pnode[0].num_actions*3+2+i] = f(maximum(child.prollout_qs[0])) # child_rollout_qs_max
            result[pnode[0].num_actions*4+2+i] = child.rollout_n / <float>pnode[0].rec_t # child_rollout_ns_enc
    pnode[0].max_q = (maximum(pnode[0].prollout_qs[0]) - pnode[0].r) / pnode[0].discounting
    if detailed and not mask_type in [1, 2]:        
        result[pnode[0].num_actions*5+2] = f(pnode[0].trail_r / pnode[0].discounting)
        result[pnode[0].num_actions*5+3] = f(pnode[0].rollout_q / pnode[0].discounting)
        result[pnode[0].num_actions*5+4] = f(pnode[0].max_q)
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

cdef float enc(float x):
    return sign(x)*(sqrt(abs(x)+1)-1)+(0.001)*x

cdef float sign(float x):
    if x > 0.: return 1.
    if x < 0.: return -1.
    return 0.

cdef float abs(float x):
    if x > 0.: return x
    if x < 0.: return -x
    return 0.

cdef class cModelWrapper():
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
    cdef int max_depth    
    cdef bool tree_carry    

    cdef int enc_type
    cdef bool pred_done
    cdef int num_actions
    cdef int obs_n    
    cdef int env_n
    
    cdef bool return_h
    cdef bool return_x
    cdef bool return_double

    cdef bool im_enable
    cdef bool cur_enable
    cdef float cur_reward_cost
    cdef float cur_v_cost
    cdef float cur_enc_cost
    cdef bool cur_done_gate
    
    cdef int stat_mask_type
    cdef bool time 

    # python object
    cdef object device
    cdef object env
    cdef object timings
    cdef object real_states
    cdef object baseline_mean_q    

    cdef readonly object observation_space
    cdef readonly object action_space
    cdef readonly object reward_range
    cdef readonly object metadata

    # tree statistic
    cdef vector[Node*] cur_nodes
    cdef vector[Node*] root_nodes    
    cdef float[:] root_nodes_qmax
    cdef float[:] root_nodes_qmax_
    cdef int[:] rollout_depth
    cdef int[:] max_rollout_depth
    cdef int[:] cur_t

    # internal variables only used in step function    
    cdef int[:] max_rollout_depth_
    cdef float[:] mean_q
    cdef float[:] max_q
    cdef int[:] status
    cdef vector[Node*] cur_nodes_
    cdef float[:] par_logits
    cdef float[:] full_reward
    cdef float[:] full_im_reward
    cdef float[:] full_cur_reward
    cdef bool[:] full_done
    cdef bool[:] full_im_done
    cdef bool[:] full_real_done
    cdef bool[:] full_truncated_done
    cdef int[:] step_status
    cdef int[:] total_step  

    def __init__(self, env, env_n, flags, model_net, device=None, time=False):        
           
        if flags.test_rec_t > 0:
            self.rec_t = flags.test_rec_t
            self.rep_rec_t = flags.rec_t         
        else:
            self.rec_t = flags.rec_t           
            self.rep_rec_t = flags.rec_t         
        self.discounting = flags.discounting
        self.max_depth = flags.max_depth
        self.tree_carry = flags.tree_carry

        self.enc_type = flags.model_enc_type
        self.pred_done = flags.model_done_loss_cost > 0.        
        self.num_actions = env.action_space[0].n
        self.obs_n = 9 + self.num_actions * 10 + self.rep_rec_t
        self.env_n = env_n

        self.return_h = flags.return_h  
        self.return_x = flags.return_x  
        self.return_double = flags.return_double

        self.im_enable = flags.im_enable
        self.cur_enable = flags.cur_enable
        self.cur_reward_cost = flags.cur_reward_cost
        self.cur_v_cost = flags.cur_v_cost
        self.cur_enc_cost = flags.cur_enc_cost
        self.cur_done_gate = flags.cur_done_gate

        self.stat_mask_type = flags.stat_mask_type
        self.time = time        
        
        self.device = torch.device("cpu") if device is None else device        
        self.env = env  
        self.timings = util.Timings()        

        self.observation_space = {
            "tree_reps": spaces.Box(low=-np.inf, high=np.inf, shape=(self.env_n, self.obs_n), 
                dtype=np.float32),
            "real_states": self.env.observation_space,
        }

        if self.return_x:
            xs_shape = list(self.env.observation_space.shape)
            if self.return_double:
                xs_shape[1] *= 2
            xs_space = spaces.Box(low=0., high=1., shape=xs_shape, dtype=np.float32)
            self.observation_space["xs"] = xs_space

        if self.return_h:
            hs_shape = [env_n,] + list(model_net.hidden_shape)
            if self.return_double:
                hs_shape[1] *= 2
            hs_space = spaces.Box(low=-np.inf, high=np.inf, shape=hs_shape, dtype=np.float32)
            self.observation_space["hs"] = hs_space

        self.observation_space = spaces.Dict(self.observation_space)

        aug_action_space = spaces.Tuple((spaces.Discrete(self.num_actions),)*self.env_n)
        reset_space = spaces.Tuple((spaces.Discrete(2),)*self.env_n)
        self.action_space = spaces.Tuple((aug_action_space, reset_space))
        self.reward_range = env.reward_range
        self.metadata = env.metadata

        # internal variable init.
        self.max_rollout_depth_ = np.zeros(self.env_n, dtype=np.intc)
        self.mean_q =  np.zeros(self.env_n, dtype=np.float32)
        self.max_q = np.zeros(self.env_n, dtype=np.float32)
        self.status = np.zeros(self.env_n, dtype=np.intc)
        self.par_logits = np.zeros(self.num_actions, dtype=np.float32)
        self.full_reward = np.zeros(self.env_n, dtype=np.float32)
        self.full_im_reward = np.zeros(self.env_n, dtype=np.float32)
        self.full_cur_reward = np.zeros(self.env_n, dtype=np.float32)
        self.full_done = np.zeros(self.env_n, dtype=np.bool_)
        self.full_im_done = np.zeros(self.env_n, dtype=np.bool_)
        self.full_real_done = np.zeros(self.env_n, dtype=np.bool_)
        self.full_truncated_done = np.zeros(self.env_n, dtype=np.bool_)
        self.total_step = np.zeros(self.env_n, dtype=np.intc)
        self.step_status = np.zeros(self.env_n, dtype=np.intc)       
        
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
            self.baseline_mean_q = torch.zeros(self.env_n, dtype=torch.float, device=self.device)        

            # reset obs
            obs = self.env.reset()

            # obtain output from model
            obs_py = torch.tensor(obs, dtype=torch.uint8, device=self.device)
            pass_action = torch.zeros(self.env_n, dtype=torch.long)
            model_net_out = model_net(obs_py, 
                                      pass_action.unsqueeze(0).to(self.device), 
                                      one_hot=False,
                                      ret_xs=self.return_x,
                                      ret_zs=False,
                                      ret_hs=self.return_h)  
            vs = model_net_out.vs.cpu()
            logits = model_net_out.logits.cpu()      

            # compute and update root node and current node
            for i in range(self.env_n):
                root_node = node_new(pparent=NULL, action=pass_action[i].item(), logit=0., num_actions=self.num_actions, 
                    discounting=self.discounting, rec_t=self.rec_t, remember_path=True)                
                encoded = {"real_states": obs_py[i],
                           "xs": model_net_out.xs[-1, i] if model_net_out.xs is not None else None,
                           "hs": model_net_out.hs[-1, i] if model_net_out.hs is not None else None,
                           "model_states": dict({sk:sv[[i]] for sk, sv in model_net_out.state.items()})
                          }  
                node_expand(pnode=root_node, r=0., v=vs[-1, i].item(), t=self.total_step[i], done=False,
                    logits=logits[-1, i].numpy(), encoded=<PyObject*>encoded, override=False)
                node_visit(pnode=root_node)
                self.root_nodes.push_back(root_node)
                self.cur_nodes.push_back(root_node)
            
            # record initial root_nodes_qmax 
            for i in range(self.env_n):
                self.root_nodes_qmax[i] = self.root_nodes[i][0].max_q
            
            states = self.prepare_state(None, None)

            return states

    def step(self, action, model_net):  
        
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

        assert type(action) is tuple and len(action) == 2, \
            "action should be a tuple of size 2, containing augmented action and reset"        
        for i in range(len(action)):
            a = action[i]            
            assert torch.is_tensor(a) or \
                isinstance(a, np.ndarray) or \
                isinstance(a, list), \
                f"action[{i}] should be either torch.tensor or np.ndarray or list"            
            if torch.is_tensor(a):
                a = a.detach().cpu().numpy()
            if isinstance(a, list):
                a = np.array(a, dtype=np.int32)
            assert a.shape == (self.env_n,), \
                f"action[{i}] shape should be {(self.env_n,)}, not {a.shape}"
            if a.dtype != np.int32:
                a = a.astype(np.int32)
            if i == 0:
                re_action = a
                im_action = a
                assert (a >= 0).all() and (a < self.num_actions).all(), \
                    f"primiary action should be in [0, {self.num_actions-1}], not {a}"
            else:
                reset = a       
                assert (a >= 0).all() and (a < 2).all(), \
                    f"reset action should be in [0, 1], not {a}"  

        pass_model_states, pass_re_model_states = [], []

        for i in range(self.env_n):            
            # compute the mask of real / imagination step                             
            self.max_rollout_depth_[i] = self.max_rollout_depth[i]            
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
                pass_action_py = torch.tensor(pass_action, dtype=long, device=self.device).unsqueeze(0)
                model_net_out_1 = model_net(obs_py, 
                        pass_action_py, 
                        one_hot=False,
                        ret_xs=self.return_x,
                        ret_zs=self.cur_enable,
                        ret_hs=self.return_h)  
                if self.cur_enable:
                    pass_re_model_states = dict({msd: torch.concat([ms[msd] for ms in pass_re_model_states], dim=0)
                        for msd in pass_re_model_states[0].keys()})
                    pred_model_net_out_1 = model_net.forward_single(
                        state=pass_re_model_states,
                        action=torch.tensor(pass_action, dtype=long, device=self.device),  
                        one_hot=False,  
                        ret_xs=False,
                        ret_zs=True,
                        ret_hs=False,
                    )                                       
                    cur_reward = 0.
                    done_mask = torch.tensor(done, dtype=torch.float, device=self.device)
                    if self.cur_v_cost > 0.:
                        pred_model_net_out_1.vs[pred_model_net_out_1.dones] = 0.
                        cur_reward += self.cur_v_cost * torch.square(model_net_out_1.vs[-1] * (1-done_mask) - pred_model_net_out_1.vs[-1])
                    if self.cur_reward_cost > 0.:
                        pred_model_net_out_1.rs[pred_model_net_out_1.dones] = 0.
                        cur_reward += self.cur_reward_cost * torch.square(torch.tensor(reward, device=self.device) * (1-done_mask) - pred_model_net_out_1.rs[-1])
                    if self.cur_enc_cost > 0.:
                        flat_dim = tuple(range(1, len(model_net_out_1.zs[-1].shape)))     
                        done_mask_ = done_mask
                        for _ in range(len(model_net_out_1.zs.shape)-2): done_mask_ = done_mask_.unsqueeze(-1)
                        pred_model_net_out_1.zs[pred_model_net_out_1.dones] = 0.     
                        cur_reward += self.cur_enc_cost * torch.mean(torch.square(model_net_out_1.zs[-1] * (1-done_mask_) - pred_model_net_out_1.zs[-1]), dim=flat_dim)
                    if self.cur_done_gate:
                        done_mask = torch.tensor(done, dtype=torch.bool, device=self.device)
                        cur_reward[torch.logical_or(done_mask, pred_model_net_out_1.dones[-1])] = 0.
                    cur_reward = cur_reward.float().cpu().numpy()                    
                    
            vs_1 = model_net_out_1.vs[-1].float().cpu().numpy()
            logits_1 = model_net_out_1.logits[-1].float().cpu().numpy()
                
        if self.time: self.timings.time("misc_2")
        # use model for status 4 transition (imagination transition)
        if pass_model_action.size() > 0:
            with torch.no_grad():
                pass_model_states = dict({msd: torch.concat([ms[msd] for ms in pass_model_states], dim=0)
                        for msd in pass_model_states[0].keys()})
                    # all batch index are in the first index 
                pass_model_action_py = torch.tensor(pass_model_action, dtype=long, device=self.device)
                model_net_out_4 = model_net.forward_single(
                    state=pass_model_states,
                    action=pass_model_action_py,                     
                    one_hot=False,
                    ret_xs=self.return_x,
                    ret_zs=False,
                    ret_hs=self.return_h)  
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
                encoded = {"real_states": obs_py[j], 
                           "xs": model_net_out_1.xs[-1, j] if model_net_out_1.xs is not None else None,
                           "hs": model_net_out_1.hs[-1, j] if model_net_out_1.hs is not None else None,
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
                encoded = {"real_states": None, 
                           "xs": model_net_out_4.xs[-1, l] if model_net_out_4.xs is not None else None,
                           "hs": model_net_out_4.hs[-1, l] if model_net_out_4.hs is not None else None,
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
        if self.max_depth > 0:
            for i in range(self.env_n):
                if self.rollout_depth[i] >= self.max_depth:
                    reset[i] = 1

        # compute states        
        states = self.prepare_state(reset, self.status)
        if self.time: self.timings.time("compute_state")
        # compute reward
        j = 0
        for i in range(self.env_n):
            # real reward
            if self.status[i] == 1:
                self.full_reward[i] = reward[j]
            else:
                self.full_reward[i] = 0.
            # planning reward
            if self.im_enable:                        
                self.root_nodes_qmax_[i] = self.root_nodes[i][0].max_q
                if self.status[i] != 1:                
                    self.full_im_reward[i] = (self.root_nodes_qmax_[i] - self.root_nodes_qmax[i])
                    if self.full_im_reward[i] < 0: self.full_im_reward[i] = 0
                else:
                    self.full_im_reward[i] = 0.
                self.root_nodes_qmax[i] = self.root_nodes_qmax_[i]
            # curisotiy reward
            if self.cur_enable:
                if self.status[i] == 1: 
                    self.full_cur_reward[i] = cur_reward[j]
                else:                    
                    self.full_cur_reward[i] = 0
            if self.status[i] == 1:
                j += 1
        if self.time: self.timings.time("compute_reward")
        
        j = 0
        for i in range(self.env_n):
            # compute done & full_real_done
            if self.status[i] == 1:
                self.full_done[i] = done[j]
                self.full_real_done[i] = real_done[j]
                self.full_truncated_done[i] = truncated_done[j]
                self.full_im_done[i] = False
            else:
                self.full_done[i] = False
                self.full_real_done[i] = False
                self.full_truncated_done[i] = False
                self.full_im_done[i] = self.cur_nodes[i][0].done
            if self.status[i] == 1:
                j += 1

            # compute reset        
            if reset[i]:
                self.rollout_depth[i] = 0
                self.cur_nodes[i] = self.root_nodes[i]
                node_visit(self.cur_nodes[i])
                self.status[i] = 5 

            # compute step status
            if self.cur_t[i] == 0:
                self.step_status[i] = 0 # real action just taken
            elif self.cur_t[i] < self.rec_t - 1:
                self.step_status[i] = 1 # im action just taken
            elif self.cur_t[i] >= self.rec_t - 1:
                self.step_status[i] = 2 # im action just taken; next action is real action

        # some extra info
        info = {
                "real_done": torch.tensor(self.full_real_done, dtype=torch.float, device=self.device).bool(),
                "truncated_done": torch.tensor(self.full_truncated_done, dtype=torch.float, device=self.device).bool(),
                #"im_done": torch.tensor(self.full_im_done, dtype=torch.float, device=self.device).bool(),
                "step_status": torch.tensor(self.step_status, device=self.device),
                "max_rollout_depth":  torch.tensor(self.max_rollout_depth_, dtype=torch.long, device=self.device),
                "baseline": self.baseline_mean_q,                
                }
        if self.im_enable:
            info["im_reward"] = torch.tensor(self.full_im_reward, dtype=torch.float, device=self.device)
        if self.cur_enable:
            info["cur_reward"] = torch.tensor(self.full_im_reward, dtype=torch.float, device=self.device)
        if self.time: self.timings.time("end")

        return (states, 
                torch.tensor(self.full_reward, dtype=torch.float, device=self.device), 
                torch.tensor(self.full_done, dtype=torch.float, device=self.device).bool(), 
                info)
    
    cdef float[:, :] compute_tree_reps(self, int[:]& reset, int[:]& status):
        cdef int i
        cdef int idx1 
        cdef int idx2 
        cdef float[:, :] result
        idx1 = self.num_actions*5+5
        idx2 = self.num_actions*10+7
        result = np.zeros((self.env_n, self.obs_n), dtype=np.float32)
        for i in range(self.env_n):
            result[i, :idx1] = node_stat(self.root_nodes[i], detailed=True, enc_type=self.enc_type, mask_type=self.stat_mask_type)
            result[i, idx1:idx2] = node_stat(self.cur_nodes[i], detailed=False, enc_type=self.enc_type, mask_type=self.stat_mask_type)    
            # reset
            if reset is None or status[i] == 1:
                result[i, idx2] = 1.
            else:
                result[i, idx2] = reset[i]
            # time
            if self.cur_t[i] < self.rep_rec_t:
                result[i, idx2+1+self.cur_t[i]] = 1.
            # deprec
            result[i, idx2+self.rep_rec_t+1] = (self.discounting ** (self.rollout_depth[i]))           
        return result
    
    cdef compute_model_out(self, vector[Node*] nodes, key):
        outs = []
        for i in range(self.env_n):
            encoded = <dict>nodes[i][0].encoded
            if key not in encoded or encoded[key] is None: 
                return None
            outs.append((encoded[key] if not nodes[i][0].done 
                else torch.zeros_like(encoded[key])).unsqueeze(0))
        outs = torch.concat(outs)
        return outs

    cdef prepare_state(self, int[:]& reset, int[:]& status):
        cdef int i

        tree_reps = self.compute_tree_reps(reset, status)
        tree_reps = torch.tensor(tree_reps, dtype=torch.float, device=self.device)

        if status is None or np.any(np.array(status)==1):
            self.real_states = self.compute_model_out(self.root_nodes, "real_states")

        states = {
            "tree_reps": tree_reps,
            "real_states": self.real_states,
        }

        if self.return_x:
            xs = self.compute_model_out(self.cur_nodes, "xs")
            if self.return_double and xs is not None:
                root_xs = self.compute_model_out(self.root_nodes, "xs")
                xs = torch.concat([root_xs, xs], dim=1)        
            assert xs is not None, "xs cannot be None"
            states["xs"] = xs

        if self.return_h:
            hs = self.compute_model_out(self.cur_nodes, "hs")
            if self.return_double and hs is not None:
                root_hs = self.compute_model_out(self.root_nodes, "hs")
                hs = torch.concat([root_hs, hs], dim=1)
            assert hs is not None, "hs cannot be None"
            states["hs"] = hs

        return states

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

cdef class cPerfectWrapper():
    """Wrap the gym environment with a perfect model (i.e. env that supports clone_state
    and restore_state); output for each step is (out, reward, done, info), where out is a tuple 
    of (gym_env_out, model_out, model_encodes) that corresponds to underlying 
    environment frame, output from the model wrapper, and encoding from the model
    Assume a perfect dynamic model.
    """
    # setting
    cdef int rec_t
    cdef int rep_rec_t
    cdef float discounting
    cdef int max_depth    
    cdef bool tree_carry    

    cdef int enc_type
    cdef int num_actions
    cdef int obs_n    
    cdef int env_n
    
    cdef bool return_h
    cdef bool return_x
    cdef bool return_double

    cdef bool im_enable
    cdef bool cur_enable
    cdef float cur_reward_cost
    cdef float cur_v_cost
    cdef float cur_enc_cost
    cdef bool cur_done_gate
    
    cdef int stat_mask_type
    cdef bool time 

    cdef float reward_clip

    # python object
    cdef object device
    cdef object env
    cdef object timings
    cdef object real_states
    cdef object baseline_mean_q    

    cdef readonly object observation_space
    cdef readonly object action_space
    cdef readonly object reward_range
    cdef readonly object metadata

    # tree statistic
    cdef vector[Node*] cur_nodes
    cdef vector[Node*] root_nodes    
    cdef float[:] root_nodes_qmax
    cdef float[:] root_nodes_qmax_
    cdef int[:] rollout_depth
    cdef int[:] max_rollout_depth
    cdef int[:] cur_t

    # internal variables only used in step function
    cdef int[:] max_rollout_depth_
    cdef float[:] mean_q
    cdef float[:] max_q
    cdef int[:] status
    cdef vector[Node*] cur_nodes_
    cdef float[:] par_logits
    cdef float[:] full_reward
    cdef float[:] full_im_reward
    cdef float[:] full_cur_reward
    cdef bool[:] full_done
    cdef bool[:] full_real_done
    cdef bool[:] full_truncated_done
    cdef bool[:] full_im_done
    cdef int[:] step_status
    cdef int[:] total_step

    def __init__(self, env, env_n, flags, model_net, device=None, time=False):
        if flags.test_rec_t > 0:
            self.rec_t = flags.test_rec_t
            self.rep_rec_t = flags.rec_t         
        else:
            self.rec_t = flags.rec_t           
            self.rep_rec_t = flags.rec_t         
        self.discounting = flags.discounting
        self.max_depth = flags.max_depth
        self.tree_carry = flags.tree_carry

        self.enc_type = flags.model_enc_type       
        self.num_actions = env.action_space[0].n
        self.obs_n = 9 + self.num_actions * 10 + self.rep_rec_t
        self.env_n = env_n

        self.return_h = flags.return_h  
        self.return_x = flags.return_x  
        self.return_double = flags.return_double

        self.im_enable = flags.im_enable
        self.cur_enable = flags.cur_enable
        self.cur_reward_cost = flags.cur_reward_cost
        self.cur_v_cost = flags.cur_v_cost
        self.cur_enc_cost = flags.cur_enc_cost
        self.cur_done_gate = flags.cur_done_gate
        
        self.stat_mask_type = flags.stat_mask_type
        self.time = time        

        self.reward_clip = flags.reward_clip
        
        self.device = torch.device("cpu") if device is None else device        
        self.env = env  
        self.timings = util.Timings()
        
        self.observation_space = {
            "tree_reps": spaces.Box(low=-np.inf, high=np.inf, shape=(self.env_n, self.obs_n), 
                dtype=np.float32),
            "real_states": self.env.observation_space,
        }

        if self.return_x:
            xs_shape = list(self.env.observation_space.shape)
            if self.return_double:
                xs_shape[1] *= 2
            xs_space = spaces.Box(low=0., high=1., shape=xs_shape, dtype=np.float32)
            self.observation_space["xs"] = xs_space

        if self.return_h:
            hs_shape = [env_n,] + list(model_net.hidden_shape)
            if self.return_double:
                hs_shape[1] *= 2
            hs_space = spaces.Box(low=-np.inf, high=np.inf, shape=hs_shape, dtype=np.float32)
            self.observation_space["hs"] = hs_space

        self.observation_space = spaces.Dict(self.observation_space)

        aug_action_space = spaces.Tuple((spaces.Discrete(self.num_actions),)*self.env_n)
        reset_space = spaces.Tuple((spaces.Discrete(2),)*self.env_n)
        self.action_space = spaces.Tuple((aug_action_space, reset_space))
        self.reward_range = env.reward_range
        self.metadata = env.metadata

        # internal variable init.
        self.max_rollout_depth_ = np.zeros(self.env_n, dtype=np.intc)
        self.mean_q =  np.zeros(self.env_n, dtype=np.float32)
        self.max_q = np.zeros(self.env_n, dtype=np.float32)
        self.status = np.zeros(self.env_n, dtype=np.intc)
        self.par_logits = np.zeros(self.num_actions, dtype=np.float32)
        self.full_reward = np.zeros(self.env_n, dtype=np.float32)
        self.full_im_reward = np.zeros(self.env_n, dtype=np.float32)
        self.full_cur_reward = np.zeros(self.env_n, dtype=np.float32)
        self.full_done = np.zeros(self.env_n, dtype=np.bool_)
        self.full_real_done = np.zeros(self.env_n, dtype=np.bool_)
        self.full_truncated_done = np.zeros(self.env_n, dtype=np.bool_)
        self.full_im_done = np.zeros(self.env_n, dtype=np.bool_)
        self.total_step = np.zeros(self.env_n, dtype=np.intc)
        self.step_status = np.zeros(self.env_n, dtype=np.intc)
        
    def reset(self, model_net):
        """reset the environment; should only be called in the initial"""
        cdef int i
        cdef Node* root_node
        cdef Node* cur_node      

        with torch.no_grad():
            # some init.
            self.root_nodes_qmax = np.zeros(self.env_n, dtype=np.float32)
            self.root_nodes_qmax_ = np.zeros(self.env_n, dtype=np.float32)
            self.rollout_depth = np.zeros(self.env_n, dtype=np.intc)
            self.max_rollout_depth = np.zeros(self.env_n, dtype=np.intc)
            self.cur_t = np.zeros(self.env_n, dtype=np.intc)
            self.baseline_mean_q = torch.zeros(self.env_n, dtype=torch.float, device=self.device)        

            # reset obs
            obs = self.env.reset()

            # obtain output from model
            obs_py = torch.tensor(obs, dtype=torch.uint8, device=self.device)
            pass_action = torch.zeros(self.env_n, dtype=torch.long)
            model_net_out = model_net(obs_py, 
                                      pass_action.unsqueeze(0).to(self.device), 
                                      one_hot=False,
                                      ret_xs=self.return_x,
                                      ret_zs=False,
                                      ret_hs=self.return_h)  
            vs = model_net_out.vs.cpu()
            logits = model_net_out.logits.cpu()
            env_state = self.env.clone_state(inds=np.arange(self.env_n))

            # compute and update root node and current node
            for i in range(self.env_n):
                root_node = node_new(pparent=NULL, action=pass_action[i].item(), logit=0., num_actions=self.num_actions, 
                    discounting=self.discounting, rec_t=self.rec_t, remember_path=False)                
                encoded = {"real_states": obs_py[i],
                           "xs": model_net_out.xs[-1, i] if model_net_out.xs is not None else None,
                           "hs": model_net_out.hs[-1, i] if model_net_out.hs is not None else None,
                           "env_states": env_state[i],
                           }
                node_expand(pnode=root_node, r=0., v=vs[-1, i].item(), t=self.total_step[i], done=False,
                    logits=logits[-1, i].numpy(), encoded=<PyObject*>encoded, override=False)
                node_visit(pnode=root_node)
                self.root_nodes.push_back(root_node)
                self.cur_nodes.push_back(root_node)
            
            # record initial root_nodes_qmax 
            for i in range(self.env_n):
                self.root_nodes_qmax[i] = self.root_nodes[i][0].max_q

            states = self.prepare_state(None, None)            
            return states

    def step(self, action, model_net):  

        cdef int i, j, k
        cdef int[:] re_action
        cdef int[:] im_action
        cdef int[:] reset

        cdef Node* root_node
        cdef Node* cur_node
        cdef Node* next_node
        cdef vector[Node*] cur_nodes_
        cdef vector[Node*] root_nodes_          

        cdef vector[int] pass_inds_restore
        cdef vector[int] pass_inds_step
        cdef vector[int] pass_inds_reset
        cdef vector[int] pass_inds_reset_
        cdef vector[int] pass_action

        cdef float[:] vs
        cdef float[:,:] logits

        if self.time: self.timings.reset()        

        assert type(action) is tuple and len(action) == 2, \
            "action should be a tuple of size 2, containing augmented action and reset"        
        for i in range(len(action)):
            a = action[i]            
            assert torch.is_tensor(a) or \
                isinstance(a, np.ndarray) or \
                isinstance(a, list), \
                f"action[{i}] should be either torch.tensor or np.ndarray or list"            
            if torch.is_tensor(a):
                a = a.detach().cpu().numpy()
            if isinstance(a, list):
                a = np.array(a, dtype=np.int32)
            assert a.shape == (self.env_n,), \
                f"action[{i}] shape should be {(self.env_n,)}, not {a.shape}"
            if a.dtype != np.int32:
                a = a.astype(np.int32)
            if i == 0:
                re_action = a
                im_action = a
                assert (a >= 0).all() and (a < self.num_actions).all(), \
                    f"primiary action should be in [0, {self.num_actions-1}], not {a}"
            else:
                reset = a       
                assert (a >= 0).all() and (a < 2).all(), \
                    f"reset action should be in [0, 1], not {a}"  

        pass_env_states = []

        for i in range(self.env_n):            
            # compute the mask of real / imagination step                             
            self.max_rollout_depth_[i] = self.max_rollout_depth[i]
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
                        pass_env_states.append(encoded["env_states"])
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
                encoded = <dict> self.root_nodes[i][0].encoded
                pass_env_states.append(encoded["env_states"])
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
                pass_action_py = torch.tensor(pass_action, dtype=long, device=self.device).unsqueeze(0)
                model_net_out = model_net(obs_py, 
                                          pass_action_py, 
                                          one_hot=False,
                                          ret_xs=self.return_x,
                                          ret_zs=False,
                                          ret_hs=self.return_h)  
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
                        discounting=self.discounting, rec_t=self.rec_t, remember_path=True)
                else:
                    root_node = self.root_nodes[i][0].ppchildren[0][re_action[i]]
                encoded = {"real_states": obs_py[j],
                            "xs": model_net_out.xs[-1, j] if model_net_out.xs is not None else None,
                            "hs": model_net_out.hs[-1, j] if model_net_out.hs is not None else None,
                            "env_states": env_state[j],}
                node_expand(pnode=root_node, r=0., v=vs[j], t=self.total_step[i], done=False,
                    logits=logits[j], encoded=<PyObject*>encoded, override=not new_root)
                except_idx = -1 if new_root else re_action[i]
                node_del(self.root_nodes[i], except_idx=except_idx)
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
                encoded = {"real_states": None,
                            "xs": model_net_out.xs[-1, j] if model_net_out.xs is not None else None,
                            "hs": model_net_out.hs[-1, j] if model_net_out.hs is not None else None,
                            "env_states": env_state[j],
                            }
                cur_node = self.cur_nodes[i][0].ppchildren[0][im_action[i]]
                if self.reward_clip > 0.: 
                    r = np.clip(reward[j], -self.reward_clip, +self.reward_clip)
                else:
                    r = reward[j]
                node_expand(pnode=cur_node, r=r, v=vs[j] if not done[j] else 0., t=self.total_step[i], done=done[j],
                        logits=logits[j], encoded=<PyObject*>encoded, override=False)
                node_visit(cur_node)
                root_nodes_.push_back(self.root_nodes[i])
                cur_nodes_.push_back(cur_node)   
                j += 1                            

        self.root_nodes = root_nodes_
        self.cur_nodes = cur_nodes_
        if self.time: self.timings.time("compute_root_cur_nodes")

        # reset if serach depth exceeds max depth
        if self.max_depth > 0:
            for i in range(self.env_n):
                if self.rollout_depth[i] >= self.max_depth:
                    reset[i] = 1

        # compute states        
        states = self.prepare_state(reset, self.status)
        if self.time: self.timings.time("compute_state")
        
        # compute reward
        j = 0
        for i in range(self.env_n):
            if self.status[i] == 1:
                self.full_reward[i] = reward[j]
            else:
                self.full_reward[i] = 0.
            if self.im_enable:                        
                self.root_nodes_qmax_[i] = self.root_nodes[i][0].max_q
                if self.status[i] != 1:                
                    self.full_im_reward[i] = (self.root_nodes_qmax_[i] - self.root_nodes_qmax[i])
                else:
                    self.full_im_reward[i] = 0.
                self.root_nodes_qmax[i] = self.root_nodes_qmax_[i]
            if self.status[i] == 1 or self.status[i] == 4:
                j += 1
        if self.time: self.timings.time("compute_reward")
        
        j = 0
        for i in range(self.env_n):
            # compute done & full_real_done
            if self.status[i] == 1:
                self.full_done[i] = done[j]
                self.full_real_done[i] = real_done[j]
                self.full_truncated_done[i] = truncated_done[j]
                self.full_im_done[i] = False
            else:
                self.full_done[i] = False
                self.full_real_done[i] = False
                self.full_truncated_done[i] = False
                self.full_im_done[i] = self.cur_nodes[i][0].done
            if self.status[i] == 1 or self.status[i] == 4:
                j += 1

            # compute reset
            if reset[i]:
                self.rollout_depth[i] = 0
                self.cur_nodes[i] = self.root_nodes[i]
                node_visit(self.cur_nodes[i])
                self.status[i] = 5 # need to restore state on the next transition, so we need to alter the status from 4

            if self.cur_t[i] == 0:
                self.step_status[i] = 0 # real action just taken
            elif self.cur_t[i] < self.rec_t - 1:
                self.step_status[i] = 1 # im action just taken
            elif self.cur_t[i] >= self.rec_t - 1:
                self.step_status[i] = 2 # im action just taken; next action is real action
        
        # some extra info
        info = {
                "real_done": torch.tensor(self.full_real_done, dtype=torch.float, device=self.device).bool(),
                "truncated_done": torch.tensor(self.full_truncated_done, dtype=torch.float, device=self.device).bool(),
                #"im_done": torch.tensor(self.full_im_done, dtype=torch.float, device=self.device).bool(),
                "step_status": torch.tensor(self.step_status, device=self.device),
                "max_rollout_depth":  torch.tensor(self.max_rollout_depth_, dtype=torch.long, device=self.device),
                "baseline": self.baseline_mean_q,
                }
        if self.im_enable:
            info["im_reward"] = torch.tensor(self.full_im_reward, dtype=torch.float, device=self.device)
        if self.cur_enable:
            info["cur_reward"] = torch.tensor(self.full_im_reward, dtype=torch.float, device=self.device)
        if self.time: self.timings.time("end")

        return (states, 
                torch.tensor(self.full_reward, dtype=torch.float, device=self.device), 
                torch.tensor(self.full_done, dtype=torch.float, device=self.device).bool(), 
                info)
    
    cdef float[:, :] compute_tree_reps(self, int[:]& reset, int[:]& status):
        cdef int i
        cdef int idx1 = self.num_actions*5+5
        cdef int idx2 = self.num_actions*10+7

        result_np = np.zeros((self.env_n, self.obs_n), dtype=np.float32)
        cdef float[:, :] result = result_np        
        for i in range(self.env_n):
            result[i, :idx1] = node_stat(self.root_nodes[i], detailed=True, enc_type=self.enc_type, mask_type=self.stat_mask_type)
            result[i, idx1:idx2] = node_stat(self.cur_nodes[i], detailed=False, enc_type=self.enc_type, mask_type=self.stat_mask_type)    
            # reset
            if reset is None or status[i] == 1:
                result[i, idx2] = 1.
            else:
                result[i, idx2] = reset[i]
            # time
            if self.cur_t[i] < self.rep_rec_t:
                result[i, idx2+1+self.cur_t[i]] = 1.
            # deprec
            result[i, idx2+self.rep_rec_t+1] = (self.discounting ** (self.rollout_depth[i]))           
        return result

    cdef compute_model_out(self, vector[Node*] nodes, key):
        outs = []
        for i in range(self.env_n):
            encoded = <dict>nodes[i][0].encoded
            if key not in encoded or encoded[key] is None: 
                return None
            outs.append((encoded[key] if not nodes[i][0].done 
                else torch.zeros_like(encoded[key])).unsqueeze(0))
        outs = torch.concat(outs)
        return outs

    cdef prepare_state(self, int[:]& reset, int[:]& status):
        cdef int i

        tree_reps = self.compute_tree_reps(reset, status)
        tree_reps = torch.tensor(tree_reps, dtype=torch.float, device=self.device)

        if status is None or np.any(np.array(status)!=1):
            self.real_states = self.compute_model_out(self.root_nodes, "real_states")

        states = {
            "tree_reps": tree_reps,
            "real_states": self.real_states,
        }

        if self.return_x:
            xs = self.compute_model_out(self.cur_nodes, "xs")
            if self.return_double and xs is not None:
                root_xs = self.compute_model_out(self.root_nodes, "xs")
                xs = torch.concat([root_xs, xs], dim=1)        
            assert xs is not None, "xs cannot be None"
            states["xs"] = xs

        if self.return_h:
            hs = self.compute_model_out(self.cur_nodes, "hs")
            if self.return_double and hs is not None:
                root_hs = self.compute_model_out(self.root_nodes, "hs")
                hs = torch.concat([root_hs, hs], dim=1)
            assert hs is not None, "hs cannot be None"
            states["hs"] = hs

        return states

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
