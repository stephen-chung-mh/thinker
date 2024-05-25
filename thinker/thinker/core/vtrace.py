# This file taken from
#     https://github.com/facebookresearch/torchbeast/blob/
#         main/torchbeast/core/vtrace.py
# and modified.

# This file taken from
#     https://github.com/deepmind/scalable_agent/blob/
#         cd66d00914d56c8ba2f0615d9cdeefcb169a8d70/vtrace.py
# and modified.

# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functions to compute V-trace off-policy actor critic targets.

For details and theory see:

"IMPALA: Scalable Distributed Deep-RL with
Importance Weighted Actor-Learner Architectures"
by Espeholt, Soyer, Munos et al.

See https://arxiv.org/abs/1802.01561 for the full paper.
"""

import collections
import torch
import torch.nn.functional as F

VTraceReturns = collections.namedtuple("VTraceReturns", "vs pg_advantages pg_advantages_nois norm_stat")

def action_log_probs(policy_logits, actions):
    return -F.nll_loss(
        F.log_softmax(torch.flatten(policy_logits, 0, -2), dim=-1),
        torch.flatten(actions),
        reduction="none",
    ).view_as(actions)


def adv_l2(target_x, x):
    return target_x - x


@torch.no_grad()
def compute_v_trace(
    log_rhos,
    discounts,
    rewards,
    values,
    bootstrap_value,
    return_norm_type,
    clip_rho_threshold=1.0,
    clip_pg_rho_threshold=1.0,
    lamb=1.0,
    norm_stat=None,
):
    """V-trace from log importance weights."""
    with torch.no_grad():
        rhos = torch.exp(log_rhos)
        if clip_rho_threshold is not None:
            clipped_rhos = torch.clamp(rhos, max=clip_rho_threshold)
        else:
            clipped_rhos = rhos

        cs = lamb * torch.clamp(rhos, max=1.0)
        # Append bootstrapped value to get [v1, ..., v_t+1]
        values_t_plus_1 = torch.cat(
            [values[1:], torch.unsqueeze(bootstrap_value, 0)], dim=0
        )
        deltas = clipped_rhos * (rewards + discounts * values_t_plus_1 - values)

        acc = torch.zeros_like(bootstrap_value)
        result = []
        for t in range(discounts.shape[0] - 1, -1, -1):
            acc = deltas[t] + discounts[t] * cs[t] * acc
            result.append(acc)
        result.reverse()
        vs_minus_v_xs = torch.stack(result)

        # Add V(x_s) to get v_s.
        vs = torch.add(vs_minus_v_xs, values)

        # Advantage for policy gradient.
        broadcasted_bootstrap_values = torch.ones_like(vs[0]) * bootstrap_value
        vs_t_plus_1 = torch.cat(
            [vs[1:], broadcasted_bootstrap_values.unsqueeze(0)], dim=0
        )
        if clip_pg_rho_threshold is not None:
            clipped_pg_rhos = torch.clamp(rhos, max=clip_pg_rho_threshold)
        else:
            clipped_pg_rhos = rhos
        target_values = rewards + discounts * vs_t_plus_1
        if not return_norm_type in [0, 1]:
            norm_stat = None
        else:
            if return_norm_type == 0:
                norm_v = target_values
            elif return_norm_type == 1:
                norm_v = target_values - values
            
            buffer = norm_stat[-1]
            buffer.push(norm_v)
            lq = buffer.get_percentile(0.05)
            uq = buffer.get_percentile(0.95)
            norm_stat = (
                lq,
                uq,
            )
            norm_factor = torch.clamp(
                norm_stat[1] - norm_stat[0], min=1
            )
            norm_stat = norm_stat + (norm_factor, buffer)

        pg_advantages_nois = adv_l2(target_values, values)
        pg_advantages = clipped_pg_rhos * pg_advantages_nois
        if return_norm_type in [0, 1]:
            # pg_advantages = torch.clamp(pg_advantages, norm_stat[0], norm_stat[1])
            pg_advantages = pg_advantages / norm_factor
            pg_advantages_nois = pg_advantages_nois / norm_factor
        elif return_norm_type == 2:
            pg_advantages = (pg_advantages - pg_advantages.mean()) / (pg_advantages.std() + 1e-8)
            pg_advantages_nois = (pg_advantages_nois - pg_advantages_nois.mean()) / (pg_advantages_nois.std() + 1e-8)


        # Make sure no gradients backpropagated through the returned values.
        return VTraceReturns(vs=vs, 
                             pg_advantages=pg_advantages, 
                             pg_advantages_nois=pg_advantages_nois,
                             norm_stat=norm_stat)

