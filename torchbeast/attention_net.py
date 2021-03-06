from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from utils.conv_lstm import ConvLSTMCell
from utils.query_network import QueryNetwork
from utils.spatial_basis import SpatialBasis
from utils.util_functions import (
    spatial_softmax,
    apply_attention,
    splice_core_state,
    splice_vision_state,
)

"""
AttentionNet is a pytorch reimpl. of the following paper for the NeurIPS reproducibility challenge.

@article{mott2019towards,
  title={Towards Interpretable Reinforcement Learning Using Attention Augmented Agents},
  author={Mott, Alex and Zoran, Daniel and Chrzanowski, Mike and Wierstra, Daan and Rezende, Danilo J},
  journal={arXiv preprint arXiv:1906.02500},
  year={2019}
}
"""


class AttentionNet(nn.Module):
    def __init__(
        self,
        num_actions: int,
        hidden_size: int = 256,
        c_v: int = 120,
        c_k: int = 8,
        c_s: int = 64,
        num_queries: int = 4,
    ):
        """AttentionNet implementing the attention agent."""
        super(AttentionNet, self).__init__()
        self.num_queries = num_queries
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.c_v, self.c_k, self.c_s, self.num_queries = c_v, c_k, c_s, num_queries
        self.num_keys = c_k + c_s
        self.num_values = c_v + c_s

        # This is a function of the CNN hp and the input image size.
        self.height, self.width = 27, 20

        self.vision = VisionNetwork()
        self.query = QueryNetwork(hidden_size, num_queries, self.num_keys)
        self.spatial = SpatialBasis(self.height, self.width)

        self.answer_processor = nn.Sequential(
            # 1043 x 512
            nn.Linear(
                self.num_values * num_queries
                + self.num_keys * num_queries
                + 1
                + self.num_actions,
                hidden_size * 2,  # 512, HP from the authors.
            ),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )

        self.policy_core = nn.LSTMCell(hidden_size, hidden_size)
        self.policy_head = nn.Sequential(nn.Linear(hidden_size, num_actions))
        self.values_head = nn.Sequential(nn.Linear(hidden_size, 1))

    def initial_state(self, batch_size):
        core_zeros = torch.zeros(batch_size, self.hidden_size).float()
        conv_zeros = torch.zeros(
            batch_size, self.hidden_size // 2, self.height, self.width
        ).float()
        return (
            core_zeros.clone(),  # hidden for policy core
            core_zeros.clone(),  # cell for policy core
            conv_zeros.clone(),  # hidden for vision lstm
            conv_zeros.clone(),  # cell for vision lstm
        )

    def forward(self, inputs, prev_state):

        # 1 (a). Vision.
        # --------------

        # [T, B, C, H, W].
        X = inputs["frame"]
        T, B, *_ = X.size()
        vision_state = splice_vision_state(prev_state)
        # -> [N, c_k+c_v, h, w] where N = T * B.
        O, next_vision_state = self.vision(X, inputs, vision_state)
        # -> [N, h, w, c_k+c_v]
        N, c, h, w = O.size()
        O = O.view(N, h, w, c)

        # -> [N, h, w, c_k], [N, h, w, c_v]
        K, V = O.split([self.c_k, self.c_v], dim=3)
        # -> [N, h, w, num_keys = c_k + c_s], [N, h, w, num_values = c_v + c_s]
        K, V = self.spatial(K), self.spatial(V)

        # 2. Prepare all inputs to operate over the T steps.
        # --------------------------------------------------
        core_output_list = []
        core_state = splice_core_state(prev_state)
        prev_output = core_state[0]

        # [N, h, w, num_keys] -> [T, B, h, w, num_keys]
        K = K.view(T, B, h, w, -1)
        # [N, h, w, num_values] -> [T, B, h, w, num_values]
        V = V.view(T, B, h, w, -1)
        # -> [T, B, 1]
        notdone = (~inputs["done"]).float().view(T, B, 1)
        # -> [T, B, 1, num_actions]
        prev_action = (
            F.one_hot(inputs["last_action"].view(T * B), self.num_actions)
            .view(T, B, 1, self.num_actions)
            .float()
        )
        # -> [T, B, 1, 1]
        prev_reward = torch.clamp(inputs["reward"], -1, 1).view(T, B, 1, 1)

        # 3. Operate over the T time steps.
        # ---------------------------------
        # NOTE: T = 1 when 'act'ing and T > 1 when 'learn'ing.

        for K_t, V_t, prev_reward_t, prev_action_t, nd_t in zip(
            K.unbind(),
            V.unbind(),
            prev_reward.unbind(),
            prev_action.unbind(),
            notdone.unbind(),
        ):

            # A. Queries.
            # --------------
            # [B, hidden_size] -> [B, num_queries, num_keys]
            Q_t = self.query(prev_output)
            # NOTE: The queries and keys don't have intrinsic meaning -- they're trained
            # to be relevant. Furthermore, the sizing relies on selecting `smart` hidden
            # and output sizes in the QueryNetwork.

            # B. Answer.
            # ----------
            # [B, h, w, num_keys] x [B, 1, num_keys, num_queries] -> [B, h, w, num_queries]
            A = torch.matmul(K_t, Q_t.transpose(2, 1).unsqueeze(1))
            # -> [B, h, w, num_queries]
            A = spatial_softmax(A)
            # [B, h, w, num_queries] x [B, h, w, num_values] -> [B, num_queries, num_values]
            answers = apply_attention(A, V_t)

            # -> [B, Z = {num_values * num_queries + num_keys * num_queries + 1 + num_actions}]
            chunks = list(
                torch.chunk(answers, self.num_queries, dim=1)
                + torch.chunk(Q_t, self.num_queries, dim=1)
                + (prev_reward_t.float(), prev_action_t.float())
            )
            answer = torch.cat(chunks, dim=2).squeeze(1)
            # [B, Z] -> [B, hidden_size]
            core_input = self.answer_processor(answer)

            # C. Policy.
            # ----------
            # NOTE: nd_t is 0 when episode is done, 1 otherwise. Thus, this resets the state
            # as episodes finish.
            core_state = tuple((nd_t * s) for s in core_state)

            # -> [B, hidden_size]
            prev_output, _ = core_state = self.policy_core(core_input, core_state)
            core_output_list.append(prev_output)
        next_core_state = core_state
        # -> [T * B, hidden_size]
        output = torch.cat(core_output_list, dim=0)

        # 4. Outputs.
        # --------------
        # [T * B, hidden_size] -> [T * B, num_actions]
        logits = self.policy_head(output)
        # [T * B, hidden_size] -> [T * B, 1]
        values = self.values_head(output)

        if self.training:
            # [T * B, num_actions] -> [T * B, 1]
            action = torch.multinomial(F.softmax(logits, dim=1), num_samples=1)
        else:
            # [T * B, num_actions] -> [T * B, 1]
            # Don't sample when testing.
            action = torch.argmax(logits, dim=1)

        # Format for torchbeast.
        # [T * B, num_actions] -> [T * B, num_actions]
        policy_logits = logits.view(T, B, self.num_actions)
        # [T * B, 1] -> [T, B]
        baseline = values.view(T, B)
        # [T * B, 1] -> [T, B]
        action = action.view(T, B)

        # Create tuple of next states.
        next_state = next_core_state + next_vision_state
        return (
            dict(policy_logits=policy_logits, baseline=baseline, action=action),
            next_state,
        )


class VisionNetwork(nn.Module):
    def __init__(self):
        super(VisionNetwork, self).__init__()
        self.cnn = nn.Sequential(
            # NOTE: The padding choices were not in the paper details, but result in the sizes
            # mentioned by the authors. We should review these.
            nn.Conv2d(
                in_channels=3, out_channels=32, kernel_size=8, stride=4, padding=(1, 2)
            ),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=(2, 1)
            ),
        )
        self.lstm = ConvLSTMCell(input_channels=64, hidden_channels=128, kernel_size=3)

    def forward(self, X, inputs, vision_lstm_state):
        T, B, *_ = X.size()
        X = torch.flatten(X, 0, 1).float()
        X = self.cnn(X)
        _, C, H, W = X.size()
        X = X.view(T, B, C, H, W)
        output_list = []
        notdone = (~inputs["done"]).float()

        # Operating over T:
        for X_t, nd in zip(X.unbind(), notdone.unbind()):
            # This vector is all 0 if the episode just finished, so it will reset the hidden
            # state as needed.
            nd = nd.view(B, 1, 1, 1)
            vision_lstm_state = tuple((nd * s) for s in vision_lstm_state)

            O_t, vision_state = self.lstm(X_t, vision_lstm_state)
            output_list.append(O_t)
        next_vision_state = vision_state
        # (T * B, h, w, c)
        O = torch.cat(output_list, dim=0)
        return O, next_vision_state
