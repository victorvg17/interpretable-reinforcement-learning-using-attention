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


class AttentionNetMinAtar(nn.Module):
    def __init__(
        self,
        num_actions: int,
        hidden_size: int = 64,
        c_v: int = 30,
        c_k: int = 2,
        c_s: int = 16,
        num_queries: int = 4,
    ):
        super(AttentionNetMinAtar, self).__init__()
        self.num_queries = num_queries
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.c_v, self.c_k, self.c_s, self.num_queries = c_v, c_k, c_s, num_queries
        self.num_keys = c_k + c_s  # 18 = 72/4
        self.num_values = c_v + c_s  # 46 = 184/4

        # This is a function of the CNN hp and the input image size.
        self.height, self.width = 6, 6
        # in_channels depend on the selected MinAtar game. For Breakout, 4
        self.vision = VisionNetwork(in_channels=4)
        self.query = QueryNetwork(
            hidden_size=self.hidden_size,
            num_queries=self.num_queries,
            num_keys=self.num_keys,
        )
        self.spatial = SpatialBasis(height=self.height, width=self.width)

        self.answer_processor = nn.Sequential(
            # 263*128
            nn.Linear(
                self.num_values * self.num_queries
                + self.num_keys * self.num_queries
                + 1
                + self.num_actions,
                self.hidden_size * 2,  # 128
            ),
            nn.ReLU(),
            # 128*64
            nn.Linear(self.hidden_size * 2, hidden_size),
        )

        self.policy_core = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.policy_head = nn.Linear(self.hidden_size, self.num_actions)
        self.values_head = nn.Linear(self.hidden_size, 1)

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

        # 1 (a). Vision core.
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
        prev_action = inputs["last_action"].view(T * B)
        prev_action = (
            F.one_hot(prev_action, self.num_actions)
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
            core_state = tuple((nd_t * s) for s in core_state)

            # -> [B, hidden_size]
            core_state = self.policy_core(core_input, core_state)
            prev_output, _ = core_state
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
        # [T * B, num_actions] -> [T, B, num_actions]
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
    def __init__(self, in_channels):
        super(VisionNetwork, self).__init__()
        self.in_channels = in_channels
        # ip size=10*10*in_channels, op size=8*8*16
        self.cnn = nn.Sequential(
            # ip size=10*10*in_channels, op size=8*8*16
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=16,
                kernel_size=3,
                stride=1,
            ),
            # ip size=8*8*16, op size=6*6*32
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1),
        )
        self.lstm = ConvLSTMCell(input_channels=16, hidden_channels=32, kernel_size=3)

    def forward(self, X, inputs, vision_lstm_state):
        T, B, *_ = X.size()
        X = torch.flatten(X, 0, 1).float()
        X = self.cnn(X)
        _, C, H, W = X.size()
        X = X.view(T, B, C, H, W)
        output_list = []
        # shape: (T, B, 1)
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