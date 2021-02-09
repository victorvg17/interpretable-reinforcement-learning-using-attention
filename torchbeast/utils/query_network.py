import torch
import torch.nn.functional as F
from torch import nn


class QueryNetwork(nn.Module):
    def __init__(self, hidden_size, num_queries, num_keys):
        super(QueryNetwork, self).__init__()
        output_size = num_queries * num_keys  # 72 * 4 = 288
        self.num_queries, self.num_keys = num_queries, num_keys
        self.model = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size),
            nn.ReLU(),
            nn.Linear(output_size, output_size),
        )

    def forward(self, query):
        # [N, H] -> [N, num_queries * num_keys]
        out = self.model(query)
        # [N, H] -> [N, num_queries, num_keys]
        return out.view(-1, self.num_queries, self.num_keys)