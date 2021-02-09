import torch
import torch.nn.functional as F
from torch import nn


class SpatialBasis:
    """
    NOTE: The `height` and `weight` depend on the inputs' size and its resulting size
    after being processed by the vision network.
    """

    def __init__(self, height=27, width=20, channels=64):
        h, w, d = height, width, channels

        p_h = torch.mul(
            torch.arange(1, h + 1).unsqueeze(1).float(), torch.ones(1, w).float()
        ) * (np.pi / h)
        p_w = torch.mul(
            torch.ones(h, 1).float(), torch.arange(1, w + 1).unsqueeze(0).float()
        ) * (np.pi / w)

        # TODO: I didn't quite see how U,V = 4 made sense given that the authors form the spatial
        # basis by taking the outer product of the values.
        # I am not confident in this step.
        U = V = 8  # size of U, V.
        u_basis = v_basis = torch.arange(1, U + 1).unsqueeze(0).float()
        a = torch.mul(p_h.unsqueeze(2), u_basis)
        b = torch.mul(p_w.unsqueeze(2), v_basis)
        out = torch.einsum("hwu,hwv->hwuv", torch.cos(a), torch.cos(b)).reshape(h, w, d)
        self.S = out

    def __call__(self, X):
        # Stack the spatial bias (for each batch) and concat to the input.
        batch_size = X.size()[0]
        S = torch.stack([self.S] * batch_size).to(X.device)
        return torch.cat([X, S], dim=3)