from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
import atari_wrappers


def spatial_softmax(A):
    r"""Softmax over the attention map.

    Ignoring batches, this operation produces a tensor of the original shape
    that has been normalized across its "spatial" dimension `h` and `w`.

    .. math::

        A^q_{h,w} = exp(A^c_{h,w}) / \sum_{h',w'} A^q_{h',w'}

    Thus, each channel is operated upon separately, and no special meaning is
    given to the relative position in the image.
    """
    b, h, w, num_queries = A.size()
    # Flatten A s.t. softmax is applied to each grid (height by width) (not over queries or channels).
    A = A.view(b, h * w, num_queries)
    A = F.softmax(A, dim=1)
    # Reshape A back to original shape.
    A = A.view(b, h, w, num_queries)
    return A


def apply_attention(A, V):
    r"""Applies set of attention matrices A over V.

    Ignoring batches the operation produces a tensor of shape [num_queries, num_values]
    and follows the following equation:

    .. math::

        out_{q,v} = \sum_{h,w} A^q_{h,w} * V^v_{h,w}

    Parameters
    ----------
    A : [B, h, w, num_queries]
    V : [B, h, w, num_values]
    --> [B, num_queries, num_values]
    """
    b, h, w, num_queries = A.shape
    num_values = V.size(3)

    # [B, h, w, num_queries] -> [B, h * w, num_queries]
    A = A.reshape(b, h * w, num_queries)
    # [B, h * w, num_queries] -> [B, num_queries, h * w]
    A = A.transpose(1, 2)
    # [B, h, w, num_values] -> [B, h * w, num_values]
    V = V.reshape(b, h * w, num_values)
    # [B, h * w, num_values] x [B, num_queries, h * w] -> [B, num_queries, num_values]
    return torch.matmul(A, V)


def splice_core_state(state: Tuple) -> Tuple:
    return state[0], state[1]


def splice_vision_state(state: Tuple) -> Tuple:
    return state[2], state[3]


def compute_baseline_loss(advantages):
    return 0.5 * torch.sum(advantages ** 2)


def compute_entropy_loss(logits):
    """Return the entropy loss, i.e., the negative entropy of the policy."""
    policy = F.softmax(logits, dim=-1)
    log_policy = F.log_softmax(logits, dim=-1)
    return torch.sum(policy * log_policy)


def compute_policy_gradient_loss(logits, actions, advantages):
    cross_entropy = F.nll_loss(
        F.log_softmax(torch.flatten(logits, 0, 1), dim=-1),
        target=torch.flatten(actions, 0, 1),
        reduction="none",
    )
    cross_entropy = cross_entropy.view_as(advantages)
    return torch.sum(cross_entropy * advantages.detach())


def create_deepmind_env(flags):
    return atari_wrappers.wrap_pytorch(
        atari_wrappers.wrap_deepmind(
            atari_wrappers.make_atari(flags.env),
            clip_rewards=False,
            frame_stack=True,
            scale=False,
        )
    )


def create_env(flags):
    return atari_wrappers.wrap_pytorch(
        atari_wrappers.wrap_interp(
            atari_wrappers.make_atari(flags.env),
        )
    )