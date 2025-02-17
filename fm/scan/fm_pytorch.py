import torch

def fm_scan_pytorch(gates, tokens, initial_state=None):
    """Solve a first-order recurrence relation:

    .. math::
        x_t = a_t x_{t-1} + b_t

    where :math:`a_t` ("gates") and :math:`b_t` ("tokens") are sequences of vectors.

    Arguments:
        gates (torch.Tensor): shape (B, M, L), must be contiguous. T must be a power of 2.
        tokens (torch.Tensor): shape (B, D, L), must be contiguous. T must be a power of 2.

    Returns:
        (torch.Tensor): shape (B, M, D, L)
    """

    B, M, L = gates.shape
    B1, D, L1 = tokens.shape

    updates = 1 - gates

    assert L1 == L
    assert B == B1

    if initial_state is None:
        initial_state = torch.zeros([B, M, D], device=gates.device, dtype=gates.dtype)
    else:
        assert initial_state.shape == (B, M, D)
        assert initial_state.is_contiguous()

    output_states = [initial_state]

    for t in range(L):
        # Compute the new state
        new_state = gates[:, :, t].unsqueeze(-1) * output_states[-1] + updates[:, :, t].unsqueeze(-1) * tokens[:, :, t].unsqueeze(1)
        output_states.append(new_state)

    # concat
    output_states = torch.stack(output_states[1:], dim=3)
    return output_states
