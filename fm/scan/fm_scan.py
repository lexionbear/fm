from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline

cuda_source = (Path(__file__).parent / 'fm_scan.cuh').read_text()

cpp_source = """
at::Tensor fm_scan_forward(const at::Tensor &gates, const at::Tensor &tokens, const at::Tensor &initial_state, const at::Tensor &out);

void fm_scan_backward(const at::Tensor &gates, const at::Tensor &outGrad, const at::Tensor &firstPropGrad, 
    const at::Tensor& tokenGradOut,
    const at::Tensor &cached_output, const at::Tensor &input_tokens, 
    const at::Tensor& gateGradOut
    );
"""

module = load_inline(
    name='fm_scan',
    cpp_sources=[cpp_source],
    cuda_sources=[cuda_source],
    functions=['fm_scan_forward', 'fm_scan_backward'],
    verbose=True,
    extra_cuda_cflags=[
        "-O3",
        "-std=c++17",
        "--ptxas-options=-v",
        "-lineinfo",
        "--fmad", "false",
        "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    ]
)
fm_scan_forward = module.fm_scan_forward
fm_scan_backward = module.fm_scan_backward

class FM_Scan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gates, tokens, initial_state=None):
        B1, M, L1 = gates.shape
        B, D, L = tokens.shape
        assert L1 == L
        assert B == B1
        assert gates.is_contiguous()
        assert tokens.is_contiguous()

        if initial_state is None:
            initial_state = torch.zeros([B, M, D], device=gates.device, dtype=gates.dtype)
        else:
            assert initial_state.shape == (B, M, D)
            assert initial_state.is_contiguous()

        output =  torch.empty([B, M, D, L], device=gates.device, dtype=gates.dtype)
        states = fm_scan_forward(gates, tokens, initial_state, output)
        ctx.save_for_backward(tokens, gates, states)
        return states

    # backward scan is a padded reverse scan
    # See https://arxiv.org/abs/1709.04057 Section 2.2
    @staticmethod
    def backward(ctx, grad_output):
        tokens, gates, states = ctx.saved_tensors
        B, M, D, L = states.shape

        firstPropGrad = torch.zeros([B, M, D], device=gates.device, dtype=gates.dtype)
        grad_output = grad_output.contiguous()
        assert states.is_contiguous()
        assert gates.is_contiguous()

        d_gates = torch.empty_like(states) # (B, M, D, L)
        d_tokens = torch.empty_like(states)

        fm_scan_backward(gates, grad_output, firstPropGrad, d_tokens, states, tokens, d_gates)

        # convert d_gates back to (B, M, L)
        d_gates = d_gates.sum(dim=2, keepdim=False)
        # convert d_tokens back to (B, D, L)
        d_tokens = d_tokens.sum(dim=1, keepdim=False)

        # TODO: connect the initial state's gradient, passing none placeholder for now
        return d_gates, d_tokens, None


def fm_scan(gates, tokens, initial_state=None):
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
    return FM_Scan.apply(gates, tokens, initial_state)


def fm_memory(alpha, update_scale, output_scale, inputs, initial_state=None, mem_norm = True, norm_eps=1e-6 ):
    '''
        alpha: (B, L, M)
        update_scale: (B, L, 1)
        output_scale: (B, L, 1)
        tokens: (B, L, D)

        initial_state: (B, M, D)
        mem_norm: bool
        norm_eps: float
        
        return: (B, D, L)
    '''

    # (B, L, M) * (B, L, D) -> (B, M, D, L) use einops
    update_weight = alpha * update_scale
    tokens = inputs.transpose(1, 2).contiguous() # (B, D, L)
    update_weight = update_weight.transpose(1, 2).contiguous() # (B, M, L)
    decay_weight = 1.0 - update_weight

    memory_states = fm_scan(decay_weight, tokens, initial_state) # (B, M, D, L)
    
    # run RMS norm
    if mem_norm:
        memory_rms = torch.rsqrt(torch.mean(memory_states ** 2, dim=2, keepdim=True) + norm_eps) # (B, M, 1, L)
        memory_states = memory_states * memory_rms # (B, M, D, L)

    # run output projection
    output_weight = alpha * output_scale
    output = torch.einsum('blm,bmdl->bld', output_weight, memory_states).contiguous()

    return output, memory_states