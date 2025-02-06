// leverage similar algorithm as https://github.com/proger/accelerated-scan
#include <cuda.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#define CHECK_STRIDE(x) TORCH_CHECK(x.stride(-1) == 1 || x.size(-1) == 1);

template <typename weight_t, int kNThreadsPerWarp, int kNWarpsPerBlock, int kNChunksPerSequence, 
          int size_B, int size_M, int size_D, int size_L,
          bool reverse, bool backward>
__global__ void fm_scan(
    const weight_t* __restrict__ gates,   // [B, M, L]
    const weight_t* __restrict__ tokens,  // Forward: [B, D, L], Backward: [B, M, D, L]
    const weight_t* __restrict__ initial_state, // [B, M, D]
    weight_t* __restrict__ result,        // [B, M, D, L]
    // Backward
    const weight_t* __restrict__ cached_output,  // [B, M, D, L]
    weight_t* __restrict__ gateGradOut    // [B, M, L]
) {
    __shared__ weight_t warpLastGate[kNWarpsPerBlock];
    __shared__ weight_t warpLastToken[kNWarpsPerBlock];
    __shared__ weight_t chunkAccGate;
    __shared__ weight_t chunkAccToken;

    const int batch_idx = blockIdx.x;
    const int m_idx = blockIdx.y;
    const int d_idx = blockIdx.z;

    const int tid = threadIdx.x;
    const int warpId = tid / kNThreadsPerWarp;
    const int laneId = tid % kNThreadsPerWarp;

    const int chunklen = blockDim.x;  // Threads per chunk (blockDim.x)
    constexpr int kBlockLast = kNWarpsPerBlock - 1;
    constexpr int kWarpLast = kNThreadsPerWarp - 1;
    constexpr weight_t kEmptyGate = 1.0;

    // Offsets
    const int gate_offset = batch_idx * size_M * size_L + m_idx * size_L;
    const int result_offset = batch_idx * size_M * size_D * size_L + m_idx * size_D * size_L + d_idx * size_L;
    const int token_offset = backward ? (batch_idx * size_M * size_D * size_L + m_idx * size_D * size_L + d_idx * size_L) 
                                      : (batch_idx * size_D * size_L + d_idx * size_L);
    const int initial_state_offset = batch_idx * size_M * size_D + m_idx * size_D + d_idx;

    for (int chunk = 0; chunk < kNChunksPerSequence; ++chunk) {
        int seq_pos;
        if (reverse) {
            // Process chunks in reverse order, elements within chunk also reversed
            seq_pos = (kNChunksPerSequence - chunk) * chunklen - 1 - tid;
        } else {
            seq_pos = chunk * chunklen + tid;
        }

        if (seq_pos >= size_L) continue;

        const int gate_idx = gate_offset + seq_pos;
        const int token_idx = token_offset + seq_pos;
        const int result_idx = result_offset + seq_pos;

        if (chunk != 0) __syncthreads();

        // Load gate
        weight_t gate;
        if (reverse) {
            if (chunk == 0 && tid == 0) {
                gate = kEmptyGate;
            } else {
                gate = gates[gate_idx + 1];
            }
        } else {
            gate = gates[gate_idx];
        }

        // Load token
        weight_t token = tokens[token_idx];

        // Initialize accumulators
        weight_t accToken, accGate;
        if (chunk == 0 && tid == 0) {
            accToken = initial_state[initial_state_offset] * gate + token;
            accGate = gate;
        } else {
            if (tid == 0) {
                accToken = chunkAccToken * gate + token;
                accGate = chunkAccGate * gate;
            } else {
                accToken = token;
                accGate = gate;
            }
        }

        // Warp-level scan
        for (int delta = 1; delta < kNThreadsPerWarp; delta *= 2) {
            weight_t prev_gate = __shfl_up_sync(0xffffffff, accGate, delta);
            weight_t prev_token = __shfl_up_sync(0xffffffff, accToken, delta);

            if (laneId >= delta) {
                accToken = prev_token * accGate + accToken;
                accGate *= prev_gate;
            }
        }

        __syncwarp();

        // Store warp results
        if (laneId == kWarpLast) {
            warpLastGate[warpId] = accGate;
            warpLastToken[warpId] = accToken;
        }

        __syncthreads();

        // Block-level scan
        if (warpId == 0) {
            weight_t warpAccGate = (laneId < kNWarpsPerBlock) ? warpLastGate[laneId] : kEmptyGate;
            weight_t warpAccToken = (laneId < kNWarpsPerBlock) ? warpLastToken[laneId] : 0;

            for (int delta = 1; delta < kNThreadsPerWarp; delta *= 2) {
                weight_t prev_gate = __shfl_up_sync(0xffffffff, warpAccGate, delta);
                weight_t prev_token = __shfl_up_sync(0xffffffff, warpAccToken, delta);

                if (laneId >= delta) {
                    warpAccToken = prev_token * warpAccGate + warpAccToken;
                    warpAccGate *= prev_gate;
                }
            }

            if (laneId < kNWarpsPerBlock) {
                warpLastGate[laneId] = warpAccGate;
                warpLastToken[laneId] = warpAccToken;
            }
        }

        __syncthreads();

        // Combine results from previous warps
        if (warpId > 0) {
            accToken = warpLastToken[warpId - 1] * accGate + accToken;
            accGate *= warpLastGate[warpId - 1];
        }

        // Store result
        result[result_idx] = accToken;

        // Backward pass: compute gate gradients
        if (backward) {
            int previous_l = seq_pos - 1;
            weight_t prev_h;

            if (previous_l >= 0 && previous_l < size_L) {
                int cached_idx = batch_idx * size_M * size_D * size_L + m_idx * size_D * size_L + d_idx * size_L + previous_l;
                prev_h = cached_output[cached_idx];
            } else {
                prev_h = initial_state[initial_state_offset];
            }

            weight_t contribution = prev_h * accToken;
            // TODO protentially use reduce sum by writing to external memory buffer. D is usually 1024 to 4096
            // gate_idx doesn't depend on  blockIdx.z, so entire z index will sequentially sum here.
            // The operation will sequentially sum D times.
            // may or may not be an issue, further profiling is needed
            atomicAdd(&gateGradOut[gate_idx], contribution);
        }

        // Update chunk accumulators
        if (tid == chunklen - 1 && warpId == kBlockLast) {
            chunkAccGate = accGate;
            chunkAccToken = accToken;
        }

        __syncthreads();
    }
}

// working on adapting the following macros
template <typename weight_t, typename torch_weight_t>
void
fmscan_launch(
    const at::Tensor &gates, // [B, M, L]
    const at::Tensor &tokens, // forward [B, D, L] or backward [B, M, D, L]
    const at::Tensor &initial_state, // [B, M, D]
    const at::Tensor &out, // [B, M, D, L]
    // for backward
    const void *cached_output, // [B, M, D, L]
    void *gateGradOut, // [B, M, L]
    const bool reverse,
    const bool backward
) {
    CHECK_STRIDE(tokens);
    CHECK_STRIDE(gates);

    const auto sizes = out.sizes();
    const int size_B = sizes[0];
    const int size_M = sizes[1];
    const int size_D = sizes[2];
    const int seqlen = sizes[3]; // size_L

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    dim3 grid(size_B, size_M, size_D);
    constexpr int kNThreadsPerWarp = 32;

    int kNWarpsPerBlock = 1;
    int kNChunksPerSequence = 1;

    if (seqlen>=32 && seqlen <=1024)
    {   
        kNWarpsPerBlock = seqlen / 32; // 32 threads per warp
    } else {
        kNWarpsPerBlock = 32;
        kNChunksPerSequence = seqlen / 1024;
    }
    int kNThreads = seqlen / kNChunksPerSequence;

    fm_scan<weight_t, kNThreadsPerWarp, kNWarpsPerBlock, kNChunksPerSequence, 
            size_B, size_M, size_D, seqlen, 
            reverse, backward><<<grid, kNThreads, kNWarpsPerBlock * sizeof(weight_t) * 2, stream>>>( 
        reinterpret_cast<const weight_t *>(gates.data_ptr<torch_weight_t>()), 
        reinterpret_cast<const weight_t *>(tokens.data_ptr<torch_weight_t>()), 
        reinterpret_cast<const weight_t *>(initial_state.data_ptr<torch_weight_t>()), 
        reinterpret_cast<weight_t *>(out.data_ptr<torch_weight_t>()), 
        reinterpret_cast<const weight_t *>(cached_output), 
        reinterpret_cast<weight_t *>(gateGradOut) 
        );
}

#define DISPATCH_FMSCAN(gates, ...) \
    if (gates.scalar_type() == at::ScalarType::BFloat16) { \
        fmscan_launch<__nv_bfloat16, at::BFloat16>(gates, __VA_ARGS__); \
    } else if (gates.scalar_type() == at::ScalarType::Half) { \
        fmscan_launch<__half, at::Half>(gates, __VA_ARGS__); \
    } else if (gates.scalar_type() == at::ScalarType::Float) { \
        fmscan_launch<float, float>(gates, __VA_ARGS__); \
    } else { \
        TORCH_CHECK(false && "Unsupported tensor dtype: expecting bfloat16, float16 or float32"); \
    }

at::Tensor
fm_scan_forward(const at::Tensor &gates, const at::Tensor &tokens, const at::Tensor &initial_state, const at::Tensor &out, const bool reverse) {
    TORCH_CHECK(tokens.is_cuda());
    TORCH_CHECK(gates.is_cuda());
    TORCH_CHECK(initial_state.is_cuda());
    TORCH_CHECK(tokens.is_contiguous());
    TORCH_CHECK(gates.is_contiguous());
    TORCH_CHECK(initial_state.is_contiguous());
    TORCH_CHECK(tokens.scalar_type() == gates.scalar_type());
    TORCH_CHECK(tokens.scalar_type() == out.scalar_type());
    TORCH_CHECK(tokens.scalar_type() == initial_state.scalar_type());

    DISPATCH_FMSCAN(gates, tokens, initial_state, out, nullptr, nullptr, reverse, false);
    return out;
}

//firstPropGrad is used as initial_state indicating gradient from the last step
void
fm_scan_backward(const at::Tensor &gates, const at::Tensor &output, const at::Tensor &firstPropGrad, const at::Tensor &outGrad, const at::Tensor& gateGradOut, const at::Tensor& tokenGradOut) {
    TORCH_CHECK(gates.is_cuda());
    TORCH_CHECK(output.is_cuda());
    TORCH_CHECK(outGrad.is_cuda());
    TORCH_CHECK(firstPropGrad.is_cuda());
    TORCH_CHECK(gateGradOut.is_contiguous());
    TORCH_CHECK(tokenGradOut.is_contiguous());
    TORCH_CHECK(gates.scalar_type() == output.scalar_type());
    TORCH_CHECK(gates.scalar_type() == firstPropGrad.scalar_type());
    TORCH_CHECK(gates.scalar_type() == outGrad.scalar_type());
    TORCH_CHECK(gates.scalar_type() == gateGradOut.scalar_type());
    TORCH_CHECK(gates.scalar_type() == tokenGradOut.scalar_type());
    TORCH_CHECK(gates.sizes() == gateGradOut.sizes()); //[B, L ,M]

    DISPATCH_FMSCAN(gates, outGrad, firstPropGrad, tokenGradOut, output.data_ptr(), gateGradOut.data_ptr(), true, true);
}


