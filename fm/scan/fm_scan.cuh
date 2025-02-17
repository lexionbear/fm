// inspired by the algorithm as https://github.com/proger/accelerated-scan
#include <cuda.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#define CHECK_STRIDE(x) TORCH_CHECK(x.stride(-1) == 1 || x.size(-1) == 1);

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <type_traits>  // For std::is_same

template <typename weight_t>
__device__ float cast_to_float(weight_t value) {
    if constexpr (std::is_same<weight_t, float>::value) {
        return value;
    } else if constexpr (std::is_same<weight_t, __half>::value) {
        return __half2float(value);
    } else if constexpr (std::is_same<weight_t, __nv_bfloat16>::value) {
        return __bfloat162float(value);
    } else {
        // Handle other types or static_assert if needed
        static_assert(sizeof(weight_t) == 0, "Unsupported type");
    }
}

template <typename weight_t>
__device__ weight_t cast_from_float(float value) {
    if constexpr (std::is_same<weight_t, float>::value) {
        return value;
    } else if constexpr (std::is_same<weight_t, __half>::value) {
        return __float2half(value);
    } else if constexpr (std::is_same<weight_t, __nv_bfloat16>::value) {
        return __float2bfloat16(value);
    } else {
        // Handle other types or static_assert if needed
        static_assert(sizeof(weight_t) == 0, "Unsupported type");
    }
}

template <typename weight_t, int kNThreadsPerWarp, int kNWarpsPerBlock, int kNChunksPerSequence, 
          int size_L, bool reverse, bool backward>
__global__ void fm_scan(
    const weight_t* __restrict__ gates,   // [B, M, L]
    const weight_t* __restrict__ tokens,  // Forward: [B, D, L], Backward: [B, M, D, L] grad input
    const weight_t* __restrict__ initial_state, // [B, M, D]
    weight_t* __restrict__ result,        // [B, M, D, L]
    
    // only for Backward
    const weight_t* __restrict__ cached_output,  // [B, M, D, L], from the forward "result"
    const weight_t* __restrict__ input_tokens, // [B, D, L] input tokens
    weight_t* __restrict__ gateGradOut,    // [B, M, D, L] keep D dim to avoid atomicAdd for now
    // Dimensions
    int size_B, int size_M, int size_D
) {
    __shared__ float warpLastGate[kNWarpsPerBlock];
    __shared__ float warpLastToken[kNWarpsPerBlock];
    __shared__ float chunkAccGate;
    __shared__ float chunkAccToken;

    const int batch_idx = blockIdx.x;
    const int m_idx = blockIdx.y;
    const int d_idx = blockIdx.z;

    const int tid = threadIdx.x;
    const int warpId = tid / kNThreadsPerWarp;
    const int laneId = tid % kNThreadsPerWarp;

    const int chunklen = blockDim.x;  // Threads per chunk (blockDim.x)
    constexpr int kBlockLast = kNWarpsPerBlock - 1;
    constexpr int kWarpLast = kNThreadsPerWarp - 1;
    const float kEmptyGate = 1.0;

    // Offsets
    const int gate_offset = batch_idx * size_M * size_L + m_idx * size_L;
    const int result_offset = batch_idx * size_M * size_D * size_L + m_idx * size_D * size_L + d_idx * size_L;
    const int input_token_offset = batch_idx * size_D * size_L + d_idx * size_L;
    const int token_offset = backward ? (batch_idx * size_M * size_D * size_L + m_idx * size_D * size_L + d_idx * size_L) 
                                      : input_token_offset;
    
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
        float gate;
        if (reverse) {
            if (chunk == 0 && tid == 0) {
                gate = kEmptyGate;
            } else {
                gate =  cast_to_float<weight_t>(gates[gate_idx + 1]);
            }
        } else {
            gate = cast_to_float<weight_t>(gates[gate_idx]);
        }

        // Load token 
        // Token = update weight * inputs
        // update_weight = 1 - gate 
        float token = backward ? cast_to_float<weight_t>(tokens[token_idx]) : cast_to_float<weight_t>(tokens[token_idx]) * (1.0 - gate);

        // Initialize accumulators
        float accToken, accGate;
        if (chunk == 0 && tid == 0) {
            accToken = cast_to_float<weight_t>(initial_state[initial_state_offset]) * gate + token;
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
            float prev_gate = __shfl_up_sync(0xffffffff, accGate, delta);
            float prev_token = __shfl_up_sync(0xffffffff, accToken, delta);

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
            float warpAccGate = (laneId < kNWarpsPerBlock) ? warpLastGate[laneId] : kEmptyGate;
            float warpAccToken = (laneId < kNWarpsPerBlock) ? warpLastToken[laneId] : 0.0;

            for (int delta = 1; delta < kNThreadsPerWarp; delta *= 2) {
                float prev_gate = __shfl_up_sync(0xffffffff, warpAccGate, delta);
                float prev_token = __shfl_up_sync(0xffffffff, warpAccToken, delta);

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
        result[result_idx] = cast_from_float<weight_t>(backward ? 
                            accToken * (1.0 - cast_to_float<weight_t>(gates[gate_idx])) : accToken);

        // Backward pass: compute gate gradients
        if (backward) {
            int previous_l = seq_pos - 1;
            float prev_h;

            if (previous_l >= 0 && previous_l < size_L) {
                int cached_idx = result_offset + previous_l;
                prev_h = cast_to_float<weight_t>(cached_output[cached_idx]);
            } else {
                prev_h = cast_to_float<weight_t>(initial_state[initial_state_offset]);
            }

            float contribution = (prev_h - cast_to_float<weight_t>(input_tokens[input_token_offset + seq_pos])) * accToken;
            // TODO protentially use reduce sum by writing to external memory buffer. D is usually 1024 to 4096
            // gate_idx doesn't depend on  blockIdx.z, so entire z index will sequentially sum here.
            // The operation will sequentially sum D times.
            // may or may not be an issue, further profiling is needed
            // atomicAdd(&gateGradOut[gate_idx], contribution);
            
            // store the full [B, M, D, L] tensor to avoid atomicAdd
            gateGradOut[result_idx] = cast_from_float<weight_t>(contribution);
        }

        // Update chunk accumulators
        if (tid == chunklen - 1 && warpId == kBlockLast) {
            chunkAccGate = accGate;
            chunkAccToken = accToken;
        }

        __syncthreads();
    }
}

// calling the kernel with correct constexprs
#define DISPATCH_FMSCAN_INNER(weight_t, kNThreadsPerWarp, seqlen, grid, stream, gates, tokens, out, initial_state, cached_output, input_tokens, gateGradOut, size_B, size_M, size_D, reverse, backward) \
    constexpr int kNWarpsPerBlock = seqlen <= 1024 ? seqlen / 32 : 32; \
    constexpr int kNChunksPerSequence = seqlen <= 1024 ? 1 : seqlen / 1024; \
    constexpr int kNThreads = seqlen / kNChunksPerSequence; \
    fm_scan<weight_t, kNThreadsPerWarp, kNWarpsPerBlock, kNChunksPerSequence, \
            seqlen, reverse, backward><<<grid, kNThreads, kNWarpsPerBlock * sizeof(float) * 2, stream>>>( \
        reinterpret_cast<const weight_t *>(gates.data_ptr<torch_weight_t>()), \
        reinterpret_cast<const weight_t *>(tokens.data_ptr<torch_weight_t>()), \
        reinterpret_cast<const weight_t *>(initial_state.data_ptr<torch_weight_t>()), \
        reinterpret_cast<weight_t *>(out.data_ptr<torch_weight_t>()), \
        reinterpret_cast<const weight_t *>(cached_output), \
        reinterpret_cast<const weight_t *>(input_tokens), \
        reinterpret_cast<weight_t *>(gateGradOut), \
        size_B, size_M, size_D \
        );

// define forward and backward
// currently doesn't support reverse = true and backward = false, but it is easy to add in the future when needed
#define DISPATCH_FMSCAN(weight_t, kNThreadsPerWarp, seqlen, grid, stream, gates, tokens, out, initial_state, cached_output, input_tokens, gateGradOut, size_B, size_M, size_D, reverse, backward) \
    if (backward) { \
        DISPATCH_FMSCAN_INNER(weight_t, kNThreadsPerWarp, seqlen, grid, stream, gates, tokens, out, initial_state, cached_output, input_tokens, gateGradOut, size_B, size_M, size_D, true, true); \
    } else { \
        DISPATCH_FMSCAN_INNER(weight_t, kNThreadsPerWarp, seqlen, grid, stream, gates, tokens, out, initial_state, cached_output, input_tokens, gateGradOut, size_B, size_M, size_D, false, false); \
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
    const void *input_tokens, // [B, D, L]
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

    if (seqlen == 32) {
        constexpr int _seqlen = 32;
        DISPATCH_FMSCAN(weight_t, kNThreadsPerWarp, _seqlen, grid, stream, 
                        gates, tokens, out, initial_state, cached_output, input_tokens, gateGradOut, size_B, size_M, size_D, reverse, backward)
    } else if (seqlen == 64) {
        constexpr int _seqlen = 64;
        DISPATCH_FMSCAN(weight_t, kNThreadsPerWarp, _seqlen, grid, stream, 
                        gates, tokens, out, initial_state, cached_output, input_tokens, gateGradOut, size_B, size_M, size_D, reverse, backward)
    } else if (seqlen == 128) {
        constexpr int _seqlen = 128;
        DISPATCH_FMSCAN(weight_t, kNThreadsPerWarp, _seqlen, grid, stream, 
                        gates, tokens, out, initial_state, cached_output, input_tokens, gateGradOut, size_B, size_M, size_D, reverse, backward)
    } else if (seqlen == 256) {
        constexpr int _seqlen = 256;
        DISPATCH_FMSCAN(weight_t, kNThreadsPerWarp, _seqlen, grid, stream, 
                        gates, tokens, out, initial_state, cached_output, input_tokens, gateGradOut, size_B, size_M, size_D, reverse, backward)
    } else if (seqlen == 512) {
        constexpr int _seqlen = 512;
        DISPATCH_FMSCAN(weight_t, kNThreadsPerWarp, _seqlen, grid, stream, 
                        gates, tokens, out, initial_state, cached_output, input_tokens, gateGradOut, size_B, size_M, size_D, reverse, backward)
    } else if (seqlen == 1024) {
        constexpr int _seqlen = 1024;
        DISPATCH_FMSCAN(weight_t, kNThreadsPerWarp, _seqlen, grid, stream, 
                        gates, tokens, out, initial_state, cached_output, input_tokens, gateGradOut, size_B, size_M, size_D, reverse, backward)
    } else if (seqlen == 2048) {
        constexpr int _seqlen = 2048;
        DISPATCH_FMSCAN(weight_t, kNThreadsPerWarp, _seqlen, grid, stream, 
                        gates, tokens, out, initial_state, cached_output, input_tokens, gateGradOut, size_B, size_M, size_D, reverse, backward)
    } else if (seqlen == 4096) {
        constexpr int _seqlen = 4096;
        DISPATCH_FMSCAN(weight_t, kNThreadsPerWarp, _seqlen, grid, stream, 
                        gates, tokens, out, initial_state, cached_output, input_tokens, gateGradOut, size_B, size_M, size_D, reverse, backward)
    } else if (seqlen == 8192) {
        constexpr int _seqlen = 8192;
        DISPATCH_FMSCAN(weight_t, kNThreadsPerWarp, _seqlen, grid, stream, 
                        gates, tokens, out, initial_state, cached_output, input_tokens, gateGradOut, size_B, size_M, size_D, reverse, backward)
    } else if (seqlen == 16384) {
        constexpr int _seqlen = 16384;
        DISPATCH_FMSCAN(weight_t, kNThreadsPerWarp, _seqlen, grid, stream, 
                        gates, tokens, out, initial_state, cached_output, input_tokens, gateGradOut, size_B, size_M, size_D, reverse, backward)
    } else if (seqlen == 32768) {
        constexpr int _seqlen = 32768;
        DISPATCH_FMSCAN(weight_t, kNThreadsPerWarp, _seqlen, grid, stream, 
                        gates, tokens, out, initial_state, cached_output, input_tokens, gateGradOut, size_B, size_M, size_D, reverse, backward)
    } else if (seqlen == 65536) {
        constexpr int _seqlen = 65536;
        DISPATCH_FMSCAN(weight_t, kNThreadsPerWarp, _seqlen, grid, stream, 
                        gates, tokens, out, initial_state, cached_output, input_tokens, gateGradOut, size_B, size_M, size_D, reverse, backward)
    } else if (seqlen == 131072) {
        constexpr int _seqlen = 131072;
        DISPATCH_FMSCAN(weight_t, kNThreadsPerWarp, _seqlen, grid, stream, 
                        gates, tokens, out, initial_state, cached_output, input_tokens, gateGradOut, size_B, size_M, size_D, reverse, backward)
    } else {
        // raise error
        TORCH_CHECK(false,  "seqlen must be a power of 2, >= 32, <= 131072");
    }
}

#define FMSCAN_LAUNCH(gates, ...) \
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
fm_scan_forward(const at::Tensor &gates, const at::Tensor &tokens, const at::Tensor &initial_state, const at::Tensor &out) {
    TORCH_CHECK(tokens.is_cuda());
    TORCH_CHECK(gates.is_cuda());
    TORCH_CHECK(initial_state.is_cuda());
    TORCH_CHECK(tokens.is_contiguous());
    TORCH_CHECK(gates.is_contiguous());
    TORCH_CHECK(initial_state.is_contiguous());
    TORCH_CHECK(tokens.scalar_type() == gates.scalar_type());
    TORCH_CHECK(tokens.scalar_type() == out.scalar_type());
    TORCH_CHECK(tokens.scalar_type() == initial_state.scalar_type());

    FMSCAN_LAUNCH(gates, tokens, initial_state, out, nullptr, nullptr, nullptr, false, false);
    return out;
}

//firstPropGrad is used as initial_state indicating gradient from the last step
void
fm_scan_backward(const at::Tensor &gates, const at::Tensor &outGrad, const at::Tensor &firstPropGrad, 
    const at::Tensor& tokenGradOut,
    const at::Tensor &cached_output, const at::Tensor &input_tokens, 
    const at::Tensor& gateGradOut
    ) {
    TORCH_CHECK(gates.is_cuda());
    TORCH_CHECK(cached_output.is_cuda());
    TORCH_CHECK(outGrad.is_cuda());
    TORCH_CHECK(firstPropGrad.is_cuda());
    TORCH_CHECK(input_tokens.is_cuda());
    TORCH_CHECK(gateGradOut.is_contiguous());
    TORCH_CHECK(tokenGradOut.is_contiguous());
    TORCH_CHECK(gates.scalar_type() == cached_output.scalar_type());
    TORCH_CHECK(gates.scalar_type() == firstPropGrad.scalar_type());
    TORCH_CHECK(gates.scalar_type() == outGrad.scalar_type());
    TORCH_CHECK(gates.scalar_type() == gateGradOut.scalar_type());
    TORCH_CHECK(gates.scalar_type() == tokenGradOut.scalar_type());

    FMSCAN_LAUNCH(gates, outGrad, firstPropGrad, tokenGradOut, cached_output.data_ptr(), input_tokens.data_ptr(), gateGradOut.data_ptr(), true, true);
}


