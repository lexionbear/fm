#include <cuda.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#define CHECK_STRIDE(x) TORCH_CHECK(x.stride(-1) == 1 || x.size(-1) == 1);

template<typename weight_t, int N>
class UnalignedTuple {
public:
    static constexpr int Size = N;
    using Type = weight_t;

    weight_t data[N];

    __device__ void reverse() {
        #pragma unroll
        for (int i = 0; i < N/2; ++i) {
            weight_t temp = data[i];
            data[i] = data[N - (i+1)];
            data[N - (i+1)] = temp;
        }
    }
};

template<typename T, int N>
class alignas(16) AlignedTuple : public UnalignedTuple<T, N> {};

template <typename Tuple, int offset>
__device__ Tuple load_shifted_tuple(const Tuple* ptr, int index, int minIdx, int maxIdx) {
    using weight_t = typename Tuple::Type;

    const weight_t* rawPtr = reinterpret_cast<const weight_t *>(ptr);
    Tuple x;
    for (int i = 0; i < Tuple::Size; ++i) {
        const int idx = index * Tuple::Size + i + offset;
        if (idx >= minIdx * Tuple::Size && idx < maxIdx * Tuple::Size) {
            x.data[i] = rawPtr[idx];
        } else {
            x.data[i] = 0.0;
        }
    }
    return x;
}
//TODO, move L into the last dimension

// backward means definitely reverse
template <typename Tuple, typename weight_t, int kNThreadsPerWarp, int kNWarpsPerBlock, int kNChunksPerSequence, 
            bool backward, bool reverse,
            int size_L, int size_M, int size_D>
__global__ void fm_scan(
    const weight_t* __restrict__ gates,   // [B, L, M]
    const Tuple* __restrict__ tokens,  // [B, L, D_seg], or [B, L, M, D_seg] in backward
    const Tuple* __restrict__ initial_state, // [B, M, D_seg] -> initial memory in forward, pass through gradient from future chunks in backward
    Tuple* __restrict__ result,        // [B, L, M, D], memory state h in forward, token gradient in backward
    // backward
    const Tuple* __restrict__ cached_output,  // Only for backward, [B, L, M, D_seg] the memory states cached from forward pass
    weight_t* __restrict__ gateGradOut,   // Only for backward, [B, L, M, D_seg]
) {
    __shared__ weight_t warpLastGate[kNWarpsPerBlock];
    __shared__ Tuple warpLastToken[kNWarpsPerBlock];
    __shared__ weight_t chunkAccGate; 
    __shared__ Tuple chunkAccToken; 

    const int batch_idx = blockIdx.x;
    const int m_idx = blockIdx.y;
    const int d_chunk = blockIdx.z;

    const int tid = threadIdx.x;
    const int warpId = tid / kNThreadsPerWarp;
    const int laneId = tid % kNThreadsPerWarp;

    const int chunklen = blockDim.x; // number of threads per block = seqlen / kNChunksPerSequence
    constexpr int kBlockLast = kNWarpsPerBlock - 1;
    constexpr int kWarpLast = kNThreadsPerWarp - 1;
    constexpr int kThreadLast = Tuple::Size - 1;
    const weight_t kEmptyGate = 1.0;

    // Offsets for gates and tokens
    const int size_D_chunk = size_D / Tuple::Size;
    const int gate_offset = batch_idx * size_L * size_M + m_idx;
    const int result_offset = batch_idx * size_L * size_M * size_D_chunk + m_idx * size_D_chunk + d_chunk;

    // in backward, input is memory gradient in the shape of [B, L, M, D_seg], and the output is [B, L, M, D_seg] in forward
    const int token_offset = backward ? result_offset : batch_idx * size_L * size_D_chunk + d_chunk;
    const int initial_state_offset = batch_idx * size_M * size_D_chunk + m_idx * size_D_chunk + d_chunk

    for (int chunk = 0; chunk < kNChunksPerSequence; chunk++) {
        // points to the begining of the chunk, or the last index of the chunk in reverse
        const int seq_pos = reverse ? (kNChunksPerSequence - chunk) * chunklen - 1 - tid : chunk * chunklen + tid; 

        const int gate_idx = gate_offset + seq_pos * size_M;
        const int token_idx = token_offset + seq_pos * size_D_chunk;
        const int result_idx = result_offset + seq_pos * size_M * size_D_chunk;

        if (chunk) __syncthreads(); // wait for previous chunk to complete so shared memory is ready

        // reverse only affects the seq_pos, it should not affect hidden dimmension and gate dimensions
        // Load gate values (scalar per sequence position)
        weight_t gate;
        if (reverse) {
            // gate needs to shift the gate by 1 in reverse, and the first gate should be 1.0
            if (chunk == 0 && tid == 0) {
                gate = kEmptyGate;
            } else {
                // shift the gate by +1 * size_M
                gate = gates[gate_idx + size_M];
            }
        } else {
            gate = gates[gate_idx];
        }

        // Load token tuple
        Tuple token = tokens[token_idx];
        
        Tuple accToken;
        weight_t accGate = gate;

        if (chunk == 0 && tid == 0) {
            #pragma unroll
            for (int i = 0; i < Tuple::Size; ++i) {
                accToken.data[i] = initial_state[initial_state_offset].data[i] * gate + token.data[i];
            }
        } else {
            #pragma unroll
            for (int i = 0; i < Tuple::Size; ++i) {
                accToken.data[i] = chunkAccToken.data[i] * gate + token.data[i];
            }
            accGate = chunkAccGate * gate;
        }

        // Warp-level scan using shuffles
        #pragma unroll
        for (int delta = 1; delta < kNThreadsPerWarp; delta *= 2) {
            weight_t prev_gate = __shfl_up_sync(0xffffffff, accGate, delta);
            Tuple prev_token;
            #pragma unroll
            for (int i = 0; i < Tuple::Size; ++i) {
                prev_token.data[i] = __shfl_up_sync(0xffffffff, accToken.data[i], delta);
            }

            if (laneId >= delta) {
                #pragma unroll
                for (int i = 0; i < Tuple::Size; ++i) {
                    accToken.data[i] = prev_token.data[i] * accGate + accToken.data[i];
                }
                accGate *= prev_gate;
            }
        }

        __syncwarp();

        // Store warp results to shared memory
        if (laneId == kWarpLast) {
            warpLastGate[warpId] = accGate;
            warpLastToken[warpId] = accToken;
        }

        __syncthreads();

        // warp Block-level scan, support up to 32 warps, thus 32 x 32 = 1024 per chunk
        if (warpId == 0) {
            weight_t warpAccGate = (laneId < kNWarpsPerBlock) ? warpLastGate[laneId] : kEmptyGate;
            Tuple warpAccToken;
            if (laneId < kNWarpsPerBlock) warpAccToken = warpLastToken[laneId];

            #pragma unroll
            for (int delta = 1; delta < kNThreadsPerWarp; delta *= 2) {
                weight_t prev_gate = __shfl_up_sync(0xffffffff, warpAccGate, delta);
                Tuple prev_token;
                #pragma unroll
                for (int i = 0; i < Tuple::Size; ++i) {
                    prev_token.data[i] = __shfl_up_sync(0xffffffff, warpAccToken.data[i], delta);
                }

                if (laneId >= delta) {
                    #pragma unroll
                    for (int i = 0; i < Tuple::Size; ++i) {
                        warpAccToken.data[i] = prev_token.data[i] * warpAccGate + warpAccToken.data[i];
                    }
                    warpAccGate *= prev_gate;
                }
            }

            if (laneId < kNWarpsPerBlock) {
                warpLastGate[laneId] = warpAccGate;
                warpLastToken[laneId] = warpAccToken;
            }
        }

        __syncthreads();

        // Combine warp results
        if (warpId > 0) {
            #pragma unroll
            for (int i = 0; i < Tuple::Size; ++i) {
                accToken.data[i] = warpLastToken[warpId-1].data[i] * accGate + accToken.data[i];
            }
            accGate *= warpLastGate[warpId-1];
        }

        // Store result to global memory
        result[result_idx] = accToken;

        // TODO: fixing this part
        if (backward) {
            int idx = result_idx - size_M * size_D_chunk; // shift sequence by 1

            Tuple gateGrad;
            // in reverse mode, idx is descending, so check for min is enough
            if (idx >= result_offset) {
                gateGrad = cached_output[idx];
            } else {
                // set to initial state if it is out of range
                // this else in theory should only be triggered once, for the last thread and last warp
                #pragma unroll
                for (int i = 0; i < Tuple::Size; ++i) {
                    gateGrad.data[i] = initial_state[initial_state_offset].data[i];
                }
            }
            
            weight_t accGateGrad = 0.0;
            for (int i = 0; i < Tuple::Size; ++i) {
                accGateGrad += gateGrad.data[i]  * accToken.data[i];
            }
            gateGradOut[gate_idx] = accGateGrad;
        }

        if (tid == blockDim.x - 1 && warpId == kBlockLast) {
            chunkAccGate = accGate;
            chunkAccToken = accToken;
        }
    }
}



#define DISPATCH_SCAN_INNER(TupleT, backward, weight_t, kNStepsPerThread, kNThreadsPerWarp, kNWarpsPerBlock, kNChunksPerSequence, grid, kNThreads, stream, gates, tokens, out, output, gateGradOut, batch_stride, dim_stride, reverse) \
    scan<TupleT, kNThreadsPerWarp, kNWarpsPerBlock, kNChunksPerSequence, backward><<<grid, kNThreads, kNWarpsPerBlock * sizeof(weight_t) * 2, stream>>>( \
        reinterpret_cast<const TupleT *>(gates.data_ptr<torch_weight_t>()), \
        reinterpret_cast<const TupleT *>(tokens.data_ptr<torch_weight_t>()), \
        reinterpret_cast<TupleT *>(out.data_ptr<torch_weight_t>()), \
        reinterpret_cast<const TupleT *>(output), \
        reinterpret_cast<TupleT *>(gateGradOut), \
        batch_stride, dim_stride, reverse \
    );

#define DISPATCH_SCAN(weight_t, kNStepsPerThread, kNThreadsPerWarp, kNWarpsPerBlock, kNChunksPerSequence, grid, kNThreads, stream, gates, tokens, out, output, gateGradOut, batch_stride, dim_stride, reverse) \
    using AlignedT = AlignedTuple<weight_t, kNStepsPerThread>; \
    using UnalignedT = UnalignedTuple<weight_t, kNStepsPerThread>; \
    if (kNStepsPerThread == 4 && \
        ((long)gates.data_ptr()) % 16 == 0 && \
        ((long)tokens.data_ptr()) % 16 == 0 && \
        ((long)out.data_ptr()) % 16 == 0 && \
        ((long)output) % 16 == 0 && \
        ((long)gateGradOut) % 16 == 0) { \
        if (output) { \
            DISPATCH_SCAN_INNER(AlignedT, true, weight_t, kNStepsPerThread, kNThreadsPerWarp, kNWarpsPerBlock, kNChunksPerSequence, grid, kNThreads, stream, gates, tokens, out, output, gateGradOut, batch_stride, dim_stride, reverse); \
        } else { \
            DISPATCH_SCAN_INNER(AlignedT, false, weight_t, kNStepsPerThread, kNThreadsPerWarp, kNWarpsPerBlock, kNChunksPerSequence, grid, kNThreads, stream, gates, tokens, out, output, gateGradOut, batch_stride, dim_stride, reverse); \
        } \
    } else { \
        if (output) { \
            DISPATCH_SCAN_INNER(UnalignedT, true, weight_t, kNStepsPerThread, kNThreadsPerWarp, kNWarpsPerBlock, kNChunksPerSequence, grid, kNThreads, stream, gates, tokens, out, output, gateGradOut, batch_stride, dim_stride, reverse); \
        } else { \
            DISPATCH_SCAN_INNER(UnalignedT, false, weight_t, kNStepsPerThread, kNThreadsPerWarp, kNWarpsPerBlock, kNChunksPerSequence, grid, kNThreads, stream, gates, tokens, out, output, gateGradOut, batch_stride, dim_stride, reverse); \
        } \
    }

template <typename weight_t, typename torch_weight_t>
void
warpscan(
    const at::Tensor &gates,
    const at::Tensor &tokens,
    const at::Tensor &out,
    const void *output,
    void *gateGradOut,
    const bool reverse
) {
    const auto strides = tokens.strides();
    const int batch_stride = strides[0];
    const int dim_stride = strides[1];
    CHECK_STRIDE(tokens);
    CHECK_STRIDE(gates);

    const auto sizes = tokens.sizes();
    const int batch_size = sizes[0];
    const int dim = sizes[1];
    const int seqlen = sizes[2];

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    dim3 grid(batch_size, dim);
    constexpr int kNThreadsPerWarp = 32;

    if (seqlen == 32) {
        constexpr int kNStepsPerThread = 1;
        constexpr int kNWarpsPerBlock = 1;
        int kNThreads = seqlen / kNStepsPerThread;
        constexpr int kNChunksPerSequence = 1;
        DISPATCH_SCAN(weight_t, kNStepsPerThread, kNThreadsPerWarp, kNWarpsPerBlock,
            kNChunksPerSequence, grid, kNThreads, stream, gates, tokens, out, output, gateGradOut,
            batch_stride, dim_stride, reverse);
    } else if (seqlen == 64) {
        constexpr int kNStepsPerThread = 2;
        constexpr int kNWarpsPerBlock = 1;
        constexpr int kNChunksPerSequence = 1;
        int kNThreads = seqlen / kNStepsPerThread / kNChunksPerSequence;
        DISPATCH_SCAN(weight_t, kNStepsPerThread, kNThreadsPerWarp, kNWarpsPerBlock,
            kNChunksPerSequence, grid, kNThreads, stream, gates, tokens, out, output, gateGradOut,
            batch_stride, dim_stride, reverse);
    } else if (seqlen == 128) {
        constexpr int kNStepsPerThread = 1;
        constexpr int kNWarpsPerBlock = 4;
        int kNThreads = seqlen / kNStepsPerThread;
        constexpr int kNChunksPerSequence = 1;
        DISPATCH_SCAN(weight_t, kNStepsPerThread, kNThreadsPerWarp, kNWarpsPerBlock,
            kNChunksPerSequence, grid, kNThreads, stream, gates, tokens, out, output, gateGradOut,
            batch_stride, dim_stride, reverse);
    } else if (seqlen == 256) {
        constexpr int kNStepsPerThread = 1;
        constexpr int kNWarpsPerBlock = 8;
        int kNThreads = seqlen / kNStepsPerThread;
        constexpr int kNChunksPerSequence = 1;
        DISPATCH_SCAN(weight_t, kNStepsPerThread, kNThreadsPerWarp, kNWarpsPerBlock,
            kNChunksPerSequence, grid, kNThreads, stream, gates, tokens, out, output, gateGradOut,
            batch_stride, dim_stride, reverse);
    } else if (seqlen == 512) {
        constexpr int kNStepsPerThread = 1;
        constexpr int kNWarpsPerBlock = 16;
        int kNThreads = seqlen / kNStepsPerThread;
        constexpr int kNChunksPerSequence = 1;
        DISPATCH_SCAN(weight_t, kNStepsPerThread, kNThreadsPerWarp, kNWarpsPerBlock,
            kNChunksPerSequence, grid, kNThreads, stream, gates, tokens, out, output, gateGradOut,
            batch_stride, dim_stride, reverse);
    } else if (seqlen == 1024) {
        constexpr int kNStepsPerThread = 2;
        constexpr int kNWarpsPerBlock = 16;
        int kNThreads = seqlen / kNStepsPerThread;
        constexpr int kNChunksPerSequence = 1;
        DISPATCH_SCAN(weight_t, kNStepsPerThread, kNThreadsPerWarp, kNWarpsPerBlock,
            kNChunksPerSequence, grid, kNThreads, stream, gates, tokens, out, output, gateGradOut,
            batch_stride, dim_stride, reverse);
    } else if (seqlen == 2048) {
        constexpr int kNStepsPerThread = 2;
        constexpr int kNWarpsPerBlock = 32;
        int kNThreads = seqlen / kNStepsPerThread;
        constexpr int kNChunksPerSequence = 1;
        DISPATCH_SCAN(weight_t, kNStepsPerThread, kNThreadsPerWarp, kNWarpsPerBlock,
            kNChunksPerSequence, grid, kNThreads, stream, gates, tokens, out, output, gateGradOut,
            batch_stride, dim_stride, reverse);
    } else if (seqlen == 4096) {
        constexpr int kNStepsPerThread = 4;
        constexpr int kNWarpsPerBlock = 32;
        int kNThreads = seqlen / kNStepsPerThread;
        constexpr int kNChunksPerSequence = 1;
        DISPATCH_SCAN(weight_t, kNStepsPerThread, kNThreadsPerWarp, kNWarpsPerBlock,
            kNChunksPerSequence, grid, kNThreads, stream, gates, tokens, out, output, gateGradOut,
            batch_stride, dim_stride, reverse);
    } else if (seqlen == 8192) {
        constexpr int kNStepsPerThread = 4;
        constexpr int kNWarpsPerBlock = 32;
        constexpr int kNChunksPerSequence = 2;
        int kNThreads = seqlen / kNStepsPerThread / kNChunksPerSequence;
        DISPATCH_SCAN(weight_t, kNStepsPerThread, kNThreadsPerWarp, kNWarpsPerBlock,
            kNChunksPerSequence, grid, kNThreads, stream, gates, tokens, out, output, gateGradOut,
            batch_stride, dim_stride, reverse);
    } else if (seqlen == 16384) {
        constexpr int kNStepsPerThread = 4;
        constexpr int kNWarpsPerBlock = 32;
        constexpr int kNChunksPerSequence = 4;
        int kNThreads = seqlen / kNStepsPerThread / kNChunksPerSequence;
        DISPATCH_SCAN(weight_t, kNStepsPerThread, kNThreadsPerWarp, kNWarpsPerBlock,
            kNChunksPerSequence, grid, kNThreads, stream, gates, tokens, out, output, gateGradOut,
            batch_stride, dim_stride, reverse);
    } else if (seqlen == 32768) {
        constexpr int kNStepsPerThread = 4;
        constexpr int kNWarpsPerBlock = 32;
        constexpr int kNChunksPerSequence = 8;
        int kNThreads = seqlen / kNStepsPerThread / kNChunksPerSequence;
        DISPATCH_SCAN(weight_t, kNStepsPerThread, kNThreadsPerWarp, kNWarpsPerBlock,
            kNChunksPerSequence, grid, kNThreads, stream, gates, tokens, out, output, gateGradOut,
            batch_stride, dim_stride, reverse);
    } else if (seqlen == 65536) {
        constexpr int kNStepsPerThread = 4;
        constexpr int kNWarpsPerBlock = 32;
        constexpr int kNChunksPerSequence = 16;
        int kNThreads = seqlen / kNStepsPerThread / kNChunksPerSequence;
        DISPATCH_SCAN(weight_t, kNStepsPerThread, kNThreadsPerWarp, kNWarpsPerBlock,
            kNChunksPerSequence, grid, kNThreads, stream, gates, tokens, out, output, gateGradOut,
            batch_stride, dim_stride, reverse);
    } else {
        TORCH_CHECK(false && "seqlen must be a power of 2, >= 32, <= 65536");
    }
}

#define DISPATCH_WARPSCAN(gates, ...) \
    if (gates.scalar_type() == at::ScalarType::BFloat16) { \
        warpscan<__nv_bfloat16, at::BFloat16>(gates, __VA_ARGS__); \
    } else if (gates.scalar_type() == at::ScalarType::Half) { \
        warpscan<__half, at::Half>(gates, __VA_ARGS__); \
    } else if (gates.scalar_type() == at::ScalarType::Float) { \
        warpscan<float, float>(gates, __VA_ARGS__); \
    } else { \
        TORCH_CHECK(false && "Unsupported tensor dtype: expecting bfloat16, float16 or float32"); \
    }

at::Tensor
warpscan_forward(const at::Tensor &gates, const at::Tensor &tokens, const at::Tensor &out, const bool reverse) {
    TORCH_CHECK(tokens.is_cuda());
    TORCH_CHECK(gates.is_cuda());
    TORCH_CHECK(tokens.is_contiguous());
    TORCH_CHECK(gates.is_contiguous());
    TORCH_CHECK(tokens.scalar_type() == gates.scalar_type());
    TORCH_CHECK(tokens.scalar_type() == out.scalar_type());

    DISPATCH_WARPSCAN(gates, tokens, out, nullptr, nullptr, reverse);
    return out;
}

void
warpscan_backward(const at::Tensor &gates, const at::Tensor &output, const at::Tensor &outGrad, const at::Tensor& gateGradOut, const at::Tensor& tokenGradOut) {
    TORCH_CHECK(gates.is_cuda());
    TORCH_CHECK(output.is_cuda());
    TORCH_CHECK(outGrad.is_cuda());
    TORCH_CHECK(gateGradOut.is_contiguous());
    TORCH_CHECK(tokenGradOut.is_contiguous());
    TORCH_CHECK(gates.scalar_type() == output.scalar_type());
    TORCH_CHECK(gates.scalar_type() == outGrad.scalar_type());
    TORCH_CHECK(gates.scalar_type() == gateGradOut.scalar_type());
    TORCH_CHECK(gates.scalar_type() == tokenGradOut.scalar_type());
    TORCH_CHECK(gates.sizes() == gateGradOut.sizes()); //[B, L ,M]
    
    // outGrad [B, L, M, D], it is different from input [B, L, D]

    DISPATCH_WARPSCAN(gates, outGrad, tokenGradOut, output.data_ptr(), gateGradOut.data_ptr(), true);
}



// // Wrapper function to launch the kernel
// torch::Tensor modified_scan_cuda(
//     torch::Tensor gates,       // [B, L, M, 1]
//     torch::Tensor tokens,      // [B, L, 1, D]
//     bool reverse) {
//     CHECK_STRIDE(gates);
//     CHECK_STRIDE(tokens);

//     auto B = gates.size(0);
//     auto L = gates.size(1);
//     auto M = gates.size(2);
//     auto D = tokens.size(3);

//     constexpr int tuple_size = 4; // Adjust based on desired vectorization
//     using Tuple = AlignedTuple<float, tuple_size>;

//     auto options = torch::TensorOptions()
//         .dtype(tokens.dtype())
//         .device(tokens.device());
//     auto result = torch::empty({B, L, M, D}, options);

//     int batch_stride_gate = L * M;
//     int size_M = M;
//     int batch_stride_token = L * D;
//     int size_D = D;
//     int batch_stride_result = L * M * D;
//     int seq_stride_result = M * D;
//     int size_D = D;

//     dim3 grid(B, M, D / tuple_size);
//     dim3 block(128); // Adjust based on tuning

//     modified_scan<Tuple, 32, 4, 1, false><<<grid, block>>>(
//         gates.data_ptr<float>(),
//         reinterpret_cast<Tuple*>(tokens.data_ptr<float>()),
//         reinterpret_cast<Tuple*>(result.data_ptr<float>()),
//         nullptr,
//         nullptr,
//         batch_stride_gate,
//         size_M,
//         batch_stride_token,
//         size_D,
//         batch_stride_result,
//         seq_stride_result,
//         size_D,
//         reverse
//     );

//     return result;
// }