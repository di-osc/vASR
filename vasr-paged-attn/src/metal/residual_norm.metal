#include <metal_stdlib>
using namespace metal;

// Match the reduction tree in Candle 5404348 reduce.metal (MIT/Apache-2.0).
// Its 1024-wide RMSNorm dispatch uses 512 threads and preserves dtype
// multiplication boundaries after the F32 reduction.
struct RmsValue { uint count; float sum; };

inline RmsValue combine(RmsValue a, RmsValue b) {
    uint count = a.count + b.count;
    uint fraction = b.count / count;
    float delta = b.sum - a.sum;
    a.sum += b.sum + delta * delta * a.count * fraction;
    a.count = count;
    return a;
}

template <typename T>
kernel void residual_rmsnorm(
    device const T *input [[buffer(0)]],
    device const T *residual [[buffer(1)]],
    device const T *weight [[buffer(2)]],
    device T *output [[buffer(3)]],
    constant uint &count [[buffer(4)]],
    constant float &eps [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]]) {
    threadgroup RmsValue shared[512];
    threadgroup float scale;
    uint offset = row * 1024;
    RmsValue value = {0, 0};
    #pragma clang loop unroll(full)
    for (uint col = tid; col < 1024; col += 512) {
        // Explicit store boundary: the reference add rounds to T first.
        volatile T sum = static_cast<T>(input[offset + col] + residual[offset + col]);
        output[count + offset + col] = sum;
        float x = static_cast<float>(sum);
        value.sum += x * x;
        value.count += 1;
    }
    shared[tid] = value;
    threadgroup_barrier(mem_flags::mem_none);
    #pragma clang loop unroll(full)
    for (ushort s = 256; s >= 64; s >>= 1) {
        if (tid < s) shared[tid] = combine(shared[tid], shared[tid + s]);
        threadgroup_barrier(mem_flags::mem_none);
    }
    if (tid < 32) {
        value = combine(shared[tid], shared[tid + 32]);
        simdgroup_barrier(mem_flags::mem_none);
        #pragma clang loop unroll(full)
        for (ushort shift = 16; shift >= 1; shift >>= 1) {
            value = combine(value, RmsValue {
                simd_shuffle_down(value.count, shift),
                simd_shuffle_down(value.sum, shift),
            });
        }
    }
    if (tid == 0) scale = rsqrt(fast::divide(value.sum, 1024.0f) + eps);
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    #pragma clang loop unroll(full)
    for (uint col = tid; col < 1024; col += 512) {
        volatile T normalized = output[count + offset + col] * static_cast<T>(scale);
        output[offset + col] = static_cast<T>(normalized * weight[col]);
    }
}

template [[host_name("residual_rmsnorm_f32")]] kernel decltype(residual_rmsnorm<float>) residual_rmsnorm<float>;
template [[host_name("residual_rmsnorm_f16")]] kernel decltype(residual_rmsnorm<half>) residual_rmsnorm<half>;
#if defined(__HAVE_BFLOAT__)
template [[host_name("residual_rmsnorm_bf16")]] kernel decltype(residual_rmsnorm<bfloat>) residual_rmsnorm<bfloat>;
#endif
