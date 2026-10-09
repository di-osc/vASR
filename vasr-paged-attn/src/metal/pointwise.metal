#include <metal_stdlib>
using namespace metal;

template <typename T>
kernel void silu_mul(device const T *gate [[buffer(0)]],
                     device const T *up [[buffer(1)]],
                     device T *output [[buffer(2)]],
                     constant uint *p [[buffer(3)]],
                     uint tid [[thread_position_in_grid]]) {
    const uint step = p[0] / 4 + (p[0] % 4 != 0);
    if (tid >= step) return;
    #pragma clang loop unroll(full)
    for (uint j = 0; j < 4; ++j) {
        uint i = tid + j * step;
        if (i >= p[0]) break;
        const uint row = i / p[1], column = i % p[1];
        T x = gate[row * p[2] + column];
        // Same expression as Candle 5404348 usilu, including T rounding.
        // Volatile keeps fast-math from reassociating the two kernels into
        // x * up / denominator; the reference stores SiLU before multiplication.
        volatile T activated = static_cast<T>(x / (1 + exp(-x)));
        output[i] = static_cast<T>(activated * up[row * p[3] + column]);
    }
}
template [[host_name("silu_mul_f32")]] kernel decltype(silu_mul<float>) silu_mul<float>;
template [[host_name("silu_mul_f16")]] kernel decltype(silu_mul<half>) silu_mul<half>;
#if defined(__HAVE_BFLOAT__)
template [[host_name("silu_mul_bf16")]] kernel decltype(silu_mul<bfloat>) silu_mul<bfloat>;
#endif
