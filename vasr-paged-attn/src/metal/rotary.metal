#include <metal_stdlib>
using namespace metal;

template <typename T>
kernel void packed_rope(device const T *input [[buffer(0)]],
                        device const T *cos [[buffer(1)]],
                        device const T *sin [[buffer(2)]],
                        device T *output [[buffer(3)]],
                        constant uint *p [[buffer(4)]],
                        uint tid [[thread_position_in_grid]]) {
    const uint step = p[0] / 4 + (p[0] % 4 != 0);
    if (tid >= step) return;
    #pragma clang loop unroll(full)
    for (uint j = 0; j < 4; ++j) {
        const uint i = tid + j * step;
        if (i >= p[0]) break;
        // Store B,S,H,D; input retains its actual B,H,S,D strides.
        const uint d = i % p[3];
        const uint h = (i / p[3]) % p[2];
        const uint s = (i / p[3] / p[2]) % p[1];
        const uint b = i / p[3] / p[2] / p[1];
        const uint base = b * p[4] + h * p[5] + s * p[6];
        const uint paired = d < p[3] / 2 ? d + p[3] / 2 : d - p[3] / 2;
        const uint position = (b * p[1] + s) * p[3] + d;
        const T x = input[base + d * p[7]];
        const T other = input[base + paired * p[7]];
        const T rotated = d < p[3] / 2 ? static_cast<T>(-other) : other;
        // The reference stores both products before adding. Volatile preserves
        // F32/F16/BF16 rounding and prevents contraction into one FMA.
        volatile T left = static_cast<T>(x * cos[position]);
        volatile T right = static_cast<T>(rotated * sin[position]);
        output[i] = static_cast<T>(left + right);
    }
}
template [[host_name("packed_rope_f32")]] kernel decltype(packed_rope<float>) packed_rope<float>;
template [[host_name("packed_rope_f16")]] kernel decltype(packed_rope<half>) packed_rope<half>;
#if defined(__HAVE_BFLOAT__)
template [[host_name("packed_rope_bf16")]] kernel decltype(packed_rope<bfloat>) packed_rope<bfloat>;
#endif
