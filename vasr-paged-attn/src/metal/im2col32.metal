#include <metal_stdlib>
using namespace metal;

// Exact copies only: ushort preserves F16/BF16 bit patterns, including zero.
// p: channels, height, width, output_height, output_width, rows, N/C/H/W strides.
template<typename T>
void im2col32(device const T* input, device T* output,
              constant uint* p, uint2 pos) {
    const uint c = pos.x, row = pos.y;
    if (c >= p[0] || row >= p[5]) return;
    const uint plane = p[3] * p[4];
    const uint b = row / plane;
    const uint spatial = row % plane;
    const int y = int((spatial / p[4]) * 2) - 1;
    const int x = int((spatial % p[4]) * 2) - 1;
    const uint dst = (row * p[0] + c) * 9;
    const uint base = b * p[6] + c * p[7];
    #pragma unroll
    for (uint ky = 0; ky < 3; ++ky) {
        #pragma unroll
        for (uint kx = 0; kx < 3; ++kx) {
            const int sy = y + int(ky), sx = x + int(kx);
            T value = T(0);
            if (sy >= 0 && sy < int(p[1]) && sx >= 0 && sx < int(p[2]))
                value = input[base + uint(sy) * p[8] + uint(sx) * p[9]];
            output[dst + ky * 3 + kx] = value;
        }
    }
}

kernel void im2col32_f32(device const float* input [[buffer(0)]],
                         device float* output [[buffer(1)]],
                         constant uint* p [[buffer(2)]],
                         uint2 pos [[thread_position_in_grid]]) {
    im2col32(input, output, p, pos);
}
kernel void im2col32_16(device const ushort* input [[buffer(0)]],
                        device ushort* output [[buffer(1)]],
                        constant uint* p [[buffer(2)]],
                        uint2 pos [[thread_position_in_grid]]) {
    im2col32(input, output, p, pos);
}
