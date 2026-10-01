#include <metal_stdlib>
using namespace metal;

// Same A&S formula and float intermediates as Candle revision 5404348 unary.metal.
template <typename T> METAL_FUNC T audio_erf(T in) {
    constexpr float a1 = 0.254829592;
    constexpr float a2 = -0.284496736;
    constexpr float a3 = 1.421413741;
    constexpr float a4 = -1.453152027;
    constexpr float a5 = 1.061405429;
    constexpr float p = 0.3275911;
    float x = static_cast<float>(in);
    int sign = x < 0 ? -1 : 1;
    x = fabs(x);
    float t = 1.0 / (1.0 + p * x);
    float y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-x*x);
    return T(sign*y);
}

template <typename T>
kernel void bias_gelu(device const T *input [[buffer(0)]],
                      device const T *bias [[buffer(1)]],
                      device T *output [[buffer(2)]],
                      constant uint *params [[buffer(3)]],
                      uint i [[thread_position_in_grid]]) {
    if (i >= params[0]) return;
    // Keep the store/load rounding boundary of the original separate add kernel.
    T x = static_cast<T>(input[i] + bias[i % params[1]]);
    output[i] = static_cast<T>(x * (1 + audio_erf(x * M_SQRT1_2_F)) / 2);
}

template [[host_name("bias_gelu_f32")]] kernel decltype(bias_gelu<float>) bias_gelu<float>;
template [[host_name("bias_gelu_f16")]] kernel decltype(bias_gelu<half>) bias_gelu<half>;
#if defined(__HAVE_BFLOAT__)
template [[host_name("bias_gelu_bf16")]] kernel decltype(bias_gelu<bfloat>) bias_gelu<bfloat>;
#endif
