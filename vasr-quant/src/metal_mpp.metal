// Test-only Metal 4 matrix candidates. Relaxed precision is explicitly separate.
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

template<typename T, int BN, bool RELAXED>
void multiply_nt(device T* input, device T* weight,
                 device T* output, constant uint* dims, uint2 group) {
    const int rows = int(dims[0]);
    const int columns_out = int(dims[1]);
    const int columns_in = int(dims[2]);
    auto left = tensor(input, dextents<int,2>{columns_in,rows}, array<int,2>{1,columns_in});
    auto right = tensor(weight, dextents<int,2>{columns_in,columns_out}, array<int,2>{1,columns_in});
    auto destination = tensor(output, dextents<int,2>{columns_out,rows}, array<int,2>{1,columns_out});
    constexpr auto descriptor = matmul2d_descriptor(64,BN,dynamic_length_v<int>,false,true,RELAXED,
                                                    matmul2d_descriptor::mode::multiply);
    matmul2d<descriptor,execution_simdgroups<4>> operation;
    auto left_tile = left.slice(0,int(group.y)*64);
    auto right_tile = right.slice(0,int(group.x)*BN);
    auto output_tile = destination.slice(int(group.x)*BN,int(group.y)*64);
    operation.run(left_tile,right_tile,output_tile);
}

#define ENTRY(T, BN, RELAXED, NAME) \
kernel void NAME(device T* x [[buffer(0)]], device T* w [[buffer(1)]], \
                 device T* y [[buffer(2)]], constant uint* dims [[buffer(3)]], \
                 uint2 group [[threadgroup_position_in_grid]]) { multiply_nt<T,BN,RELAXED>(x,w,y,dims,group); }
ENTRY(float,32,false,mpp_nt_f32_32)
ENTRY(float,64,false,mpp_nt_f32_64)
ENTRY(half,32,false,mpp_nt_f16_32)
ENTRY(half,64,false,mpp_nt_f16_64)
ENTRY(half,32,true,mpp_nt_f16_32_relaxed)
ENTRY(half,64,true,mpp_nt_f16_64_relaxed)
ENTRY(bfloat,32,false,mpp_nt_bf16_32)
ENTRY(bfloat,64,false,mpp_nt_bf16_64)
ENTRY(bfloat,32,true,mpp_nt_bf16_32_relaxed)
ENTRY(bfloat,64,true,mpp_nt_bf16_64_relaxed)
