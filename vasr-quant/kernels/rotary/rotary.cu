#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

#include "cuda_compat.h"

namespace vasr {

template <typename scalar_t, bool IS_NEOX>
inline __device__ void apply_rotary_embedding(
    scalar_t *__restrict__ arr, const scalar_t *__restrict__ cos_ptr,
    const scalar_t *__restrict__ sin_ptr, int rot_offset, int rot_dim) {
  int x_index, y_index;
  scalar_t cos, sin;
  if (IS_NEOX) {
    x_index = rot_offset;
    y_index = rot_dim + rot_offset;
    cos = VLLM_LDG(cos_ptr + x_index);
    sin = VLLM_LDG(sin_ptr + x_index);
  } else {
    x_index = 2 * rot_offset;
    y_index = 2 * rot_offset + 1;
    cos = VLLM_LDG(cos_ptr + x_index / 2);
    sin = VLLM_LDG(sin_ptr + x_index / 2);
  }

  const scalar_t x = arr[x_index];
  const scalar_t y = arr[y_index];
  arr[x_index] = x * cos - y * sin;
  arr[y_index] = y * cos + x * sin;
}

template <typename scalar_t, bool IS_NEOX>
__global__ void rotary_embedding_kernel(
    const scalar_t *__restrict__ query, const scalar_t *__restrict__ key,
    const scalar_t *__restrict__ cos_cache,
    const scalar_t *__restrict__ sin_cache, scalar_t *__restrict__ query_out,
    scalar_t *__restrict__ key_out,
    const int rot_dim, const int64_t query_stride, const int64_t key_stride,
    const int num_heads, const int num_kv_heads, const int head_size) {
  const int token_idx = blockIdx.x;
  const scalar_t *cos_ptr = cos_cache + token_idx * rot_dim;
  const scalar_t *sin_ptr = sin_cache + token_idx * rot_dim;

  const int nq = num_heads * rot_dim;
  for (int i = threadIdx.x; i < nq; i += blockDim.x) {
    const int head_idx = i / rot_dim;
    const int64_t token_head = token_idx * query_stride + head_idx * head_size;
    const int rot_offset = i % rot_dim;
    query_out[token_head + rot_offset] = query[token_head + rot_offset];
    query_out[token_head + rot_dim + rot_offset] =
        query[token_head + rot_dim + rot_offset];
    apply_rotary_embedding<scalar_t, IS_NEOX>(query_out + token_head, cos_ptr,
                                              sin_ptr, rot_offset, rot_dim);
  }

  const int nk = num_kv_heads * rot_dim;
  for (int i = threadIdx.x; i < nk; i += blockDim.x) {
    const int head_idx = i / rot_dim;
    const int64_t token_head = token_idx * key_stride + head_idx * head_size;
    const int rot_offset = i % rot_dim;
    key_out[token_head + rot_offset] = key[token_head + rot_offset];
    key_out[token_head + rot_dim + rot_offset] =
        key[token_head + rot_dim + rot_offset];
    apply_rotary_embedding<scalar_t, IS_NEOX>(key_out + token_head, cos_ptr,
                                              sin_ptr, rot_offset, rot_dim);
  }
}

} // namespace vasr

#define CALL_ROTARY(T, IS_NEOX)                                                \
  vasr::rotary_embedding_kernel<T, IS_NEOX><<<grid, block, 0, custream>>>(     \
      reinterpret_cast<const T *>(query), reinterpret_cast<const T *>(key),    \
      reinterpret_cast<const T *>(cos_cache),                                  \
      reinterpret_cast<const T *>(sin_cache),                                  \
      reinterpret_cast<T *>(query_out), reinterpret_cast<T *>(key_out),        \
      rot_dim, query_stride, key_stride, num_heads, num_kv_heads, head_size);

extern "C" void vasr_rotary_embedding(
    const void *query, const void *key, const void *cos_cache,
    const void *sin_cache, void *query_out, void *key_out, int32_t is_neox,
    int32_t head_size, int64_t num_tokens, int32_t rot_dim, int32_t num_heads,
    int32_t num_kv_heads, int64_t query_stride, int64_t key_stride,
    uint32_t dtype, int64_t stream) {
  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * rot_dim, 512));
  const cudaStream_t custream = (cudaStream_t)stream;

  if (is_neox) {
    if (dtype == 0) {
      CALL_ROTARY(half, true);
    } else if (dtype == 1) {
      CALL_ROTARY(__nv_bfloat16, true);
    } else if (dtype == 2) {
      CALL_ROTARY(float, true);
    }
  } else {
    if (dtype == 0) {
      CALL_ROTARY(half, false);
    } else if (dtype == 1) {
      CALL_ROTARY(__nv_bfloat16, false);
    } else if (dtype == 2) {
      CALL_ROTARY(float, false);
    }
  }
}
