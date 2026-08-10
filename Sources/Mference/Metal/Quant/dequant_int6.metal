#include <metal_stdlib>
using namespace metal;

// MLX affine 6-bit weights are a dense little-endian bitstream in U32 words.
// Sixteen values occupy exactly three words. Scales and biases are BF16 with
// one pair per 128 input channels in the OptiQ checkpoint.
constant constexpr uint kInt6GroupSize = 128;
constant constexpr uint kInt6RowsPerTG = 8;

inline uint int6_value(device const uint* row, uint index) {
    const uint bit = index * 6u;
    const uint word = bit >> 5;
    const uint shift = bit & 31u;
    uint value = row[word] >> shift;
    if (shift > 26u) value |= row[word + 1u] << (32u - shift);
    return value & 0x3fu;
}

[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void dequant_int6_gemv_simd(
    device const uint* W       [[buffer(0)]],
    device const bfloat* scales [[buffer(1)]],
    device const bfloat* biases [[buffer(2)]],
    device const half* x        [[buffer(3)]],
    device half* y              [[buffer(4)]],
    constant uint& M            [[buffer(5)]],
    constant uint& N            [[buffer(6)]],
    constant uint& group_size   [[buffer(7)]],
    uint tg_idx                 [[threadgroup_position_in_grid]],
    uint sg_idx                 [[simdgroup_index_in_threadgroup]],
    uint lane                   [[thread_index_in_simdgroup]]) {
    const uint row = tg_idx * kInt6RowsPerTG + sg_idx;
    if (row >= M) return;
    const uint groups = N / group_size;
    const uint row_words = (N * 6u) / 32u;
    device const uint* w_row = W + row * row_words;
    device const bfloat* s_row = scales + row * groups;
    device const bfloat* b_row = biases + row * groups;

    float acc = 0.0f;
    const uint values_per_lane = group_size / 32u;
    for (uint g = 0; g < groups; ++g) {
        const uint base = g * group_size + lane * values_per_lane;
        float dot = 0.0f;
        float sum = 0.0f;
        for (uint j = 0; j < values_per_lane; ++j) {
            const float xv = float(x[base + j]);
            dot = fma(float(int6_value(w_row, base + j)), xv, dot);
            sum += xv;
        }
        acc = fma(float(s_row[g]), dot, acc);
        acc = fma(float(b_row[g]), sum, acc);
    }
    acc = simd_sum(acc);
    if (lane == 0) y[row] = half(acc);
}
