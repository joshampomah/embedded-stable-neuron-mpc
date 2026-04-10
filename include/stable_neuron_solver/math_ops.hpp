#pragma once

// Lightweight C-style math operations for embedded (no Eigen dependency).
// All matrices are column-major: M[row, col] = M[col * rows + row].

#include "stable_neuron_solver/config.hpp"
#include <cstring>

namespace stable_neuron {

// Precision type (match types.hpp without pulling in Eigen)
#ifdef EMBEDDED_TARGET
using MathScalar = float;
#else
using MathScalar = double;
#endif

namespace math {

// --- Vector operations ---

static inline void vec_zero(MathScalar* dst, int n) {
    std::memset(dst, 0, sizeof(MathScalar) * n);
}

static inline void vec_copy(MathScalar* dst, const MathScalar* src, int n) {
    std::memcpy(dst, src, sizeof(MathScalar) * n);
}

// dst += alpha * src
static inline void vec_axpy(MathScalar* dst, MathScalar alpha, const MathScalar* src, int n) {
    for (int i = 0; i < n; ++i) dst[i] += alpha * src[i];
}

// dot product
static inline MathScalar vec_dot(const MathScalar* a, const MathScalar* b, int n) {
    MathScalar s = 0;
    for (int i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

// --- Matrix-vector operations (column-major) ---

// dst += M * v  (M: rows x cols column-major)
static inline void matvec_add(MathScalar* dst, const MathScalar* M, const MathScalar* v,
                               int rows, int cols) {
    for (int c = 0; c < cols; ++c) {
        MathScalar vc = v[c];
        const MathScalar* col = M + c * rows;
        for (int r = 0; r < rows; ++r) dst[r] += col[r] * vc;
    }
}

// dst = M * v  (M: rows x cols column-major)
static inline void matvec(MathScalar* dst, const MathScalar* M, const MathScalar* v,
                           int rows, int cols) {
    vec_zero(dst, rows);
    matvec_add(dst, M, v, rows, cols);
}

// --- Matrix-matrix operations (column-major) ---

// C += A * B  (A: M x K, B: K x N_cols, C: M x N_cols)
// Inner loop reads A columns sequentially (good for flash prefetch).
static inline void gemm_add(MathScalar* C, const MathScalar* A, const MathScalar* B,
                              int M, int K, int N_cols) {
    for (int j = 0; j < N_cols; ++j) {
        const MathScalar* B_col = B + j * K;
        MathScalar* C_col = C + j * M;
        for (int p = 0; p < K; ++p) {
            MathScalar b_pj = B_col[p];
            const MathScalar* A_col = A + p * M;
            for (int i = 0; i < M; ++i) {
                C_col[i] += b_pj * A_col[i];
            }
        }
    }
}

// C = A * B  (A: M x K, B: K x N_cols, C: M x N_cols)
static inline void gemm(MathScalar* C, const MathScalar* A, const MathScalar* B,
                          int M, int K, int N_cols) {
    vec_zero(C, M * N_cols);
    gemm_add(C, A, B, M, K, N_cols);
}

}  // namespace math
}  // namespace stable_neuron
