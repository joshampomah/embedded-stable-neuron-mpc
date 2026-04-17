#include "stable_neuron_solver/custom_ipm.hpp"
#include <cstring>
#include <cmath>

// Timer for sub-iteration profiling
#if defined(EMBEDDED_TARGET) && defined(__arm__)
extern "C" uint32_t get_microseconds();
#else
#include <chrono>
static uint32_t get_microseconds_ipm() {
    static auto start = std::chrono::high_resolution_clock::now();
    auto now = std::chrono::high_resolution_clock::now();
    return static_cast<uint32_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(now - start).count());
}
#define get_microseconds get_microseconds_ipm
#endif

namespace stable_neuron {

// Static member definitions for G sparsity structure (BSS, not heap).
int CustomIPM::g_nnz_total_ = 0;
int CustomIPM::col_nnz_[MAX_VARS];
int CustomIPM::col_start_[MAX_VARS + 1];
int CustomIPM::g_row_idx_[CustomIPM::MAX_G_NNZ];
int CustomIPM::row_start_[MAX_INEQ_IPM + 1];
int CustomIPM::g_col_idx_[CustomIPM::MAX_G_NNZ];

// ============================================================================
// Plain C math helpers (no Eigen)
// ============================================================================

static inline void vec_zero(Scalar* dst, int n) {
    std::memset(dst, 0, sizeof(Scalar) * n);
}

static inline void vec_copy(Scalar* dst, const Scalar* src, int n) {
    std::memcpy(dst, src, sizeof(Scalar) * n);
}

static inline Scalar vec_dot(const Scalar* a, const Scalar* b, int n) {
    Scalar s = 0;
    for (int i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

static inline void vec_axpy(Scalar* dst, Scalar alpha, const Scalar* src, int n) {
    for (int i = 0; i < n; ++i) dst[i] += alpha * src[i];
}

// NaN-aware: returns huge value if any element is NaN, so callers trip their
// convergence/feasibility checks rather than propagating silently.
static inline Scalar vec_inf_norm(const Scalar* v, int n) {
    Scalar mx = 0;
    for (int i = 0; i < n; ++i) {
        // NaN != NaN, so detect it explicitly
        if (v[i] != v[i]) return Scalar(1e30);
        Scalar a = v[i] < 0 ? -v[i] : v[i];
        if (a > mx) mx = a;
    }
    return mx;
}

static inline bool vec_has_nan(const Scalar* v, int n) {
    for (int i = 0; i < n; ++i) {
        if (v[i] != v[i]) return true;
    }
    return false;
}

// Matrix-vector: dst += M_cm' * v  (M is rows x cols column-major, dst is cols)
static void mat_t_vec_add(Scalar* dst, const Scalar* M_cm, const Scalar* v,
                          int rows, int cols) {
    for (int c = 0; c < cols; ++c) {
        Scalar s = 0;
        const Scalar* col = M_cm + c * rows;
        for (int r = 0; r < rows; ++r) s += col[r] * v[r];
        dst[c] += s;
    }
}

// Matrix-vector: dst += M_cm * v  (M is rows x cols column-major, dst is rows)
static void mat_vec_add(Scalar* dst, const Scalar* M_cm, const Scalar* v,
                        int rows, int cols) {
    for (int c = 0; c < cols; ++c) {
        Scalar vc = v[c];
        const Scalar* col = M_cm + c * rows;
        for (int r = 0; r < rows; ++r) dst[r] += col[r] * vc;
    }
}

// ============================================================================
// CustomIPM
// ============================================================================

CustomIPM::CustomIPM()
    : n_(0), p_(0), m_(0), p_orig_(0), m_orig_(0), initialized_(false),
      rho_cur_(0), delta_cur_(0)
{
    result_.status = 0;
    result_.iter = 0;
    result_.primal_obj = 0;
}

// Convert row-major QPData to internal column-major storage.
// P_rm is upper-triangular row-major; symmetrized + regularized.
// A_rm, G_rm are row-major.
// Equalities (A_rm, b) are converted to paired inequalities:
//   Ax = b  →  Ax ≤ b AND -Ax ≤ -b
// This avoids PMM's (1/delta)*A'A penalty in the KKT system.
void CustomIPM::store_problem(
    int p_orig, int m_orig,
    const Scalar* P_rm, const Scalar* q,
    const Scalar* A_rm, const Scalar* b,
    const Scalar* G_rm, const Scalar* h)
{
    const int n = n_;

    // Convert equalities to paired inequalities
    // Final: p_=0, m_= m_orig + 2*p_orig
    p_ = 0;
    m_ = m_orig + 2 * p_orig;

    // P: row-major upper triangle → column-major symmetric + PD regularization
    vec_zero(P_, n * n);
    for (int c = 0; c < n; ++c) {
        for (int r = 0; r <= c; ++r) {
            Scalar val = P_rm[r * n + c];
            P_[c * n + r] = val;
            P_[r * n + c] = val;
        }
    }
    for (int i = 0; i < n; ++i) {
        if (P_[i * n + i] < settings_.P_ZERO) {
            P_[i * n + i] = settings_.P_REG;
        }
    }

    vec_copy(q_, q, n);

    // G: row-major → column-major (original inequalities)
    if (m_orig > 0) {
        for (int c = 0; c < n; ++c)
            for (int r = 0; r < m_orig; ++r)
                G_[c * m_ + r] = G_rm[r * n + c];
        vec_copy(h_, h, m_orig);
    }

    // Append equality pairs: +A rows then -A rows
    if (p_orig > 0) {
        for (int c = 0; c < n; ++c) {
            for (int r = 0; r < p_orig; ++r) {
                Scalar val = A_rm[r * n + c];
                G_[c * m_ + m_orig + r] = val;            // +A row
                G_[c * m_ + m_orig + p_orig + r] = -val;  // -A row
            }
        }
        for (int i = 0; i < p_orig; ++i) {
            h_[m_orig + i] = b[i];           // +b
            h_[m_orig + p_orig + i] = -b[i]; // -b
        }
    }

    ruiz_equilibrate();
    analyze_sparsity();
}

void CustomIPM::ruiz_equilibrate() {
    const int n = n_, m = m_;
    const int ruiz_iters = settings_.ruiz_iter;

    // Initialize accumulated scalings to 1
    for (int i = 0; i < n; ++i) d_x_[i] = Scalar(1);
    for (int i = 0; i < m; ++i) d_g_[i] = Scalar(1);

    if (ruiz_iters <= 0) return;

    constexpr Scalar MIN_NORM = Scalar(1e-4);
    constexpr Scalar MAX_NORM = Scalar(1e4);

    for (int it = 0; it < ruiz_iters; ++it) {
        // Per-variable scaling: inf-norm of each variable's KKT column
        // KKT column j = [P col j; G col j] (A is empty since p_=0)
        Scalar dx[MAX_VARS];
        for (int j = 0; j < n; ++j) {
            Scalar norm = Scalar(0);
            // P column j (full symmetric matrix, column-major)
            const Scalar* Pj = P_ + j * n;
            for (int r = 0; r < n; ++r) {
                Scalar a = Pj[r];
                if (a < 0) a = -a;
                if (a > norm) norm = a;
            }
            // G column j
            const Scalar* Gj = G_ + j * m;
            for (int i = 0; i < m; ++i) {
                Scalar a = Gj[i];
                if (a < 0) a = -a;
                if (a > norm) norm = a;
            }
            // Limit and invert-sqrt
            if (norm < MIN_NORM) {
                dx[j] = Scalar(1);
            } else {
                if (norm > MAX_NORM) norm = MAX_NORM;
                dx[j] = Scalar(1) / std::sqrt(norm);
            }
        }

        // Per-constraint scaling: inf-norm of each G row
        Scalar dg[MAX_INEQ_IPM];
        for (int i = 0; i < m; ++i) {
            Scalar norm = Scalar(0);
            for (int c = 0; c < n; ++c) {
                Scalar a = G_[c * m + i];
                if (a < 0) a = -a;
                if (a > norm) norm = a;
            }
            if (norm < MIN_NORM) {
                dg[i] = Scalar(1);
            } else {
                if (norm > MAX_NORM) norm = MAX_NORM;
                dg[i] = Scalar(1) / std::sqrt(norm);
            }
        }

        // Scale P: P[r,c] *= dx[r] * dx[c]
        for (int c = 0; c < n; ++c) {
            Scalar dxc = dx[c];
            Scalar* Pc = P_ + c * n;
            for (int r = 0; r < n; ++r) {
                Pc[r] *= dx[r] * dxc;
            }
        }

        // Scale G: G[i,c] *= dg[i] * dx[c]
        for (int c = 0; c < n; ++c) {
            Scalar dxc = dx[c];
            Scalar* Gc = G_ + c * m;
            for (int i = 0; i < m; ++i) {
                Gc[i] *= dg[i] * dxc;
            }
        }

        // Scale q and h
        for (int j = 0; j < n; ++j) q_[j] *= dx[j];
        for (int i = 0; i < m; ++i) h_[i] *= dg[i];

        // Accumulate scaling factors
        for (int j = 0; j < n; ++j) d_x_[j] *= dx[j];
        for (int i = 0; i < m; ++i) d_g_[i] *= dg[i];
    }
}

void CustomIPM::analyze_sparsity() {
    // Build CSC and CSR indices for nonzero entries of G_ (column-major).
    // CSC (column → rows): used by sparse matvecs (compute_residuals, recover).
    // CSR (row → cols): used by build_kkt row-wise outer product.
    const int n = n_, m = m_;

    // --- CSC index ---
    int offset = 0;
    for (int c = 0; c < n; ++c) {
        col_start_[c] = offset;
        int nnz = 0;
        for (int i = 0; i < m; ++i) {
            if (G_[c * m + i] != Scalar(0)) {
                if (offset < MAX_G_NNZ) {
                    g_row_idx_[offset++] = i;
                }
                ++nnz;
            }
        }
        col_nnz_[c] = nnz;
    }
    col_start_[n] = offset;
    g_nnz_total_ = offset;

    // --- CSR index (transpose of CSC) ---
    // Count nonzeros per row
    int row_nnz[MAX_INEQ_IPM];
    for (int i = 0; i < m; ++i) row_nnz[i] = 0;
    for (int c = 0; c < n; ++c) {
        const int end = col_start_[c + 1];
        for (int idx = col_start_[c]; idx < end; ++idx) {
            row_nnz[g_row_idx_[idx]]++;
        }
    }
    // Prefix sum → row_start_
    row_start_[0] = 0;
    for (int i = 0; i < m; ++i) {
        row_start_[i + 1] = row_start_[i] + row_nnz[i];
    }
    // Fill column indices (scatter from CSC into CSR)
    int row_pos[MAX_INEQ_IPM];
    for (int i = 0; i < m; ++i) row_pos[i] = row_start_[i];
    for (int c = 0; c < n; ++c) {
        const int end = col_start_[c + 1];
        for (int idx = col_start_[c]; idx < end; ++idx) {
            int row = g_row_idx_[idx];
            g_col_idx_[row_pos[row]++] = c;
        }
    }
}

void CustomIPM::setup(int n, int p, int m,
                      const Scalar* P_rm, const Scalar* q,
                      const Scalar* A_rm, const Scalar* b,
                      const Scalar* G_rm, const Scalar* h)
{
    n_ = n;
    p_orig_ = p;
    m_orig_ = m;
    store_problem(p, m, P_rm, q, A_rm, b, G_rm, h);
    initialized_ = false;
}

void CustomIPM::update(const Scalar* P_rm, const Scalar* q,
                       const Scalar* A_rm, const Scalar* b,
                       const Scalar* G_rm, const Scalar* h)
{
    // Always cold-start: store new data and reinitialize.
    store_problem(p_orig_, m_orig_, P_rm, q, A_rm, b, G_rm, h);
    initialized_ = false;
}

void CustomIPM::initialize() {
    // Mehrotra's starting point heuristic:
    // x0 = -P^{-1} q  (approximate: use diagonal)
    // Then shift s and z to be positive.

    for (int i = 0; i < n_; ++i) {
        Scalar d = P_[i * n_ + i];
        x_[i] = (d > Scalar(1e-10)) ? -q_[i] / d : Scalar(0);
    }

    init_slacks_and_duals();
    initialized_ = true;
}

void CustomIPM::warm_start_x(const Scalar* x0) {
    // Use the provided x as starting point (from previous MPC timestep).
    // x0 is in original (unscaled) space; convert to scaled space: x̃ = x0/d_x.
    // The QP has changed (new Jacobians, possibly different constraints),
    // so we recompute slacks and duals for the new problem.
    for (int i = 0; i < n_; ++i)
        x_[i] = x0[i] / d_x_[i];
    init_slacks_and_duals();
    initialized_ = true;
}

void CustomIPM::init_slacks_and_duals() {
    // s = h - G*x, then shift to ensure s > 0
    vec_copy(s_, h_, m_);
    for (int c = 0; c < n_; ++c) {
        Scalar xc = x_[c];
        const Scalar* col = G_ + c * m_;
        for (int r = 0; r < m_; ++r) s_[r] -= col[r] * xc;
    }

    // Shift s to be strictly positive
    Scalar s_min = s_[0];
    for (int i = 1; i < m_; ++i)
        if (s_[i] < s_min) s_min = s_[i];

    Scalar shift_s = (s_min < Scalar(1e-4)) ? (-s_min + Scalar(1.0)) : Scalar(0);
    for (int i = 0; i < m_; ++i) s_[i] += shift_s;

    // z = 1 (uniform positive start)
    for (int i = 0; i < m_; ++i) z_[i] = Scalar(1.0);

    // y = 0
    vec_zero(y_, p_);

    // Use delta > 0 from the start: w = z/(s + delta*z) ≤ 1/delta.
    // This bounds the KKT condition number, preventing LDLT NaN in float32.
    rho_cur_ = settings_.rho_init;
    delta_cur_ = settings_.delta_init;
}

void CustomIPM::compute_residuals(Scalar& dual_inf_nr, Scalar& ineq_inf_nr) {
    const int n = n_, m = m_, p = p_;

    // r_dual = P*x + q  (symmetric P: upper triangle only, ~half the FMAs)
    for (int i = 0; i < n; ++i)
        r_dual_[i] = P_[i * n + i] * x_[i] + q_[i];   // diagonal + q
    for (int c = 1; c < n; ++c) {
        const Scalar xc = x_[c];
        const Scalar* Pc = P_ + c * n;  // column c
        Scalar acc = Scalar(0);
        for (int r = 0; r < c; ++r) {
            const Scalar prc = Pc[r];    // P[r,c] = P[c,r]
            r_dual_[r] += prc * xc;      // row r contribution from column c
            acc += prc * x_[r];           // row c contribution from column r
        }
        r_dual_[c] += acc;
    }

    if (m > 0) {
        // Fused r_ineq init + mu: one pass over m elements
        Scalar mu_acc = Scalar(0);
        for (int i = 0; i < m; ++i) {
            r_ineq_[i] = s_[i] - h_[i];
            mu_acc += s_[i] * z_[i];
        }
        mu_ = mu_acc / Scalar(m);

        // Fused G pass: compute G'z (into r_dual) and Gx (into r_ineq) simultaneously.
        // Uses CSC sparsity index — ~80% of G entries are zero.
        for (int c = 0; c < n; ++c) {
            Scalar xc = x_[c];
            const Scalar* Gc = G_ + c * m;
            Scalar acc = 0;
            const int end = col_start_[c + 1];
            for (int idx = col_start_[c]; idx < end; ++idx) {
                const int i = g_row_idx_[idx];
                Scalar gi = Gc[i];
                acc += gi * z_[i];       // G'z
                r_ineq_[i] += gi * xc;   // Gx
            }
            r_dual_[c] += acc;
        }
    } else {
        mu_ = Scalar(0);
    }

    // A'y (p_=0 after equality conversion, so this is dead code)
    if (p > 0) mat_t_vec_add(r_dual_, A_, y_, p, n);

    // r_eq = Ax - b
    if (p > 0) {
        vec_zero(r_eq_, p);
        mat_vec_add(r_eq_, A_, x_, p, n);
        vec_axpy(r_eq_, Scalar(-1), b_, p);
    }

    // Non-regularized residual norms (for convergence check)
    dual_inf_nr = vec_inf_norm(r_dual_, n);
    ineq_inf_nr = (m > 0) ? vec_inf_norm(r_ineq_, m) : Scalar(0);
}

void CustomIPM::build_kkt() {
    // K = P + rho*I + G' * diag(w) * G + (1/delta) * A'A
    // where w_i = z_i / (s_i + delta * z_i)  (PMM-modified)
    const int n = n_, m = m_;

    // Start with P + rho*I
    for (int c = 0; c < n; ++c) {
        for (int r = 0; r < n; ++r) {
            K_[c * n + r] = P_[c * n + r];
        }
        K_[c * n + c] += rho_cur_;
    }

    // += G' * diag(w) * G  using row-wise rank-1 updates.
    // For each row i of G, add w[i] * g_i * g_i' to K (symmetric).
    // Uses CSR index to iterate only nonzero columns per row.
    // Accumulation order per K[r,c] entry is identical to the merge-join
    // (both iterate rows in increasing order), preserving float32 precision.
    if (m > 0) {
        for (int i = 0; i < m; ++i) {
            Scalar wi = z_[i] / (s_[i] + delta_cur_ * z_[i]);
            if (!(wi >= Scalar(1e-8))) wi = Scalar(1e-8);
            w_[i] = wi;
        }

        for (int i = 0; i < m; ++i) {
            const Scalar wi = w_[i];
            const int rstart = row_start_[i];
            const int rend = row_start_[i + 1];
            const int ncols = rend - rstart;
            if (ncols == 0) continue;

            // Load nonzero G values for this row (indexed via CSR col indices)
            // gvals[j] = G_[col_j * m + i]
            const int* cols = g_col_idx_ + rstart;
            Scalar gvals[MAX_VARS];
            for (int j = 0; j < ncols; ++j) {
                gvals[j] = G_[cols[j] * m + i];
            }

            // Symmetric rank-1 update: K += w[i] * g * g'
            for (int j = 0; j < ncols; ++j) {
                const Scalar wgj = wi * gvals[j];
                const int cj = cols[j];
                // Diagonal
                K_[cj * n + cj] += wgj * gvals[j];
                // Off-diagonal (upper triangle, mirrored)
                for (int k = j + 1; k < ncols; ++k) {
                    const Scalar val = wgj * gvals[k];
                    const int ck = cols[k];
                    K_[ck * n + cj] += val;
                    K_[cj * n + ck] += val;
                }
            }
        }
    }

    // += (1/delta) * A'A
    if (p_ > 0) {
        Scalar inv_delta = Scalar(1) / delta_cur_;
        for (int c = 0; c < n; ++c) {
            const Scalar* Ac = A_ + c * p_;
            for (int r = 0; r <= c; ++r) {
                const Scalar* Ar = A_ + r * p_;
                Scalar acc = 0;
                for (int i = 0; i < p_; ++i) {
                    acc += Ar[i] * Ac[i];
                }
                K_[c * n + r] += inv_delta * acc;
                if (r != c) K_[r * n + c] += inv_delta * acc;
            }
        }
    }
}

bool CustomIPM::ldlt_factor() {
    // In-place LDLT of K_ (column-major, n x n), no pivoting.
    // After: lower triangle of K_ holds L (unit diagonal implicit),
    //        D_ holds diagonal.
    // MIN_DIAG prevents catastrophic LDLT instability: if a pivot D[j] falls
    // below this threshold, it's clamped. Too small → L factors explode (1/MIN_DIAG),
    // causing super-exponential r_dual divergence. 1e-6 matches PIQP's approach.
    constexpr Scalar MIN_DIAG = Scalar(1e-6);
    const int n = n_;

    for (int j = 0; j < n; ++j) {
        // Precompute tmp[k] = D[k] * L[j,k] for k < j.
        // This is reused across all i in the L[i,j] column below,
        // reducing the inner loop from 2 FMA cycles to 1 on Cortex-M4.
        Scalar tmp[MAX_VARS];
        Scalar dj = K_[j * n + j];
        for (int k = 0; k < j; ++k) {
            Scalar ljk = K_[k * n + j];  // L[j,k]
            tmp[k] = D_[k] * ljk;
            dj -= ljk * tmp[k];          // -= L[j,k]^2 * D[k]
        }

        if (dj < MIN_DIAG) dj = MIN_DIAG;
        D_[j] = dj;
        Scalar inv_dj = Scalar(1) / dj;
        inv_D_[j] = inv_dj;

        // L[i,j] = (K[i,j] - sum_{k<j} L[i,k] * tmp[k]) / D[j]
        for (int i = j + 1; i < n; ++i) {
            Scalar lij = K_[j * n + i];
            for (int k = 0; k < j; ++k) {
                lij -= K_[k * n + i] * tmp[k];
            }
            K_[j * n + i] = lij * inv_dj;
        }
    }
    return true;
}

void CustomIPM::ldlt_solve(const Scalar* rhs, Scalar* dx) {
    // Solve L*D*L'*dx = rhs
    const int n = n_;

    // 1. Forward: L * tmp = rhs  (L has unit diagonal, stored in lower K_)
    vec_copy(dx, rhs, n);
    for (int j = 0; j < n; ++j) {
        Scalar xj = dx[j];
        for (int i = j + 1; i < n; ++i) {
            dx[i] -= K_[j * n + i] * xj;
        }
    }

    // 2. Diagonal: D^{-1} * tmp  (uses precomputed inv_D_, avoids VDIV)
    for (int i = 0; i < n; ++i) {
        dx[i] *= inv_D_[i];
    }

    // 3. Backward: L' * dx = tmp2
    // Uses register accumulator to avoid load-modify-store of dx[j] per iteration.
    for (int j = n - 1; j >= 0; --j) {
        Scalar s = Scalar(0);
        for (int i = j + 1; i < n; ++i) {
            s += K_[j * n + i] * dx[i];
        }
        dx[j] -= s;
    }
}

void CustomIPM::kkt_matvec(const Scalar* v, Scalar* out) {
    // Compute K*v = (P + rho*I)*v + G'*diag(w)*G*v + (1/delta)*A'*A*v
    // from stored components — NOT from the factored K_ matrix.
    // This gives a full-precision matrix-vector product for iterative refinement.
    const int n = n_;
    const int m = m_;
    const int p = p_;

    // out = P*v + rho*v
    vec_zero(out, n);
    mat_vec_add(out, P_, v, n, n);
    for (int i = 0; i < n; ++i) out[i] += rho_cur_ * v[i];

    // += G'*diag(w)*G*v
    if (m > 0) {
        Scalar Gv[MAX_INEQ_IPM];
        vec_zero(Gv, m);
        mat_vec_add(Gv, G_, v, m, n);
        for (int i = 0; i < m; ++i) Gv[i] *= w_[i];
        mat_t_vec_add(out, G_, Gv, m, n);
    }

    // += (1/delta)*A'*A*v
    if (p > 0) {
        Scalar inv_delta = Scalar(1) / delta_cur_;
        Scalar Av[MAX_EQ];
        vec_zero(Av, p);
        mat_vec_add(Av, A_, v, p, n);
        for (int c = 0; c < n; ++c) {
            Scalar acc = 0;
            const Scalar* Ac = A_ + c * p;
            for (int i = 0; i < p; ++i) acc += Ac[i] * Av[i];
            out[c] += inv_delta * acc;
        }
    }
}

void CustomIPM::ldlt_solve_refined(const Scalar* rhs, Scalar* dx) {
    // Solve K*dx = rhs with one step of iterative refinement.
    // This improves accuracy from O(eps*cond(K)) to O(eps^2*cond(K)),
    // critical for float32 where eps≈1.2e-7 and cond(K) can reach 1e8+.
    const int n = n_;

    // 1. Initial solve
    ldlt_solve(rhs, dx);

    // 2. Compute residual: r = rhs - K*dx (using original components, not factored K_)
    Scalar Kx[MAX_VARS];
    kkt_matvec(dx, Kx);

    Scalar r[MAX_VARS];
    for (int i = 0; i < n; ++i) r[i] = rhs[i] - Kx[i];

    // 3. Solve for correction (reuse LDLT factorization)
    Scalar corr[MAX_VARS];
    ldlt_solve(r, corr);

    // 4. Apply correction
    for (int i = 0; i < n; ++i) dx[i] += corr[i];
}

Scalar CustomIPM::step_size(const Scalar* v, const Scalar* dv, int len) {
    // Max alpha in (0, 1] such that v + alpha*dv >= 0 for all i.
    // Returns RAW step size (no tau damping — caller applies tau where needed).
    // NaN-safe: NaN comparisons return false, so NaN entries are skipped.
    Scalar alpha = Scalar(1);
    for (int i = 0; i < len; ++i) {
        if (dv[i] < Scalar(0)) {
            Scalar a = -v[i] / dv[i];
            // NaN-safe: !(NaN < alpha) is true, so NaN a is skipped
            if (a < alpha) alpha = a;
        }
    }
    // Guard: alpha should be positive; if somehow 0 or negative, use tiny step
    if (!(alpha > Scalar(0))) alpha = Scalar(1e-10);
    return alpha;
}

int CustomIPM::solve() {
    if (!initialized_) initialize();

    const int n = n_;
    const int p = p_;
    const int m = m_;

    // Sub-iteration timing accumulators
    uint32_t acc_residuals = 0, acc_build_kkt = 0, acc_ldlt_factor = 0;
    uint32_t acc_ldlt_solve = 0, acc_recover = 0, acc_linesearch = 0;

    for (int iter = 0; iter < settings_.max_iter; ++iter) {
        // FTZ guard: Cortex-M4 flushes subnormals to 0.0, causing
        // division-by-zero (→ NaN) in w=z/s and dz=(rs-z*ds)/s.
        // Use NaN-safe comparisons: !(x >= floor) catches x<floor, x=NaN, x=-inf.
        if (m > 0) {
            constexpr Scalar SZ_FLOOR = Scalar(1e-12);
            for (int i = 0; i < m; ++i) {
                if (!(s_[i] >= SZ_FLOOR)) s_[i] = SZ_FLOOR;
                if (!(z_[i] >= SZ_FLOOR)) z_[i] = SZ_FLOOR;
            }
        }

        // --- Compute residuals ---
        // Returns non-reg norms, then adds proximal terms in-place
        uint32_t t_res0 = get_microseconds();
        Scalar dual_inf_nr, ineq_inf_nr;
        compute_residuals(dual_inf_nr, ineq_inf_nr);
        Scalar r_eq_norm = (p > 0) ? vec_inf_norm(r_eq_, p) : Scalar(0);
        acc_residuals += get_microseconds() - t_res0;

        // Store diagnostics (non-regularized norms)
        if (iter == 0) {
            result_.r_dual_0 = dual_inf_nr;
            result_.r_eq_0 = r_eq_norm;
            result_.r_ineq_0 = ineq_inf_nr;
            result_.mu_0 = mu_;
        }
        result_.r_dual_final = dual_inf_nr;
        result_.r_eq_final = r_eq_norm;
        result_.r_ineq_final = ineq_inf_nr;
        result_.mu_final = mu_;

        // PMM equality penalty: r_eq converges to O(delta), not O(eps_abs).
        // Use max(eps_abs, 10*delta_init) for equality tolerance.
        Scalar eq_tol = settings_.eps_abs;
        if (p > 0 && Scalar(10) * settings_.delta_init > eq_tol)
            eq_tol = Scalar(10) * settings_.delta_init;

        // Convergence on non-regularized residuals
        if (dual_inf_nr < settings_.eps_abs &&
            r_eq_norm < eq_tol &&
            ineq_inf_nr < settings_.eps_abs &&
            mu_ < settings_.eps_abs) {
            // Compute objective in scaled space (invariant: x̃'P̃x̃ = x'Px)
            Scalar Px[MAX_VARS];
            vec_zero(Px, n);
            mat_vec_add(Px, P_, x_, n, n);
            Scalar obj = Scalar(0.5) * vec_dot(x_, Px, n) + vec_dot(q_, x_, n);

            // Unscale x from Ruiz-scaled space to original space
            for (int i = 0; i < n; ++i) x_[i] *= d_x_[i];

            result_.status = 1;
            result_.iter = iter;
            result_.primal_obj = obj;
            result_.residuals_us = acc_residuals;
            result_.build_kkt_us = acc_build_kkt;
            result_.ldlt_factor_us = acc_ldlt_factor;
            result_.ldlt_solve_us = acc_ldlt_solve;
            result_.recover_us = acc_recover;
            result_.linesearch_us = acc_linesearch;
            return 1;
        }

        // --- Build KKT ---
        uint32_t t_kkt0 = get_microseconds();
        build_kkt();
        acc_build_kkt += get_microseconds() - t_kkt0;

        // --- LDLT factorization ---
        uint32_t t_ldlt0 = get_microseconds();
        if (!ldlt_factor()) {
            result_.status = -8;
            result_.iter = iter;
            result_.primal_obj = Scalar(1e30);
            result_.residuals_us = acc_residuals;
            result_.build_kkt_us = acc_build_kkt;
            result_.ldlt_factor_us = acc_ldlt_factor;
            result_.ldlt_solve_us = acc_ldlt_solve;
            result_.recover_us = acc_recover;
            result_.linesearch_us = acc_linesearch;
            return -8;
        }
        acc_ldlt_factor += get_microseconds() - t_ldlt0;

        // =================================================================
        // Predictor step (affine scaling direction)
        // =================================================================
        //
        // Schur complement RHS:
        //   rhs = -r_dual - G'*tmp - (1/delta)*A'*r_eq
        //
        // Predictor: rs = -s.*z → rs/z = -s
        //   delta>0: w*(r_ineq - s)  (consistent with bounded-w KKT)
        //   delta=0: -z + w*r_ineq   (equivalent but avoids cancellation)
        uint32_t t_rec0 = get_microseconds();
        for (int i = 0; i < n; ++i) rhs_[i] = -r_dual_[i];

        if (m > 0) {
            if (delta_cur_ > Scalar(0)) {
                for (int i = 0; i < m; ++i)
                    tmp_m_[i] = w_[i] * (r_ineq_[i] - s_[i]);
            } else {
                for (int i = 0; i < m; ++i)
                    tmp_m_[i] = -z_[i] + w_[i] * r_ineq_[i];
            }
            // rhs -= G' * tmp  (sparse)
            for (int c = 0; c < n; ++c) {
                Scalar acc = 0;
                const Scalar* Gc = G_ + c * m;
                const int end = col_start_[c + 1];
                for (int idx = col_start_[c]; idx < end; ++idx) {
                    const int i = g_row_idx_[idx];
                    acc += Gc[i] * tmp_m_[i];
                }
                rhs_[c] -= acc;
            }
        }

        if (p > 0) {
            Scalar inv_delta = Scalar(1) / delta_cur_;
            for (int c = 0; c < n; ++c) {
                Scalar acc = 0;
                const Scalar* Ac = A_ + c * p;
                for (int i = 0; i < p; ++i) acc += Ac[i] * r_eq_[i];
                rhs_[c] -= inv_delta * acc;
            }
        }
        acc_recover += get_microseconds() - t_rec0;

        // Solve predictor
        uint32_t t_slv0 = get_microseconds();
        if (settings_.iterative_refinement)
            ldlt_solve_refined(rhs_, dx_);
        else
            ldlt_solve(rhs_, dx_);
        acc_ldlt_solve += get_microseconds() - t_slv0;

        // Recover ds, dz (predictor) — ds-first from primal feasibility.
        // ds is exact (no division). dz uses w from build_kkt (bounded).
        uint32_t t_rec1 = get_microseconds();
        if (m > 0) {
            // ds = -r_ineq - G*dx  (fused: init + sparse scatter)
            for (int i = 0; i < m; ++i) ds_[i] = -r_ineq_[i];
            for (int c = 0; c < n; ++c) {
                Scalar dxc = dx_[c];
                const Scalar* Gc = G_ + c * m;
                const int end = col_start_[c + 1];
                for (int idx = col_start_[c]; idx < end; ++idx) {
                    const int i = g_row_idx_[idx];
                    ds_[i] -= Gc[i] * dxc;
                }
            }
            for (int i = 0; i < m; ++i)
                dz_[i] = -z_[i] - w_[i] * ds_[i];  // dz from complementarity

            // Save affine directions for corrector
            vec_copy(dz_aff_, dz_, m);
            vec_copy(ds_aff_, ds_, m);
        }

        // --- Affine step sizes ---
        Scalar alpha_s_aff = (m > 0) ? step_size(s_, ds_, m) : Scalar(1);
        Scalar alpha_z_aff = (m > 0) ? step_size(z_, dz_, m) : Scalar(1);

        // --- Centering parameter ---
        Scalar mu_aff = 0;
        if (m > 0) {
            for (int i = 0; i < m; ++i) {
                mu_aff += (s_[i] + alpha_s_aff * ds_[i]) *
                          (z_[i] + alpha_z_aff * dz_[i]);
            }
            mu_aff /= Scalar(m);
        }

        Scalar sigma_ratio = (mu_ > Scalar(1e-20)) ? (mu_aff / mu_) : Scalar(0);
        Scalar sigma = sigma_ratio * sigma_ratio * sigma_ratio;
        if (sigma > Scalar(1)) sigma = Scalar(1);

        // =================================================================
        // Corrector step
        // =================================================================
        // rs_corrected = -s.*z - ds_aff.*dz_aff + sigma*mu
        // K is the same, so we REUSE the LDLT factorization.

        // Corrector complementarity perturbation (consistent with w = z/(s+delta*z)):
        //   tmp = w * (r_ineq + rs_corrected/z)
        if (m > 0) {
            Scalar sigma_mu = sigma * mu_;
            for (int i = 0; i < m; ++i) {
                Scalar rs_i = -s_[i] * z_[i] - ds_aff_[i] * dz_aff_[i] + sigma_mu;
                tmp_m_[i] = w_[i] * (r_ineq_[i] + rs_i / z_[i]);
            }
        }

        // Build corrector RHS (same structure, different tmp)
        for (int i = 0; i < n; ++i) rhs_[i] = -r_dual_[i];

        if (m > 0) {
            // rhs -= G' * tmp  (sparse)
            for (int c = 0; c < n; ++c) {
                Scalar acc = 0;
                const Scalar* Gc = G_ + c * m;
                const int end = col_start_[c + 1];
                for (int idx = col_start_[c]; idx < end; ++idx) {
                    const int i = g_row_idx_[idx];
                    acc += Gc[i] * tmp_m_[i];
                }
                rhs_[c] -= acc;
            }
        }

        if (p > 0) {
            Scalar inv_delta = Scalar(1) / delta_cur_;
            for (int c = 0; c < n; ++c) {
                Scalar acc = 0;
                const Scalar* Ac = A_ + c * p;
                for (int i = 0; i < p; ++i) acc += Ac[i] * r_eq_[i];
                rhs_[c] -= inv_delta * acc;
            }
        }
        acc_recover += get_microseconds() - t_rec1;

        // Solve corrector (reuse LDLT factorization)
        uint32_t t_slv1 = get_microseconds();
        if (settings_.iterative_refinement)
            ldlt_solve_refined(rhs_, dx_);
        else
            ldlt_solve(rhs_, dx_);
        acc_ldlt_solve += get_microseconds() - t_slv1;

        // Recover ds, dz (corrector) — hybrid per-constraint recovery.
        // ds is always from primal feasibility (exact, no division).
        // dz uses whichever formula avoids dividing by a small quantity:
        //   Inactive (s >= z): dz = (rs - z*ds) / s  (standard, divides by large s)
        //   Active   (z > s):  dz = w*(-ds + rs/z)   (bounded w, divides by large z)
        // The active formula is algebraically equivalent to w*(G*dx + r_ineq + rs/z)
        // since ds = -r_ineq - G*dx → G*dx + r_ineq = -ds. This avoids a
        // separate G*dx pass — ds encodes the same information.
        uint32_t t_rec2 = get_microseconds();
        if (m > 0) {
            // ds = -r_ineq - G*dx  (fused: init + sparse scatter)
            for (int i = 0; i < m; ++i) ds_[i] = -r_ineq_[i];
            for (int c = 0; c < n; ++c) {
                Scalar dxc = dx_[c];
                const Scalar* Gc = G_ + c * m;
                const int end = col_start_[c + 1];
                for (int idx = col_start_[c]; idx < end; ++idx) {
                    const int i = g_row_idx_[idx];
                    ds_[i] -= Gc[i] * dxc;
                }
            }
            Scalar sigma_mu_c = sigma * mu_;
            for (int i = 0; i < m; ++i) {
                Scalar rs_i = -s_[i] * z_[i] - ds_aff_[i] * dz_aff_[i] + sigma_mu_c;
                if (s_[i] >= z_[i]) {
                    // Inactive: standard, divides by large s
                    dz_[i] = (rs_i - z_[i] * ds_[i]) / s_[i];
                } else {
                    // Active: derived from ds (G*dx + r_ineq = -ds)
                    dz_[i] = w_[i] * (-ds_[i] + rs_i / z_[i]);
                }
            }
        }
        acc_recover += get_microseconds() - t_rec2;

        // --- Step sizes + updates + delta scheduling ---
        uint32_t t_ls0 = get_microseconds();

        Scalar alpha_primal = settings_.tau *
            ((m > 0) ? step_size(s_, ds_, m) : Scalar(1));
        Scalar alpha_dual = settings_.tau *
            ((m > 0) ? step_size(z_, dz_, m) : Scalar(1));

        // --- Update variables ---
        vec_axpy(x_, alpha_primal, dx_, n);

        if (p > 0) {
            Scalar inv_delta = Scalar(1) / delta_cur_;
            vec_zero(dy_, p);
            mat_vec_add(dy_, A_, dx_, p, n);
            for (int i = 0; i < p; ++i) {
                y_[i] += inv_delta * (r_eq_[i] + alpha_primal * dy_[i]);
            }
        }
        if (m > 0) {
            vec_axpy(s_, alpha_primal, ds_, m);
            vec_axpy(z_, alpha_dual, dz_, m);
        }

        // --- Delta scheduling: decay delta with complementarity progress ---
        // Follows PIQP: delta *= (1 - mu_rate), where mu_rate is the fractional
        // complementarity reduction this iteration. The bounded-w perturbation
        // vanishes at convergence. rho is NOT decayed — holding it at rho_init
        // keeps LDLT pivots safely above the float32 clamp.
        if (m > 0) {
            Scalar mu_new = vec_dot(s_, z_, m) / Scalar(m);
            Scalar mu_rate = (mu_ > Scalar(1e-30))
                ? (mu_ - mu_new) / mu_
                : Scalar(0);
            if (mu_rate < Scalar(0)) mu_rate = Scalar(0);
            if (mu_rate > Scalar(1)) mu_rate = Scalar(1);

            delta_cur_ *= (Scalar(1) - mu_rate);
            if (delta_cur_ < settings_.reg_lower_limit)
                delta_cur_ = settings_.reg_lower_limit;
        }

        acc_linesearch += get_microseconds() - t_ls0;
    }

    // Max iterations reached
    // Compute objective in scaled space (invariant: x̃'P̃x̃ = x'Px)
    Scalar Px[MAX_VARS];
    vec_zero(Px, n);
    mat_vec_add(Px, P_, x_, n, n);
    Scalar obj = Scalar(0.5) * vec_dot(x_, Px, n) + vec_dot(q_, x_, n);

    // Unscale x from Ruiz-scaled space to original space
    for (int i = 0; i < n; ++i) x_[i] *= d_x_[i];

    result_.status = -1;
    result_.iter = settings_.max_iter;
    result_.primal_obj = obj;
    result_.residuals_us = acc_residuals;
    result_.build_kkt_us = acc_build_kkt;
    result_.ldlt_factor_us = acc_ldlt_factor;
    result_.ldlt_solve_us = acc_ldlt_solve;
    result_.recover_us = acc_recover;
    result_.linesearch_us = acc_linesearch;
    return -1;
}

}  // namespace stable_neuron
