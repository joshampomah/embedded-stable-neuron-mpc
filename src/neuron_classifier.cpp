#include "condensed_solver/neuron_classifier.hpp"
#include "condensed_solver/math_ops.hpp"
#include <algorithm>
#include <cmath>

namespace condensed {

using math::vec_zero;
using math::vec_copy;
using math::matvec;
using math::matvec_add;

void NeuronClassifier::classify(
    const Scalar* z_k,
    Scalar u_prev,
    Scalar u_min,
    Scalar u_max,
    Scalar delta_u_max,
    Scalar margin,
    const ModelWeights& weights)
{
    for (int step = 0; step < N; ++step) {
        int n_u = step + 1;

        // Compute rate-constrained u bounds for each position in the sequence
        Scalar u_lo[N], u_hi[N];
        for (int i = 0; i < n_u; ++i) {
            if (i == 0) {
                u_lo[i] = std::max(u_min, u_prev - delta_u_max);
                u_hi[i] = std::min(u_max, u_prev + delta_u_max);
            } else {
                u_lo[i] = std::max(u_min, u_lo[i - 1] - delta_u_max);
                u_hi[i] = std::min(u_max, u_hi[i - 1] + delta_u_max);
            }
        }

        for (int net_idx = 0; net_idx < N_NETWORKS; ++net_idx) {
            classify_network(
                weights.nets[step][net_idx],
                z_k, u_lo, u_hi, n_u, margin,
                step, net_idx
            );
        }
    }
}

// Fused interval arithmetic: compute a_min, a_max over u in [u_lo, u_hi]
// given base (constant part) and W_u (N_HIDDEN × n_u, column-major).
// Replaces: pos_W = max(W,0); neg_W = min(W,0); min = base + pos_W*ul + neg_W*uh; ...
// Single pass over W_u columns (halves flash reads, no temporaries).
static inline void fused_interval_bounds(
    Scalar* a_min, Scalar* a_max,
    const Scalar* base,
    const Scalar* W_u,
    const Scalar* u_lo, const Scalar* u_hi,
    int n_u)
{
    vec_copy(a_min, base, N_HIDDEN);
    vec_copy(a_max, base, N_HIDDEN);
    for (int k = 0; k < n_u; ++k) {
        const Scalar* col = W_u + k * N_HIDDEN;  // column-major
        Scalar lo_k = u_lo[k], hi_k = u_hi[k];
        for (int j = 0; j < N_HIDDEN; ++j) {
            Scalar w = col[j];
            if (w > Scalar(0)) {
                a_min[j] += w * lo_k;
                a_max[j] += w * hi_k;
            } else {
                a_min[j] += w * hi_k;
                a_max[j] += w * lo_k;
            }
        }
    }
}

// Fused Wx bounds: since Wx is non-negative, Wx*h_min gives min and Wx*h_max gives max.
// Single pass with two accumulators (replaces 2 separate matvecs).
static inline void fused_wx_bounds(
    Scalar* wx_min, Scalar* wx_max,
    const Scalar* Wx,
    const Scalar* h_min, const Scalar* h_max)
{
    vec_zero(wx_min, N_HIDDEN);
    vec_zero(wx_max, N_HIDDEN);
    for (int k = 0; k < N_HIDDEN; ++k) {
        const Scalar* col = Wx + k * N_HIDDEN;  // column-major
        Scalar hmin_k = h_min[k], hmax_k = h_max[k];
        for (int j = 0; j < N_HIDDEN; ++j) {
            Scalar w = col[j];
            wx_min[j] += w * hmin_k;
            wx_max[j] += w * hmax_k;
        }
    }
}

// Classify neurons based on pre-activation bounds
static inline void classify_neurons(
    const Scalar* a_min, const Scalar* a_max,
    Scalar margin,
    LayerClassification& res)
{
    for (int j = 0; j < N_HIDDEN; ++j) {
        if (a_min[j] > margin) {
            res.status[j] = static_cast<int8_t>(NeuronStatus::SAFE_ACTIVE);
        } else if (a_max[j] < -margin) {
            res.status[j] = static_cast<int8_t>(NeuronStatus::SAFE_INACTIVE);
        } else {
            res.status[j] = static_cast<int8_t>(NeuronStatus::AMBIGUOUS);
        }
        res.h_min[j] = std::max(a_min[j], Scalar(0));
        res.h_max[j] = std::max(a_max[j], Scalar(0));
    }
}

void NeuronClassifier::classify_network(
    const NetworkWeights& nw,
    const Scalar* z_k,
    const Scalar* u_lo,
    const Scalar* u_hi,
    int n_u,
    Scalar margin,
    int step,
    int net_idx)
{
    Scalar* c0 = c0_cache_[step][net_idx];

    // c0 = W0_z * z + b0
    // W0 is (N_HIDDEN × n_input) column-major. First N_STATE columns = W0_z.
    const Scalar* W0_z = nw.W0;  // columns 0..N_STATE-1
    const Scalar* W0_u = nw.W0 + N_STATE * N_HIDDEN;  // columns N_STATE..N_STATE+n_u-1

    // Products from zero, add bias separately (float32 accumulation order)
    matvec(c0, W0_z, z_k, N_HIDDEN, N_STATE);
    for (int j = 0; j < N_HIDDEN; ++j) c0[j] += nw.b0[j];

    // Fused interval arithmetic for layer 0: single pass over W0_u
    Scalar a_min[N_HIDDEN], a_max[N_HIDDEN];
    fused_interval_bounds(a_min, a_max, c0, W0_u, u_lo, u_hi, n_u);

    // Classify layer 0
    classify_neurons(a_min, a_max, margin, results_[step][net_idx][0]);

    // Internal layers
    const Scalar* prev_h_min = results_[step][net_idx][0].h_min;
    const Scalar* prev_h_max = results_[step][net_idx][0].h_max;

    for (int layer = 0; layer < N_INTERNAL; ++layer) {
        const Scalar* Wx = nw.Wx[layer];
        const Scalar* W_skip = nw.W_skip[layer];
        const Scalar* W_skip_z = W_skip;  // first N_STATE columns
        const Scalar* W_skip_u = W_skip + N_STATE * N_HIDDEN;  // columns N_STATE..

        Scalar* skip_const = skip_const_cache_[step][net_idx][layer];

        // skip_const = W_skip_z * z + bx + b_skip
        // Products from zero, add biases separately
        matvec(skip_const, W_skip_z, z_k, N_HIDDEN, N_STATE);
        for (int j = 0; j < N_HIDDEN; ++j)
            skip_const[j] += nw.bx[layer][j] + nw.b_skip[layer][j];

        // Fused Wx bounds: single pass (Wx non-negative)
        Scalar wx_min[N_HIDDEN], wx_max[N_HIDDEN];
        fused_wx_bounds(wx_min, wx_max, Wx, prev_h_min, prev_h_max);

        // Combine: a = wx + skip_const + interval(W_skip_u, u)
        // Fuse the skip_const addition with W_skip_u interval to avoid extra arrays
        Scalar a_min_l[N_HIDDEN], a_max_l[N_HIDDEN];
        for (int j = 0; j < N_HIDDEN; ++j) {
            a_min_l[j] = wx_min[j] + skip_const[j];
            a_max_l[j] = wx_max[j] + skip_const[j];
        }
        // Fuse W_skip_u interval directly into a_min_l/a_max_l
        for (int k = 0; k < n_u; ++k) {
            const Scalar* col = W_skip_u + k * N_HIDDEN;
            Scalar lo_k = u_lo[k], hi_k = u_hi[k];
            for (int j = 0; j < N_HIDDEN; ++j) {
                Scalar w = col[j];
                if (w > Scalar(0)) {
                    a_min_l[j] += w * lo_k;
                    a_max_l[j] += w * hi_k;
                } else {
                    a_min_l[j] += w * hi_k;
                    a_max_l[j] += w * lo_k;
                }
            }
        }

        int curr_layer = layer + 1;
        classify_neurons(a_min_l, a_max_l, margin,
                          results_[step][net_idx][curr_layer]);

        prev_h_min = results_[step][net_idx][curr_layer].h_min;
        prev_h_max = results_[step][net_idx][curr_layer].h_max;
    }
}

}  // namespace condensed
