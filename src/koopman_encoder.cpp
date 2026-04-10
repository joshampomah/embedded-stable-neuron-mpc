#include "stable_neuron_solver/koopman_encoder.hpp"
#include "stable_neuron_solver/math_ops.hpp"

#include <cmath>
#include <cstring>

namespace stable_neuron {

void KoopmanEncoder::compute_features(const KScalar* y, const KScalar* u,
                                       KScalar* features) const {
    int idx = 0;

    // 14 signed first differences: dy[i] = y[i+1] - y[i]
    for (int i = 0; i < 14; ++i) {
        features[idx++] = y[i + 1] - y[i];
    }

    // y_mean
    if (koopman_config::HAS_YMEAN) {
        KScalar sum = 0;
        for (int i = 0; i < KN_STATE_Y; ++i) sum += y[i];
        features[idx++] = sum / KScalar(KN_STATE_Y);
    }

    // u_sum
    if (koopman_config::HAS_USUM) {
        KScalar sum = 0;
        for (int i = 0; i < KN_STATE_U; ++i) sum += u[i];
        features[idx++] = sum;
    }

    // 7 absolute first differences at specific indices
    static constexpr int abs_dy_idx[] = {0, 1, 7, 10, 11, 12, 13};
    for (int k = 0; k < koopman_config::N_ABS_DIFFS; ++k) {
        int i = abs_dy_idx[k];
        features[idx++] = std::fabs(y[i + 1] - y[i]);
    }

    // 9 absolute second differences at specific indices
    static constexpr int abs_ddy_idx[] = {1, 2, 4, 6, 8, 9, 10, 11, 12};
    for (int k = 0; k < koopman_config::N_ABS_DDY; ++k) {
        int i = abs_ddy_idx[k];
        features[idx++] = std::fabs(y[i + 2] - KScalar(2) * y[i + 1] + y[i]);
    }

    // 12 quadratic y products
    static constexpr int quad_i[] = {10, 10, 10, 11, 11, 11, 12, 12, 12, 13, 13, 14};
    static constexpr int quad_j[] = {12, 13, 14, 11, 13, 14, 12, 13, 14, 13, 14, 14};
    for (int k = 0; k < koopman_config::N_QUAD_PAIRS; ++k) {
        features[idx++] = y[quad_i[k]] * y[quad_j[k]];
    }

    // 1 cross-term: y[12] * u[14]
    for (int k = 0; k < koopman_config::N_CROSS_TERMS; ++k) {
        features[idx++] = y[12] * u[14];
    }

    // y_range
    if (koopman_config::HAS_YRANGE) {
        KScalar ymin = y[0], ymax = y[0];
        for (int i = 1; i < KN_STATE_Y; ++i) {
            if (y[i] < ymin) ymin = y[i];
            if (y[i] > ymax) ymax = y[i];
        }
        features[idx++] = ymax - ymin;
    }
}

void KoopmanEncoder::encode(const KScalar* z, KScalar* psi,
                             const KoopmanWeights& weights) const {
    // psi = [z, phi(z)] where phi = proj(features(z))

    // Copy z into first N_STATE elements of psi
    std::memcpy(psi, z, KN_STATE * sizeof(KScalar));

    // Compute features
    KScalar features[KN_FEATURES];
    const KScalar* y = z;               // y = z[0:N_STATE_Y]
    const KScalar* u = z + KN_STATE_Y;  // u = z[N_STATE_Y:]
    compute_features(y, u, features);

    // Linear projection: phi = proj_weight @ features + proj_bias
    // proj_weight: (D_EXTRA x N_FEATURES), column-major
    KScalar* phi = psi + KN_STATE;  // phi goes after z in psi

    // Copy bias first, then accumulate W @ features
    std::memcpy(phi, weights.proj_bias, KD_EXTRA * sizeof(KScalar));
    math::matvec_add(phi, weights.proj_weight, features, KD_EXTRA, KN_FEATURES);
}

}  // namespace stable_neuron
