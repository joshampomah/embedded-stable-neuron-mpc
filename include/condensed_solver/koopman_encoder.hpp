#pragma once

// Koopman encoder: computes psi(z) = [z, proj(features(z))].
//
// The Lasso46 encoder extracts 46 handcrafted features from z = [y(15), u(15)]:
//   14 signed first differences, y_mean, u_sum,
//   7 absolute first differences, 9 absolute second differences,
//   12 quadratic y products, 1 cross-term (y*u), y_range.
// Then projects through a learned linear layer: phi = W @ features + b.
//
// All operations are pure arithmetic — no neural network forward pass.

#include "condensed_solver/koopman_types.hpp"

namespace condensed {

class KoopmanEncoder {
public:
    // Compute lifted state psi(z) = [z, phi(z)].
    // z:   input state, shape (KN_STATE,) = [y(15), u(15)]
    // psi: output lifted state, shape (KD_LIFT,)
    // weights: encoder projection weights
    void encode(const KScalar* z, KScalar* psi, const KoopmanWeights& weights) const;

private:
    // Compute the 46 handcrafted features from y and u.
    // y: (KN_STATE_Y,), u: (KN_STATE_U,), features: (KN_FEATURES,)
    void compute_features(const KScalar* y, const KScalar* u, KScalar* features) const;
};

}  // namespace condensed
