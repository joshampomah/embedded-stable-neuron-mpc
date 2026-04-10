#pragma once

#include "condensed_solver/config.hpp"
#include "condensed_solver/types.hpp"
#include "condensed_solver/weights.hpp"

namespace condensed {

// Neuron classification via interval arithmetic.
// Determines which ICNN neurons are always active, always inactive,
// or ambiguous over the rate-constrained control input set.
class NeuronClassifier {
public:
    NeuronClassifier() = default;

    // Classify all neurons for all (step, net, layer) combinations.
    // z_k: current state (N_STATE,)
    // u_prev: previous control input
    // u_min, u_max, delta_u_max: constraint parameters
    // margin: classification margin (neurons within margin of 0 are ambiguous)
    void classify(
        const Scalar* z_k,
        Scalar u_prev,
        Scalar u_min,
        Scalar u_max,
        Scalar delta_u_max,
        Scalar margin,
        const ModelWeights& weights
    );

    // Access classification results
    const LayerClassification& get(int step, int net_idx, int layer) const {
        return results_[step][net_idx][layer];
    }

    // Access cached z-dependent constants (computed during classify)
    // c0 = W0_z * z + b0 for layer 0
    const Scalar* get_c0(int step, int net_idx) const {
        return c0_cache_[step][net_idx];
    }
    // skip_const = W_skip_z * z + bx + b_skip for internal layers
    const Scalar* get_skip_const(int step, int net_idx, int layer) const {
        return skip_const_cache_[step][net_idx][layer];
    }

private:
    // Classify one network at one step
    void classify_network(
        const NetworkWeights& nw,
        const Scalar* z_k,
        const Scalar* u_lo,
        const Scalar* u_hi,
        int n_u,
        Scalar margin,
        int step,
        int net_idx
    );

    // Classification results: [step][net_idx][layer]
    LayerClassification results_[N][N_NETWORKS][N_LAYERS];

    // Cached z-dependent constants (populated during classify_network)
    // c0 = W0_z * z + b0  (N_HIDDEN per network)
    Scalar c0_cache_[N][N_NETWORKS][N_HIDDEN];
    // skip_const = W_skip_z * z + bx + b_skip  (N_HIDDEN per internal layer per network)
    Scalar skip_const_cache_[N][N_NETWORKS][N_INTERNAL][N_HIDDEN];
};

}  // namespace condensed
