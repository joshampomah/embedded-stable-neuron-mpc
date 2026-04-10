#pragma once

#include "condensed_solver/config.hpp"
#include "condensed_solver/types.hpp"
#include "condensed_solver/weights.hpp"
#include "condensed_solver/neuron_classifier.hpp"

namespace condensed {

// Builds the reduced QP with only ambiguous neurons as variables.
//
// Variable layout: [u(N), s_max(N), s_min(N), t(N), y_f1?(<=N), y_f2?(<=N),
//                   h_amb_1, h_amb_2, ...]
//
// Memory-efficient: uses ping-pong buffers for layer expression propagation
// instead of caching all layer_coeffs (saves ~115KB SRAM).
class QPBuilder {
public:
    QPBuilder() = default;

    // Configure parameters (call once at startup or when Q/R change)
    void configure(
        Scalar Q, Scalar R, Scalar R_delta,
        Scalar tube_weight, Scalar beta_0,
        Scalar u_min, Scalar u_max, Scalar delta_u_max,
        Scalar margin = DEFAULT_MARGIN
    );

    // Sub-timing breakdown of classify_and_setup (populated when called)
    struct ClassifyTiming {
        float classify_us;
        float layout_us;
        float neuron_exprs_us;
        float static_constraints_us;
    };

    // Classify neurons and precompute static QP parts.
    // Call when z_k or u_prev changes (new MPC timestep).
    void classify_and_setup(
        const Scalar* z_k,
        Scalar u_prev,
        const ModelWeights& weights
    );

    // Access last sub-timing (valid after classify_and_setup)
    const ClassifyTiming& classify_timing() const { return classify_timing_; }

    // Build the full QP (dynamic constraints depend on SCP iteration data).
    // Returns QPData with pointers to internal storage.
    QPData build_qp(
        const Scalar* z_k,
        Scalar u_prev,
        const Scalar* u_nominal,    // (N,)
        const Scalar* y_nominal,    // (N,)
        const Scalar* jacobians_f1, // Packed triangular: [J1(1), J2(2), J3(3), ...]
        const Scalar* jacobians_f2, // Same format
        const Scalar* f1_nominal,   // (N,)
        const Scalar* f2_nominal,   // (N,)
        const Scalar* W_bounds,     // (N, 2): [w_min, w_max] per step
        const Scalar* linear_jacobians  // Packed triangular, or nullptr
    );

    int n_vars() const { return n_vars_; }
    int n_amb() const { return n_amb_; }
    int n_amb_total() const { return n_amb_total_; }

    // Access classifier for cached z-dependent constants
    const NeuronClassifier& classifier() const { return classifier_; }

    // Scratch buffers available after build_qp() returns.
    // These reuse internal storage that is no longer needed:
    //   scratch_P()   -> layer_coeffs_[0]  (N_HIDDEN * MAX_VARS floats)
    //   scratch_G()   -> h_lo_             (MAX_INEQ floats, only used temporarily)
    //   scratch_Aeq() -> layer_coeffs_[1]  (N_HIDDEN * MAX_VARS floats)
    //
    // Note: scratch_G() needs (MAX_INEQ + 2*MAX_EQ) * MAX_VARS floats for G column-major.
    // The extra 2*MAX_EQ rows are for equality constraints converted to inequality pairs
    // (avoids PIQP's ill-conditioned (1/δ)A^TA KKT formulation on ARM float32).
    static constexpr int MAX_G_ROWS = MAX_INEQ + 2 * MAX_EQ;
    Scalar scratch_g_[MAX_G_ROWS * MAX_VARS];
    Scalar* scratch_P() { return layer_coeffs_[0]; }
    Scalar* scratch_G() { return scratch_g_; }
    Scalar* scratch_Aeq() { return layer_coeffs_[1]; }

private:
    // Build variable layout from classification
    void build_variable_layout();

    // Precompute neuron expressions with ping-pong memory
    void precompute_neuron_exprs(const Scalar* z_k, const ModelWeights& weights);

    // Precompute static constraints (P, q, equality, static inequality)
    void precompute_static_constraints(const Scalar* z_k, Scalar u_prev,
                                       const ModelWeights& weights);

    // Add epigraph constraints for ambiguous neurons of one network
    void add_icnn_epigraph(int step, int net_idx,
                           const NetworkWeights& nw,
                           const Scalar* z_k, int n_u,
                           int& row_out);

    // --- Parameters ---
    Scalar Q_, R_, R_delta_, tube_weight_, beta_0_;
    Scalar u_min_, u_max_, delta_u_max_, margin_;

    // --- Classification ---
    NeuronClassifier classifier_;

    // --- Variable layout ---
    int n_vars_;
    int n_amb_;
    int n_amb_total_;  // Total ambiguous (before capping at MAX_AMB_VARS)
    YFInfo y_f_info_[N][N_NETWORKS];
    // Ambiguous neuron variable index: -1 if not ambiguous
    int amb_var_idx_[N][N_NETWORKS][N_LAYERS][N_HIDDEN];

    // --- Neuron expressions (output only — layer exprs use ping-pong) ---
    // output_const_[step][net_idx]: constant part of network output
    Scalar output_const_[N][N_NETWORKS];
    // output_coeffs_[step][net_idx][var]: coefficients on QP variables
    Scalar output_coeffs_[N][N_NETWORKS][MAX_VARS];

    // --- Ping-pong layer buffers ---
    // Two buffers for layer expressions: (N_HIDDEN,) constants and (N_HIDDEN, MAX_VARS) coefficients
    Scalar layer_const_[2][N_HIDDEN];
    Scalar layer_coeffs_[2][N_HIDDEN * MAX_VARS];

    // --- Static QP matrices (persist between SCP iterations) ---
    Scalar P_[MAX_VARS * MAX_VARS];
    Scalar q_static_[MAX_VARS];
    Scalar A_eq_[MAX_EQ * MAX_VARS];
    Scalar b_eq_[MAX_EQ];
    int n_eq_;

    // Static constraints are built directly into A_ineq_/h_hi_ during
    // classify_and_setup(). build_qp() appends dynamic rows after n_static_ineq_.
    // This eliminates the separate A_static_ buffer (~10KB saved).
    int n_static_ineq_;
    int n_dynamic_ineq_;

    // --- Full QP output buffers ---
    Scalar A_ineq_[MAX_INEQ * MAX_VARS];
    Scalar h_lo_[MAX_INEQ];
    Scalar h_hi_[MAX_INEQ];
    Scalar q_[MAX_VARS];

    // --- Sub-timing ---
    ClassifyTiming classify_timing_;
};

}  // namespace condensed
