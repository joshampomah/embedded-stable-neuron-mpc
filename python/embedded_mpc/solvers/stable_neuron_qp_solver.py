# Canonical owner: embedded-condensed-mpc
"""stability-reduced QP solver for DC-MPC with stable neuron elimination.

Reduces the QP from ~670 variables to ~25-50 by classifying ICNN neurons as
stable-active (always on), stable-inactive (always off), or unstable (needs
epigraph variable). Only unstable neurons get QP variables.

Method:
  For each neuron, interval arithmetic over rate-constrained u bounds
  determines if the neuron's activation can flip. ~96% of neurons are stable,
  so the reduced QP is dramatically smaller.

Uses PIQP dense solver with warm-starting between SCP iterations.
Neuron expressions are vectorized as numpy arrays for fast matrix builds.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from embedded_mpc.solvers.direct_qp_solver import DirectQPSolution


# =============================================================================
# Neuron classification via interval arithmetic
# =============================================================================


def classify_neurons(
    weights_f1: List[List[np.ndarray]],
    weights_f2: List[List[np.ndarray]],
    n_state: int,
    z_k: np.ndarray,
    u_prev: float,
    u_min: float,
    u_max: float,
    delta_u_max: float,
    margin: float = 1e-4,
) -> dict:
    """Classify all ICNN neurons as stable-active, stable-inactive, or unstable.

    Returns:
        Dict keyed by (step, net_idx, layer) -> {
            'status': int8 array (1=stable-active, -1=stable-inactive, 0=unstable),
            'h_min': post-ReLU lower bounds,
            'h_max': post-ReLU upper bounds,
        }
    """
    N = len(weights_f1)
    result = {}

    for step in range(N):
        n_u = step + 1
        u_lo = np.empty(n_u)
        u_hi = np.empty(n_u)
        for i in range(n_u):
            if i == 0:
                u_lo[i] = max(u_min, u_prev - delta_u_max)
                u_hi[i] = min(u_max, u_prev + delta_u_max)
            else:
                u_lo[i] = max(u_min, u_lo[i - 1] - delta_u_max)
                u_hi[i] = min(u_max, u_hi[i - 1] + delta_u_max)

        for net_idx, weights in enumerate([weights_f1[step], weights_f2[step]]):
            _classify_network(
                weights, n_state, z_k, u_lo, u_hi, n_u, margin,
                step, net_idx, result,
            )

    return result


def _classify_network(
    weights, n_state, z_k, u_lo, u_hi, n_u, margin, step, net_idx, result
):
    """Classify neurons for one network at one step."""
    W0 = weights[0]
    b0 = weights[1]
    n_hidden = W0.shape[0]

    W0_z = W0[:, :n_state]
    W0_u = W0[:, n_state : n_state + n_u]
    const = W0_z @ z_k + b0

    pos_W = np.maximum(W0_u, 0)
    neg_W = np.minimum(W0_u, 0)
    a_min = const + (pos_W * u_lo + neg_W * u_hi).sum(axis=1)
    a_max = const + (pos_W * u_hi + neg_W * u_lo).sum(axis=1)

    status = np.zeros(n_hidden, dtype=np.int8)
    status[a_min > margin] = 1
    status[a_max < -margin] = -1

    result[(step, net_idx, 0)] = {
        "status": status,
        "h_min": np.maximum(a_min, 0.0),
        "h_max": np.maximum(a_max, 0.0),
    }

    # Internal layers
    n_internal = (len(weights) - 4) // 4
    prev_h_min = result[(step, net_idx, 0)]["h_min"]
    prev_h_max = result[(step, net_idx, 0)]["h_max"]

    for layer in range(n_internal):
        Wx = np.maximum(weights[2 + layer * 4], 0)
        bx = weights[2 + layer * 4 + 1]
        W_skip = weights[2 + layer * 4 + 2]
        b_skip = weights[2 + layer * 4 + 3]

        W_skip_z = W_skip[:, :n_state]
        W_skip_u = W_skip[:, n_state : n_state + n_u]

        wx_min = Wx @ prev_h_min
        wx_max = Wx @ prev_h_max

        skip_const = W_skip_z @ z_k + bx + b_skip
        pos_Ws = np.maximum(W_skip_u, 0)
        neg_Ws = np.minimum(W_skip_u, 0)
        skip_var_min = (pos_Ws * u_lo + neg_Ws * u_hi).sum(axis=1)
        skip_var_max = (pos_Ws * u_hi + neg_Ws * u_lo).sum(axis=1)

        a_min_l = wx_min + skip_const + skip_var_min
        a_max_l = wx_max + skip_const + skip_var_max

        status_l = np.zeros(n_hidden, dtype=np.int8)
        status_l[a_min_l > margin] = 1
        status_l[a_max_l < -margin] = -1

        h_min_l = np.maximum(a_min_l, 0.0)
        h_max_l = np.maximum(a_max_l, 0.0)

        result[(step, net_idx, layer + 1)] = {
            "status": status_l,
            "h_min": h_min_l,
            "h_max": h_max_l,
        }

        prev_h_min = h_min_l
        prev_h_max = h_max_l


# =============================================================================
# stability-reduced QP Builder (vectorized numpy)
# =============================================================================


class StableNeuronQPBuilder:
    """Builds a reduced QP with only unstable neurons as variables.

    Variable layout: [u(N), s_max(N), s_min(N), t(N), y_f1?(<=N), y_f2?(<=N),
                      h_amb_1, h_amb_2, ...]

    Neuron expressions are stored as numpy arrays:
      h_const: shape (n_hidden,) — constant parts
      h_coeffs: shape (n_hidden, n_vars) — coefficients on QP variables
    This enables vectorized layer propagation via matrix multiply.
    """

    def __init__(
        self,
        N: int,
        n_state: int,
        weights_f1: List[List[np.ndarray]],
        weights_f2: List[List[np.ndarray]],
        Q: float,
        R: float,
        R_delta: float = 0.0,
        tube_weight: float = 0.0,
        beta_0: float = 2.3,
        u_min: float = 0.0,
        u_max: float = 3.0,
        delta_u_max: float = 0.5,
        margin: float = 1e-4,
        pe_gamma: float = 0.0,
    ):
        self.N = N
        self.n_state = n_state
        self.weights_f1 = weights_f1
        self.weights_f2 = weights_f2
        self.Q = Q
        self.R = R
        self.R_delta = R_delta
        self.pe_gamma = pe_gamma
        self.tube_weight = tube_weight
        self.beta_0 = beta_0
        self.u_min = u_min
        self.u_max = u_max
        self.delta_u_max = delta_u_max
        self.margin = margin

        self.n_hidden = weights_f1[0][0].shape[0]
        self.n_layers = 1 + (len(weights_f1[0]) - 4) // 4

        self._classification = None
        self.n_vars = 0
        self._n_amb = 0
        self._setup_done = False

    def classify_and_setup(self, z_k: np.ndarray, u_prev: float):
        """Classify neurons and precompute coefficient expressions."""
        self._classification = classify_neurons(
            self.weights_f1, self.weights_f2, self.n_state, z_k,
            u_prev, self.u_min, self.u_max, self.delta_u_max, self.margin,
        )

        self._build_variable_layout()
        self._precompute_neuron_exprs(z_k)
        self._precompute_static_constraints(z_k, u_prev)
        self._setup_done = True

    def _build_variable_layout(self):
        """Build the dynamic variable layout based on classification."""
        N = self.N
        offset = 4 * N  # after u, s_max, s_min, t

        self._y_f_info = {}
        self._amb_vars = {}

        # y_f variables: only if last layer has unstable neurons
        for step in range(N):
            for net_idx in range(2):
                last_layer = self.n_layers - 1
                last_status = self._classification[(step, net_idx, last_layer)]["status"]
                if np.any(last_status == 0):
                    self._y_f_info[(step, net_idx)] = {"is_var": True, "var_idx": offset}
                    offset += 1
                else:
                    self._y_f_info[(step, net_idx)] = {"is_var": False, "var_idx": None}

        # unstable neuron variables
        for step in range(N):
            for net_idx in range(2):
                for layer in range(self.n_layers):
                    status = self._classification[(step, net_idx, layer)]["status"]
                    amb_mask = (status == 0)
                    for j in np.where(amb_mask)[0]:
                        self._amb_vars[(step, net_idx, layer, int(j))] = offset
                        offset += 1

        self.n_vars = offset
        self._n_amb = len(self._amb_vars)

    def _precompute_neuron_exprs(self, z_k: np.ndarray):
        """Precompute vectorized neuron expressions (bottom-up).

        For each (step, net_idx, layer), stores:
          h_const: shape (n_hidden,) — constant part of each neuron
          h_coeffs: shape (n_hidden, n_vars) — coefficients on QP variables

        Neuron output h[j] = h_const[j] + h_coeffs[j,:] @ x
        where x is the QP variable vector.
        """
        n_hidden = self.n_hidden
        n_vars = self.n_vars
        N = self.N

        self._layer_const = {}   # (step, net_idx, layer) -> (n_hidden,)
        self._layer_coeffs = {}  # (step, net_idx, layer) -> (n_hidden, n_vars)
        self._output_const = {}  # (step, net_idx) -> float
        self._output_coeffs = {} # (step, net_idx) -> (n_vars,)

        for step in range(N):
            n_u = step + 1
            for net_idx in range(2):
                weights = [self.weights_f1, self.weights_f2][net_idx][step]
                self._compute_layer_exprs_vec(step, net_idx, weights, z_k, n_u)

        # Precompute output expressions
        for step in range(N):
            for net_idx in range(2):
                weights = [self.weights_f1, self.weights_f2][net_idx][step]
                W_out = np.maximum(weights[-2], 0)  # (1, n_hidden)
                b_out = float(weights[-1][0])
                last_layer = self.n_layers - 1

                h_c = self._layer_const[(step, net_idx, last_layer)]
                h_v = self._layer_coeffs[(step, net_idx, last_layer)]

                # y = b_out + W_out @ h  (W_out is 1×n_hidden)
                w = W_out[0, :]  # (n_hidden,)
                self._output_const[(step, net_idx)] = b_out + w @ h_c
                self._output_coeffs[(step, net_idx)] = w @ h_v  # (n_vars,)

    def _compute_layer_exprs_vec(self, step, net_idx, weights, z_k, n_u):
        """Compute vectorized neuron expressions for all layers of one network."""
        n_state = self.n_state
        n_hidden = self.n_hidden
        n_vars = self.n_vars

        # === Layer 1 ===
        W0 = weights[0]
        b0 = weights[1]
        W0_z = W0[:, :n_state]
        W0_u = W0[:, n_state : n_state + n_u]
        c0 = W0_z @ z_k + b0  # (n_hidden,)

        status = self._classification[(step, net_idx, 0)]["status"]

        h_const = np.zeros(n_hidden)
        h_coeffs = np.zeros((n_hidden, n_vars))

        # Stable-active: h[j] = c0[j] + W0_u[j,:] @ u
        active = (status == 1)
        h_const[active] = c0[active]
        # W0_u[j, m] goes into h_coeffs[j, m] (u variables are indices 0..N-1)
        h_coeffs[np.ix_(active, np.arange(n_u))] = W0_u[active, :]

        # Stable-inactive: h[j] = 0 (already zero)

        # unstable: h[j] = 0 + 1.0 * x[var_idx]
        amb_mask = (status == 0)
        for j in np.where(amb_mask)[0]:
            var_idx = self._amb_vars[(step, net_idx, 0, int(j))]
            h_coeffs[j, var_idx] = 1.0

        self._layer_const[(step, net_idx, 0)] = h_const
        self._layer_coeffs[(step, net_idx, 0)] = h_coeffs

        # === Internal layers ===
        n_internal = (len(weights) - 4) // 4
        for layer in range(n_internal):
            Wx = np.maximum(weights[2 + layer * 4], 0)        # (n_hidden, n_hidden)
            bx = weights[2 + layer * 4 + 1]                    # (n_hidden,)
            W_skip = weights[2 + layer * 4 + 2]                # (n_hidden, n_state+n_u)
            b_skip = weights[2 + layer * 4 + 3]                # (n_hidden,)

            W_skip_z = W_skip[:, :n_state]
            W_skip_u = W_skip[:, n_state : n_state + n_u]
            skip_const = W_skip_z @ z_k + bx + b_skip  # (n_hidden,)

            curr_layer = layer + 1
            prev_c = self._layer_const[(step, net_idx, layer)]   # (n_hidden,)
            prev_v = self._layer_coeffs[(step, net_idx, layer)]  # (n_hidden, n_vars)

            # Pre-activation = Wx @ h_prev + skip_u @ u + skip_const
            # = Wx @ (prev_c + prev_v @ x) + skip_u @ u + skip_const
            # = (Wx @ prev_c + skip_const) + (Wx @ prev_v + skip_u_padded) @ x
            pre_const = Wx @ prev_c + skip_const  # (n_hidden,)
            pre_coeffs = Wx @ prev_v              # (n_hidden, n_vars)
            # Add skip connection u terms
            pre_coeffs[:, :n_u] += W_skip_u

            status_curr = self._classification[(step, net_idx, curr_layer)]["status"]

            h_const_l = np.zeros(n_hidden)
            h_coeffs_l = np.zeros((n_hidden, n_vars))

            # Stable-active: h = pre_activation (pass through)
            active = (status_curr == 1)
            h_const_l[active] = pre_const[active]
            h_coeffs_l[active, :] = pre_coeffs[active, :]

            # Stable-inactive: h = 0 (already zero)

            # unstable: h = x[var_idx]
            amb_mask = (status_curr == 0)
            for j in np.where(amb_mask)[0]:
                var_idx = self._amb_vars[(step, net_idx, curr_layer, int(j))]
                h_coeffs_l[j, var_idx] = 1.0

            self._layer_const[(step, net_idx, curr_layer)] = h_const_l
            self._layer_coeffs[(step, net_idx, curr_layer)] = h_coeffs_l

    def _precompute_static_constraints(self, z_k: np.ndarray, u_prev: float):
        """Precompute P, q, equality constraints, and static inequality rows.

        These don't change between SCP iterations (only DC bounds and tracking
        constraints depend on y_nominal, Jacobians, W_bounds).
        """
        N = self.N
        n_vars = self.n_vars

        # === Cost matrix P (dense, upper triangular) ===
        P = np.zeros((n_vars, n_vars))

        # Q on t variables
        for i in range(N):
            P[3*N + i, 3*N + i] = 2.0 * self.Q

        # R on u variables
        for i in range(N):
            P[i, i] = 2.0 * self.R

        # Tube weight: (s_max - s_min)^2 = s_max^2 - 2*s_max*s_min + s_min^2
        for i in range(N):
            si, sj = N + i, 2*N + i
            P[si, si] += 2.0 * self.tube_weight
            P[sj, sj] += 2.0 * self.tube_weight
            # Cross terms: only store upper triangle
            if si < sj:
                P[si, sj] -= 2.0 * self.tube_weight
            else:
                P[sj, si] -= 2.0 * self.tube_weight

        # R_delta: (u[i] - u[i-1])^2
        if self.R_delta > 0:
            for i in range(1, N):
                P[i, i] += 2.0 * self.R_delta
                P[i-1, i-1] += 2.0 * self.R_delta
                # Cross term: u[i-1] < u[i] so upper triangle at [i-1, i]
                P[i-1, i] -= 2.0 * self.R_delta

        # PE excitation reward: -gamma * (u[i] - u[i-1])^2
        # Same tridiagonal structure as R_delta but with opposite sign
        if self.pe_gamma > 0:
            for i in range(1, N):
                P[i, i] -= 2.0 * self.pe_gamma
                P[i-1, i-1] -= 2.0 * self.pe_gamma
                P[i-1, i] += 2.0 * self.pe_gamma

        # Store upper triangular
        self._P = np.triu(P)

        # === Linear cost q ===
        self._q_static = np.zeros(n_vars)
        if self.R_delta > 0:
            self._q_static[0] -= 2.0 * self.R_delta * u_prev
        # PE: opposite sign to R_delta for (u[0] - u_prev)^2 cross-term
        if self.pe_gamma > 0:
            self._q_static[0] += 2.0 * self.pe_gamma * u_prev

        # === Equality constraints (y_f = output expression) ===
        eq_rows = []
        for step in range(N):
            for net_idx in range(2):
                info = self._y_f_info[(step, net_idx)]
                if not info["is_var"]:
                    continue

                y_idx = info["var_idx"]
                out_c = self._output_const[(step, net_idx)]
                out_v = self._output_coeffs[(step, net_idx)]

                # y_f - out_v @ x = out_c
                row = np.zeros(n_vars)
                row[y_idx] = 1.0
                row -= out_v
                # But y_idx coefficient should be 1.0, not 1.0 - out_v[y_idx]
                # since out_v[y_idx] should be 0 (y_f doesn't appear in output expr)
                eq_rows.append((row, out_c))

        self._n_eq = len(eq_rows)
        if self._n_eq > 0:
            self._A_eq = np.array([r[0] for r in eq_rows])
            self._b_eq = np.array([r[1] for r in eq_rows])
        else:
            self._A_eq = np.zeros((0, n_vars))
            self._b_eq = np.array([])

        # === Static inequality constraints ===
        # Count: u bounds (2N) + rate (2N) + tube (3N) + tracking (2N)
        #       + epigraph (2 * n_amb) + DC bounds (2N) [dynamic]
        # We build static rows here, dynamic (DC + tracking) in build_qp

        static_rows = []
        static_rhs = []

        # u bounds: -u[i] <= -u_min, u[i] <= u_max
        for i in range(N):
            row = np.zeros(n_vars)
            row[i] = -1.0
            static_rows.append(row)
            static_rhs.append(-self.u_min)

            row = np.zeros(n_vars)
            row[i] = 1.0
            static_rows.append(row)
            static_rhs.append(self.u_max)

        # Rate constraints
        row = np.zeros(n_vars); row[0] = 1.0
        static_rows.append(row); static_rhs.append(u_prev + self.delta_u_max)
        row = np.zeros(n_vars); row[0] = -1.0
        static_rows.append(row); static_rhs.append(-u_prev + self.delta_u_max)
        for i in range(1, N):
            row = np.zeros(n_vars); row[i] = 1.0; row[i-1] = -1.0
            static_rows.append(row); static_rhs.append(self.delta_u_max)
            row = np.zeros(n_vars); row[i] = -1.0; row[i-1] = 1.0
            static_rows.append(row); static_rhs.append(self.delta_u_max)

        # Tube: s_max >= 0, s_min <= 0, s_min <= s_max
        for i in range(N):
            row = np.zeros(n_vars); row[N+i] = -1.0
            static_rows.append(row); static_rhs.append(0.0)
            row = np.zeros(n_vars); row[2*N+i] = 1.0
            static_rows.append(row); static_rhs.append(0.0)
        for i in range(N):
            row = np.zeros(n_vars); row[2*N+i] = 1.0; row[N+i] = -1.0
            static_rows.append(row); static_rhs.append(0.0)

        # ICNN epigraph constraints (unstable neurons only)
        for step in range(N):
            n_u = step + 1
            for net_idx in range(2):
                weights = [self.weights_f1, self.weights_f2][net_idx][step]
                self._add_icnn_epigraph_vec(
                    step, net_idx, weights, z_k, n_u,
                    static_rows, static_rhs,
                )

        self._n_static_ineq = len(static_rows)
        if self._n_static_ineq > 0:
            self._A_static = np.array(static_rows)
            self._h_static = np.array(static_rhs)
        else:
            self._A_static = np.zeros((0, n_vars))
            self._h_static = np.array([])

        # Number of dynamic rows: tracking (2N) + DC bounds (2N)
        self._n_dynamic_ineq = 4 * N

    def _add_icnn_epigraph_vec(self, step, net_idx, weights, z_k, n_u,
                                rows_list, rhs_list):
        """Add epigraph constraints for unstable neurons using vectorized exprs."""
        n_hidden = self.n_hidden
        n_state = self.n_state
        n_vars = self.n_vars

        # Layer 1
        W0 = weights[0]
        b0 = weights[1]
        W0_z = W0[:, :n_state]
        W0_u = W0[:, n_state : n_state + n_u]
        c0 = W0_z @ z_k + b0

        status_l1 = self._classification[(step, net_idx, 0)]["status"]
        amb_mask_l1 = (status_l1 == 0)

        for j in np.where(amb_mask_l1)[0]:
            h_idx = self._amb_vars[(step, net_idx, 0, int(j))]

            # h >= W0_u[j,:] @ u + c0[j]  →  W0_u @ u - h <= -c0[j]
            row = np.zeros(n_vars)
            row[h_idx] = -1.0
            row[:n_u] += W0_u[j, :]
            rows_list.append(row)
            rhs_list.append(-c0[j])

            # h >= 0
            row = np.zeros(n_vars)
            row[h_idx] = -1.0
            rows_list.append(row)
            rhs_list.append(0.0)

        # Internal layers
        n_internal = (len(weights) - 4) // 4
        for layer in range(n_internal):
            Wx = np.maximum(weights[2 + layer * 4], 0)
            bx = weights[2 + layer * 4 + 1]
            W_skip = weights[2 + layer * 4 + 2]
            b_skip = weights[2 + layer * 4 + 3]

            W_skip_z = W_skip[:, :n_state]
            W_skip_u = W_skip[:, n_state : n_state + n_u]
            skip_const = W_skip_z @ z_k + bx + b_skip

            curr_layer = layer + 1
            prev_c = self._layer_const[(step, net_idx, layer)]
            prev_v = self._layer_coeffs[(step, net_idx, layer)]

            # Pre-activation for all neurons at once:
            # pre_const = Wx @ prev_c + skip_const
            # pre_coeffs = Wx @ prev_v; pre_coeffs[:, :n_u] += W_skip_u
            pre_const = Wx @ prev_c + skip_const  # (n_hidden,)
            pre_coeffs = Wx @ prev_v              # (n_hidden, n_vars)

            status_curr = self._classification[(step, net_idx, curr_layer)]["status"]
            amb_mask = (status_curr == 0)

            for j in np.where(amb_mask)[0]:
                h_idx = self._amb_vars[(step, net_idx, curr_layer, int(j))]

                # h >= pre_activation  →  pre_coeffs[j,:] @ x + skip_u[j,:] @ u - h <= -pre_const[j]
                row = np.zeros(n_vars)
                row[h_idx] = -1.0
                row += pre_coeffs[j, :]
                row[:n_u] += W_skip_u[j, :]
                rows_list.append(row)
                rhs_list.append(-pre_const[j] - np.dot(W_skip_u[j, :], np.zeros(n_u)))
                # skip_u contribution to constant is 0 (u terms are in coefficients)
                rhs_list[-1] = -pre_const[j]

                # h >= 0
                row = np.zeros(n_vars)
                row[h_idx] = -1.0
                rows_list.append(row)
                rhs_list.append(0.0)

    def build_qp(
        self,
        z_k: np.ndarray,
        u_prev: float,
        u_nominal: np.ndarray,
        y_nominal: np.ndarray,
        jacobians_f1: List[np.ndarray],
        jacobians_f2: List[np.ndarray],
        f1_nominal: np.ndarray,
        f2_nominal: np.ndarray,
        W_bounds: np.ndarray,
        linear_jacobians: List[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
               np.ndarray, np.ndarray, np.ndarray]:
        """Build the stability-reduced QP matrices (dense numpy arrays).

        Returns P (upper tri), q, A_eq, b_eq, A_ineq, h_lo, h_hi.
        """
        N = self.N
        n_vars = self.n_vars

        # === Dynamic inequality rows: tracking (2N) + DC bounds (2N) ===
        n_dyn = self._n_dynamic_ineq
        A_dyn = np.zeros((n_dyn, n_vars))
        h_dyn = np.zeros(n_dyn)
        row = 0

        # Tracking: t >= y_nom + s_max - beta_0  →  s_max - t <= beta_0 - y_nom
        for i in range(N):
            A_dyn[row, N + i] = 1.0      # s_max
            A_dyn[row, 3*N + i] = -1.0   # -t
            h_dyn[row] = self.beta_0 - y_nominal[i]
            row += 1

        # Tracking: t >= 0  →  -t <= 0
        for i in range(N):
            A_dyn[row, 3*N + i] = -1.0
            h_dyn[row] = 0.0
            row += 1

        # DC upper bound: y_f1 - s_max - J_f2_eff @ u <= f1_nom - J_f2_eff @ u_nom - w_max
        for step in range(N):
            n_u = step + 1
            J_f2 = jacobians_f2[step].flatten()
            J_f2_eff = J_f2 - linear_jacobians[step].flatten() if linear_jacobians else J_f2

            w_max = W_bounds[step, 1]
            rhs = f1_nominal[step] - np.dot(J_f2_eff, u_nominal[:n_u]) - w_max

            A_dyn[row, N + step] = -1.0  # -s_max

            # y_f1 term
            f1_info = self._y_f_info[(step, 0)]
            if f1_info["is_var"]:
                A_dyn[row, f1_info["var_idx"]] = 1.0
            else:
                out_c = self._output_const[(step, 0)]
                out_v = self._output_coeffs[(step, 0)]
                rhs -= out_c
                A_dyn[row, :] += out_v

            # -J_f2_eff @ u
            A_dyn[row, :n_u] -= J_f2_eff

            h_dyn[row] = rhs
            row += 1

        # DC lower bound: s_min + y_f2 - J_f1_eff @ u <= f2_nom - J_f1_eff @ u_nom + w_min
        for step in range(N):
            n_u = step + 1
            J_f1 = jacobians_f1[step].flatten()
            J_f1_eff = J_f1 + linear_jacobians[step].flatten() if linear_jacobians else J_f1

            w_min = W_bounds[step, 0]
            rhs = f2_nominal[step] - np.dot(J_f1_eff, u_nominal[:n_u]) + w_min

            A_dyn[row, 2*N + step] = 1.0  # s_min

            # y_f2 term
            f2_info = self._y_f_info[(step, 1)]
            if f2_info["is_var"]:
                A_dyn[row, f2_info["var_idx"]] = 1.0
            else:
                out_c = self._output_const[(step, 1)]
                out_v = self._output_coeffs[(step, 1)]
                rhs -= out_c
                A_dyn[row, :] += out_v

            # -J_f1_eff @ u
            A_dyn[row, :n_u] -= J_f1_eff

            h_dyn[row] = rhs
            row += 1

        # === Combine static + dynamic inequalities ===
        n_total_ineq = self._n_static_ineq + n_dyn
        A_ineq = np.empty((n_total_ineq, n_vars))
        A_ineq[:self._n_static_ineq, :] = self._A_static
        A_ineq[self._n_static_ineq:, :] = A_dyn

        h_hi = np.empty(n_total_ineq)
        h_hi[:self._n_static_ineq] = self._h_static
        h_hi[self._n_static_ineq:] = h_dyn
        h_lo = np.full(n_total_ineq, -1e30)

        return self._P, self._q_static.copy(), self._A_eq, self._b_eq, A_ineq, h_lo, h_hi


# =============================================================================
# Stability-reduced PIQP Solver (Dense)
# =============================================================================


class StableNeuronPIQPSolver:
    """stability-reduced QP solver using PIQP dense with neuron elimination."""

    def __init__(self, builder: StableNeuronQPBuilder):
        self._builder = builder
        self._solver = None
        self._first_solve = True
        self._prev_z_k = None
        self._prev_u_prev = None

        self.N = builder.N
        N = builder.N
        self.idx_u = slice(0, N)
        self.idx_s_max = slice(N, 2 * N)
        self.idx_s_min = slice(2 * N, 3 * N)

    def solve(
        self,
        z_k: np.ndarray,
        u_prev: float,
        u_nominal: np.ndarray,
        y_nominal: np.ndarray,
        jacobians_f1: List[np.ndarray],
        jacobians_f2: List[np.ndarray],
        f1_nominal: np.ndarray,
        f2_nominal: np.ndarray,
        W_bounds: np.ndarray,
        linear_jacobians: List[np.ndarray] = None,
    ) -> DirectQPSolution:
        """Solve the stability-reduced QP."""
        import piqp

        start = time.perf_counter()

        # Reclassify if z_k or u_prev changed (new MPC timestep)
        need_reclassify = (
            self._prev_z_k is None
            or not np.array_equal(z_k, self._prev_z_k)
            or u_prev != self._prev_u_prev
        )

        if need_reclassify:
            self._builder.classify_and_setup(z_k, u_prev)
            self._prev_z_k = z_k.copy()
            self._prev_u_prev = u_prev
            self._first_solve = True  # dimensions may change

        classify_time = time.perf_counter() - start

        # Build QP (only dynamic rows recomputed)
        build_start = time.perf_counter()
        P, q, A_eq, b_eq, A_ineq, h_lo, h_hi = self._builder.build_qp(
            z_k, u_prev, u_nominal, y_nominal,
            jacobians_f1, jacobians_f2, f1_nominal, f2_nominal,
            W_bounds, linear_jacobians,
        )
        build_time = time.perf_counter() - build_start

        # Solve with PIQP dense
        solve_start = time.perf_counter()

        if self._first_solve:
            self._solver = piqp.DenseSolver()
            self._solver.settings.verbose = False
            self._solver.settings.eps_abs = 1e-6
            self._solver.settings.eps_rel = 1e-6
            self._solver.settings.max_iter = 500

            self._solver.setup(
                P, q,
                A_eq, b_eq,
                A_ineq, h_lo, h_hi,
            )
            self._first_solve = False
        else:
            self._solver.update(
                P=P, c=q,
                A=A_eq, b=b_eq,
                G=A_ineq, h_l=h_lo, h_u=h_hi,
            )

        status = self._solver.solve()
        solve_time = time.perf_counter() - solve_start
        total_time = time.perf_counter() - start

        is_feasible = (status == piqp.PIQP_SOLVED)

        if is_feasible:
            x = np.array(self._solver.result.x)
            return DirectQPSolution(
                u_optimal=np.asarray(x[self.idx_u], dtype=np.float32),
                s_max_optimal=np.asarray(x[self.idx_s_max], dtype=np.float32),
                s_min_optimal=np.asarray(x[self.idx_s_min], dtype=np.float32),
                cost=self._solver.result.info.primal_obj,
                status=f"solved (n_vars={self._builder.n_vars}, "
                       f"n_amb={self._builder._n_amb})",
                solve_time=total_time,
                is_feasible=True,
                setup_time=classify_time + build_time,
                update_time=build_time,
            )
        else:
            return DirectQPSolution(
                u_optimal=np.asarray(u_nominal, dtype=np.float32),
                s_max_optimal=np.zeros(self.N, dtype=np.float32),
                s_min_optimal=np.zeros(self.N, dtype=np.float32),
                cost=np.inf,
                status=f"infeasible ({status}, n_vars={self._builder.n_vars})",
                solve_time=total_time,
                is_feasible=False,
                update_time=build_time,
            )


# =============================================================================
# Stability-reduced ProxQP Solver (Dense)
# =============================================================================


class StableNeuronProxQPSolver:
    """stability-reduced QP solver using ProxQP dense with neuron elimination.

    WARNING: ProxQP's ADMM algorithm diverges on the stability-reduced QP structure
    (PSD Hessian with zero-cost epigraph variables) regardless of regularization.
    Use StableNeuronPIQPSolver or StableNeuronDAQPSolver instead.
    """

    def __init__(self, builder: StableNeuronQPBuilder):
        self._builder = builder
        self._solver = None
        self._first_solve = True
        self._prev_z_k = None
        self._prev_u_prev = None

        self.N = builder.N
        N = builder.N
        self.idx_u = slice(0, N)
        self.idx_s_max = slice(N, 2 * N)
        self.idx_s_min = slice(2 * N, 3 * N)

    def solve(
        self,
        z_k: np.ndarray,
        u_prev: float,
        u_nominal: np.ndarray,
        y_nominal: np.ndarray,
        jacobians_f1: List[np.ndarray],
        jacobians_f2: List[np.ndarray],
        f1_nominal: np.ndarray,
        f2_nominal: np.ndarray,
        W_bounds: np.ndarray,
        linear_jacobians: List[np.ndarray] = None,
    ) -> DirectQPSolution:
        """Solve the stability-reduced QP using ProxQP dense."""
        import proxsuite

        start = time.perf_counter()

        need_reclassify = (
            self._prev_z_k is None
            or not np.array_equal(z_k, self._prev_z_k)
            or u_prev != self._prev_u_prev
        )

        if need_reclassify:
            self._builder.classify_and_setup(z_k, u_prev)
            self._prev_z_k = z_k.copy()
            self._prev_u_prev = u_prev
            self._first_solve = True

        classify_time = time.perf_counter() - start

        build_start = time.perf_counter()
        P, q, A_eq, b_eq, A_ineq, h_lo, h_hi = self._builder.build_qp(
            z_k, u_prev, u_nominal, y_nominal,
            jacobians_f1, jacobians_f2, f1_nominal, f2_nominal,
            W_bounds, linear_jacobians,
        )
        build_time = time.perf_counter() - build_start

        solve_start = time.perf_counter()

        # ProxQP struggles with PSD (not PD) Hessians — regularize zero diags
        zero_diag = np.diag(P) < 1e-12
        if np.any(zero_diag):
            P = P.copy()
            P[np.diag_indices_from(P)] += zero_diag * 1e-8

        n_vars = P.shape[0]
        n_eq = A_eq.shape[0] if A_eq.shape[0] > 0 else 0
        n_ineq = A_ineq.shape[0]

        if self._first_solve:
            self._solver = proxsuite.proxqp.dense.QP(n_vars, n_eq, n_ineq)
            self._solver.settings.eps_abs = 1e-6
            self._solver.settings.eps_rel = 0.0
            self._solver.settings.max_iter = 500
            self._solver.settings.verbose = False
            self._solver.init(P, q, A_eq, b_eq, A_ineq, h_lo, h_hi)
            self._first_solve = False
        else:
            self._solver.settings.initial_guess = (
                proxsuite.proxqp.InitialGuess.WARM_START_WITH_PREVIOUS_RESULT
            )
            self._solver.update(H=P, g=q, A=A_eq, b=b_eq, C=A_ineq, l=h_lo, u=h_hi)

        self._solver.solve()
        solve_time = time.perf_counter() - solve_start
        total_time = time.perf_counter() - start

        is_feasible = self._solver.results.info.status == proxsuite.proxqp.PROXQP_SOLVED

        if is_feasible:
            x = np.array(self._solver.results.x)
            return DirectQPSolution(
                u_optimal=np.asarray(x[self.idx_u], dtype=np.float32),
                s_max_optimal=np.asarray(x[self.idx_s_max], dtype=np.float32),
                s_min_optimal=np.asarray(x[self.idx_s_min], dtype=np.float32),
                cost=self._solver.results.info.objValue,
                status=f"solved (n_vars={self._builder.n_vars}, "
                       f"n_amb={self._builder._n_amb})",
                solve_time=total_time,
                is_feasible=True,
                setup_time=classify_time + build_time,
                update_time=build_time,
            )
        else:
            return DirectQPSolution(
                u_optimal=np.asarray(u_nominal, dtype=np.float32),
                s_max_optimal=np.zeros(self.N, dtype=np.float32),
                s_min_optimal=np.zeros(self.N, dtype=np.float32),
                cost=np.inf,
                status=f"infeasible (n_vars={self._builder.n_vars})",
                solve_time=total_time,
                is_feasible=False,
                update_time=build_time,
            )


# =============================================================================
# Stability-reduced DAQP Solver (Dense active-set)
# =============================================================================


class StableNeuronDAQPSolver:
    """stability-reduced QP solver using DAQP dense active-set with neuron elimination."""

    def __init__(self, builder: StableNeuronQPBuilder):
        self._builder = builder
        self._prev_z_k = None
        self._prev_u_prev = None

        self.N = builder.N
        N = builder.N
        self.idx_u = slice(0, N)
        self.idx_s_max = slice(N, 2 * N)
        self.idx_s_min = slice(2 * N, 3 * N)

    def solve(
        self,
        z_k: np.ndarray,
        u_prev: float,
        u_nominal: np.ndarray,
        y_nominal: np.ndarray,
        jacobians_f1: List[np.ndarray],
        jacobians_f2: List[np.ndarray],
        f1_nominal: np.ndarray,
        f2_nominal: np.ndarray,
        W_bounds: np.ndarray,
        linear_jacobians: List[np.ndarray] = None,
    ) -> DirectQPSolution:
        """Solve the stability-reduced QP using DAQP."""
        import daqp

        start = time.perf_counter()

        need_reclassify = (
            self._prev_z_k is None
            or not np.array_equal(z_k, self._prev_z_k)
            or u_prev != self._prev_u_prev
        )

        if need_reclassify:
            self._builder.classify_and_setup(z_k, u_prev)
            self._prev_z_k = z_k.copy()
            self._prev_u_prev = u_prev

        classify_time = time.perf_counter() - start

        build_start = time.perf_counter()
        P, q, A_eq, b_eq, A_ineq, h_lo, h_hi = self._builder.build_qp(
            z_k, u_prev, u_nominal, y_nominal,
            jacobians_f1, jacobians_f2, f1_nominal, f2_nominal,
            W_bounds, linear_jacobians,
        )
        build_time = time.perf_counter() - build_start

        solve_start = time.perf_counter()

        # DAQP uses a combined constraint matrix:
        #   A_combined x <= b_upper  (with b_lower for two-sided)
        # sense flags: 0 = inequality (<=), 5 = equality
        n_eq = A_eq.shape[0]
        n_ineq = A_ineq.shape[0]

        # Stack: equalities first, then inequalities
        A_combined = np.vstack([A_eq, A_ineq]) if n_eq > 0 else A_ineq.copy()
        bupper = np.concatenate([b_eq, h_hi]) if n_eq > 0 else h_hi.copy()
        blower = np.concatenate([b_eq, h_lo]) if n_eq > 0 else h_lo.copy()

        sense = np.zeros(n_eq + n_ineq, dtype=np.int32)
        sense[:n_eq] = 5  # equality constraints

        # eps_prox regularization handles the PSD (not PD) Hessian from
        # zero-cost epigraph variables without needing explicit P modification.
        x_sol, fval, exitflag, info = daqp.solve(
            P, q, A_combined, bupper, blower, sense, eps_prox=1e-6,
        )
        solve_time = time.perf_counter() - solve_start
        total_time = time.perf_counter() - start

        # exitflag: 1 = optimal, 2 = cycling detected, -1 = infeasible
        is_feasible = (exitflag == 1)

        if is_feasible:
            return DirectQPSolution(
                u_optimal=np.asarray(x_sol[self.idx_u], dtype=np.float32),
                s_max_optimal=np.asarray(x_sol[self.idx_s_max], dtype=np.float32),
                s_min_optimal=np.asarray(x_sol[self.idx_s_min], dtype=np.float32),
                cost=fval,
                status=f"solved (n_vars={self._builder.n_vars}, "
                       f"n_amb={self._builder._n_amb})",
                solve_time=total_time,
                is_feasible=True,
                setup_time=classify_time + build_time,
                update_time=build_time,
            )
        else:
            return DirectQPSolution(
                u_optimal=np.asarray(u_nominal, dtype=np.float32),
                s_max_optimal=np.zeros(self.N, dtype=np.float32),
                s_min_optimal=np.zeros(self.N, dtype=np.float32),
                cost=np.inf,
                status=f"infeasible (exitflag={exitflag}, "
                       f"n_vars={self._builder.n_vars})",
                solve_time=total_time,
                is_feasible=False,
                update_time=build_time,
            )


# =============================================================================
# Factory function
# =============================================================================


def create_stable_neuron_solver(predictor, config, backend: str = "piqp"):
    """Factory function to create a stability-reduced QP solver from predictor and config.

    Args:
        predictor: MultiStepDCNN model.
        config: SCPConfig with QP parameters.
        backend: Dense QP solver backend - "piqp", "proxqp", or "daqp".

    Returns:
        StableNeuronPIQPSolver, StableNeuronProxQPSolver, or StableNeuronDAQPSolver.
    """
    try:
        from dcnn_tube_mpc.analysis.jacobian import extract_weights_from_convex_nn
    except ImportError as exc:
        raise ImportError(
            "create_stable_neuron_solver requires dcnn-tube-mpc-dbs to be installed. "
            "Install it with: pip install -e path/to/dcnn-tube-mpc-dbs  "
            "Or use StableNeuronQPBuilder directly with pre-extracted weights."
        ) from exc

    N_ctrl = getattr(config, "control_horizon", config.prediction_horizon)
    N_pred = config.prediction_horizon
    if N_ctrl < N_pred:
        import warnings
        warnings.warn(
            f"Extended horizon (N_ctrl={N_ctrl} < N_pred={N_pred}) not supported "
            "by stability-reduced solver. Falling back to CVXPY solver."
        )
        return None

    N = config.prediction_horizon
    n_state = predictor.n_state

    weights_f1 = [
        extract_weights_from_convex_nn(predictor.networks[i].f1)
        for i in range(N)
    ]
    weights_f2 = [
        extract_weights_from_convex_nn(predictor.networks[i].f2)
        for i in range(N)
    ]

    builder = StableNeuronQPBuilder(
        N=N, n_state=n_state,
        weights_f1=weights_f1, weights_f2=weights_f2,
        Q=config.Q, R=config.R,
        R_delta=getattr(config, "R_delta", 0.0),
        tube_weight=getattr(config, "tube_weight", 0.0),
        beta_0=config.beta_0,
        u_min=config.u_min, u_max=config.u_max,
        delta_u_max=config.delta_u_max,
        pe_gamma=getattr(config, "pe_gamma", 0.0),
    )

    solver_map = {
        "piqp": StableNeuronPIQPSolver,
        "proxqp": StableNeuronProxQPSolver,
        "daqp": StableNeuronDAQPSolver,
    }
    solver_cls = solver_map.get(backend, StableNeuronPIQPSolver)
    return solver_cls(builder)
