# Canonical owner: embedded-stable-neuron-mpc
"""PIQP solver backend for DC-MPC.

Uses the QPMatrixBuilder from direct_qp_solver.py and calls PIQP (Proximal
Interior Point QP solver) directly. PIQP is designed for ill-conditioned QPs
and handles the extreme condition number (~10^11) of this problem well.

PIQP format:
    min  0.5 x'Px + c'x
    s.t. Ax = b           (equalities)
         Gx <= h           (inequalities)

This maps directly from QPMatrixBuilder output with no conversion needed.
"""
from __future__ import annotations

import time
from typing import List

import numpy as np
from scipy import sparse

from embedded_mpc.solvers.direct_qp_solver import (
    DirectQPSolution,
    QPMatrixBuilder,
    _create_builder,
)


class PIQPSolver:
    """PIQP solver for DC-MPC with warm-starting support.

    Creates a persistent PIQP solver that is set up once and updated between
    SCP iterations using solver.update() + solver.solve() for automatic
    warm-starting.
    """

    def __init__(self, builder: QPMatrixBuilder):
        self._builder = builder
        self._setup_done = False
        self._solver = None
        self._first_solve = True

        # Expose variable indices from builder
        self.N = builder.N
        self.idx_u = builder.idx_u
        self.idx_s_max = builder.idx_s_max
        self.idx_s_min = builder.idx_s_min

    def setup(self):
        """Build the QP structure and create persistent PIQP solver."""
        import piqp

        start = time.perf_counter()

        if not self._builder._setup_done:
            self._builder.setup()

        b = self._builder

        # PIQP needs upper-triangular P in CSC format
        self._P_upper = sparse.triu(b.P).tocsc()

        # Create solver with settings
        self._solver = piqp.SparseSolver()
        self._solver.settings.verbose = False
        self._solver.settings.eps_abs = 1e-6
        self._solver.settings.eps_rel = 1e-6
        self._solver.settings.max_iter = 500

        self._setup_done = True
        self._first_solve = True
        self._setup_time = time.perf_counter() - start

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
        """Solve the QP with updated parameters."""
        import piqp

        if not self._setup_done:
            self.setup()

        start = time.perf_counter()

        b = self._builder

        # Update linear cost
        q = b.compute_linear_cost(y_nominal, u_prev)

        # Update constraints
        b_eq, h, A_ineq = b.update_constraints(
            z_k, u_prev, u_nominal, y_nominal,
            jacobians_f1, jacobians_f2, f1_nominal, f2_nominal, W_bounds,
            linear_jacobians=linear_jacobians,
        )

        # Prepare equality constraints
        if b.A_eq is not None:
            A_eq_csc = b.A_eq.tocsc()
        else:
            A_eq_csc = sparse.csc_matrix((0, b.n_vars))
            b_eq = np.array([])

        # Inequality constraints: h_l <= Gx <= h_u
        # For Gx <= h, use h_l = -inf, h_u = h
        A_ineq_csc = A_ineq.tocsc()
        h_l = np.full(b.n_ineq, -1e30)

        update_time = time.perf_counter() - start

        # Solve
        solve_start = time.perf_counter()

        if self._first_solve:
            self._solver.setup(
                self._P_upper, q,
                A_eq_csc, b_eq,
                A_ineq_csc, h_l, h,
            )
            self._first_solve = False
        else:
            self._solver.update(
                P=self._P_upper, c=q,
                A=A_eq_csc, b=b_eq,
                G=A_ineq_csc, h_l=h_l, h_u=h,
            )

        status = self._solver.solve()
        solve_time_inner = time.perf_counter() - solve_start

        total_time = time.perf_counter() - start

        is_feasible = (status == piqp.PIQP_SOLVED)

        if is_feasible:
            x = np.array(self._solver.result.x)

            return DirectQPSolution(
                u_optimal=np.asarray(x[self.idx_u], dtype=np.float32),
                s_max_optimal=np.asarray(x[self.idx_s_max], dtype=np.float32),
                s_min_optimal=np.asarray(x[self.idx_s_min], dtype=np.float32),
                cost=self._solver.result.info.primal_obj,
                status=str(status),
                solve_time=total_time,
                is_feasible=True,
                setup_time=getattr(self, '_setup_time', 0),
                update_time=update_time,
            )
        else:
            return DirectQPSolution(
                u_optimal=np.asarray(u_nominal, dtype=np.float32),
                s_max_optimal=np.zeros(self.N, dtype=np.float32),
                s_min_optimal=np.zeros(self.N, dtype=np.float32),
                cost=np.inf,
                status=str(status),
                solve_time=total_time,
                is_feasible=False,
                update_time=update_time,
            )


def create_piqp_solver(predictor, config):
    """Factory function to create PIQPSolver from predictor and config.

    Returns:
        PIQPSolver instance, or None if extended horizon is detected.
    """
    builder = _create_builder(predictor, config)
    if builder is None:
        return None
    return PIQPSolver(builder)
