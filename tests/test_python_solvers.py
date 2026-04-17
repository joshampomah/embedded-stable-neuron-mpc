"""Tests for Python solver modules in embedded-condensed-mpc."""
from __future__ import annotations

import importlib
import pytest
import numpy as np


# =============================================================================
# Import tests
# =============================================================================


def test_direct_qp_solver_import():
    """stable_neuron_qp_solver imports without error."""
    from embedded_mpc.solvers import direct_qp_solver  # noqa: F401
    assert hasattr(direct_qp_solver, "DirectQPSolution")
    assert hasattr(direct_qp_solver, "QPMatrixBuilder")
    assert hasattr(direct_qp_solver, "DirectQPSolver")


def test_stable_neuron_qp_solver_import():
    """stable_neuron_qp_solver imports without error."""
    from embedded_mpc.solvers import stable_neuron_qp_solver  # noqa: F401
    assert hasattr(stable_neuron_qp_solver, "StableNeuronQPBuilder")
    assert hasattr(stable_neuron_qp_solver, "classify_neurons")
    assert hasattr(stable_neuron_qp_solver, "StableNeuronPIQPSolver")


def test_piqp_solver_import():
    """piqp_solver imports without error (skip if piqp not installed)."""
    pytest.importorskip("piqp", reason="piqp not installed")
    from embedded_mpc.solvers import piqp_solver  # noqa: F401
    assert hasattr(piqp_solver, "PIQPSolver")
    assert hasattr(piqp_solver, "create_piqp_solver")


# =============================================================================
# Device config tests
# =============================================================================


def test_device_config_loads():
    """DeviceConfig loads from the bundled JSON."""
    from embedded_mpc.config.device_config import get_device_config, DeviceConfig
    config = get_device_config(reload=True)
    assert isinstance(config, DeviceConfig)
    assert config.constraints.u_max == pytest.approx(0.03)
    assert config.constraints.u_min == pytest.approx(0.0)
    assert config.sample_time == pytest.approx(0.02)
    assert config.mpc.prediction_horizon == 5
    assert len(config.disturbance_bounds) == 5


def test_device_config_get_patient_defaults():
    """get_patient returns defaults for an unknown patient."""
    from embedded_mpc.config.device_config import get_device_config
    config = get_device_config()
    patient = config.get_patient("SyntheticPatient")
    assert patient.id == "SyntheticPatient"
    assert patient.u_max == pytest.approx(config.constraints.u_max)


# =============================================================================
# QPMatrixBuilder instantiation test (no solver needed)
# =============================================================================


def _make_dummy_weights(n_state: int = 5, n_hidden: int = 4, n_u: int = 1):
    """Build minimal ICNN weight list for one network at one step."""
    rng = np.random.default_rng(42)
    # Format: [W0, b0, W_out, b_out] for a 1-layer ICNN
    W0 = rng.normal(0, 0.1, (n_hidden, n_state + n_u))
    b0 = rng.normal(0, 0.01, (n_hidden,))
    W_out = np.abs(rng.normal(0, 0.1, (1, n_hidden)))  # must be non-negative
    b_out = rng.normal(0, 0.01, (1,))
    return [W0, b0, W_out, b_out]


def test_qp_matrix_builder_setup():
    """QPMatrixBuilder can be instantiated and set up with minimal weights."""
    from embedded_mpc.solvers.direct_qp_solver import QPMatrixBuilder

    N = 2
    n_state = 5
    n_hidden = 4

    weights_f1 = [_make_dummy_weights(n_state, n_hidden, i + 1) for i in range(N)]
    weights_f2 = [_make_dummy_weights(n_state, n_hidden, i + 1) for i in range(N)]

    builder = QPMatrixBuilder(
        N=N,
        n_state=n_state,
        weights_f1=weights_f1,
        weights_f2=weights_f2,
        Q=50000.0,
        R=1.0,
        beta_0=2.3,
        u_min=0.0,
        u_max=0.03,
        delta_u_max=0.0024,
    )
    builder.setup()

    assert builder.n_vars > 0
    assert builder.P is not None
    assert builder.A_ineq is not None


def test_stable_neuron_qp_builder_classify():
    """StableNeuronQPBuilder classifies neurons without error."""
    from embedded_mpc.solvers.stable_neuron_qp_solver import StableNeuronQPBuilder

    N = 2
    n_state = 5
    n_hidden = 4

    weights_f1 = [_make_dummy_weights(n_state, n_hidden, i + 1) for i in range(N)]
    weights_f2 = [_make_dummy_weights(n_state, n_hidden, i + 1) for i in range(N)]

    builder = StableNeuronQPBuilder(
        N=N,
        n_state=n_state,
        weights_f1=weights_f1,
        weights_f2=weights_f2,
        Q=50000.0,
        R=1.0,
        beta_0=2.3,
        u_min=0.0,
        u_max=0.03,
        delta_u_max=0.0024,
    )

    rng = np.random.default_rng(0)
    z_k = rng.normal(2.3, 0.1, n_state)
    builder.classify_and_setup(z_k, u_prev=0.01)

    assert builder.n_vars >= 4 * N  # at minimum u, s_max, s_min, t
    assert builder._n_amb >= 0
