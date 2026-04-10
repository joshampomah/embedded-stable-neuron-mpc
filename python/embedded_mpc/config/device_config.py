# Canonical owner: embedded-stable-neuron-mpc
"""Centralized device configuration loader.

Provides a single source of truth for all device parameters.
No external dependencies - uses only stdlib json.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class StimulationParams:
    """Stimulation response model parameters."""
    gain: float  # k = 62.11 (CDC24 paper, units mismatch resolved)
    tau1: float  # Fast time constant (s)
    tau2: float  # Slow time constant (s)


@dataclass(frozen=True)
class ConstraintParams:
    """Physical device constraints."""
    u_max: float  # Maximum stimulation amplitude
    u_min: float  # Minimum stimulation amplitude
    delta_u_max: float  # Maximum rate of change per time step


@dataclass(frozen=True)
class BetaParams:
    """Beta oscillation parameters."""
    default_threshold: float  # Default pathological threshold
    ar_coefficients: Tuple[float, ...]  # AR model coefficients
    noise_std: float  # Noise standard deviation
    bias: float  # Bias term


@dataclass(frozen=True)
class MPCParams:
    """MPC controller default parameters."""
    prediction_horizon: int  # N
    Q: float  # Tracking weight
    R: float  # Control magnitude weight
    R_delta: float  # Control rate weight
    maxiters: int  # Maximum SCP iterations
    delta_J_min: float  # Absolute convergence threshold |ΔJ|


@dataclass(frozen=True)
class KalmanParams:
    """Kalman filter parameters."""
    gamma_base: float  # Base process noise
    gamma_beta_factor: float  # Beta state noise scaling
    R_meas: float  # Measurement noise


@dataclass(frozen=True)
class PatientParams:
    """Patient-specific parameters."""
    id: str  # Patient identifier
    beta_threshold: float  # Patient-specific threshold
    u_max: float  # Patient-specific max stimulation


@dataclass(frozen=True)
class DeviceConfig:
    """Complete device configuration - single source of truth."""

    stimulation: StimulationParams
    sample_time: float
    constraints: ConstraintParams
    beta: BetaParams
    mpc: MPCParams
    kalman: KalmanParams
    disturbance_bounds: List[Tuple[float, float]]
    patients: Dict[str, PatientParams]

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "DeviceConfig":
        """Load config from JSON file.

        Args:
            path: Path to device_params.json. If None, uses default location.

        Returns:
            DeviceConfig instance.
        """
        if path is None:
            path = Path(__file__).parent / "device_params.json"

        with open(path) as f:
            data = json.load(f)

        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict) -> "DeviceConfig":
        """Construct DeviceConfig from parsed JSON dict."""
        # Filter out comment fields (starting with _)
        def filter_comments(d: dict) -> dict:
            return {k: v for k, v in d.items() if not k.startswith("_")}

        return cls(
            stimulation=StimulationParams(**filter_comments(data["stimulation"])),
            sample_time=data["sampling"]["time"],
            constraints=ConstraintParams(**filter_comments(data["constraints"])),
            beta=BetaParams(
                default_threshold=data["beta"]["default_threshold"],
                ar_coefficients=tuple(data["beta"]["ar_coefficients"]),
                noise_std=data["beta"]["noise_std"],
                bias=data["beta"]["bias"],
            ),
            mpc=MPCParams(**data["mpc"]),
            kalman=KalmanParams(**data["kalman"]),
            disturbance_bounds=[tuple(b) for b in data["disturbance_bounds"]],
            patients={
                name: PatientParams(
                    id=params.get("id", name),
                    **{k: v for k, v in params.items() if k != "id"}
                )
                for name, params in data.get("patients", {}).items()
            },
        )

    def get_patient(self, name: str) -> PatientParams:
        """Get patient-specific parameters, or defaults if not found.

        Args:
            name: Patient identifier (e.g., "Patient4").

        Returns:
            PatientParams with patient-specific or default values.
        """
        if name in self.patients:
            return self.patients[name]

        # Return defaults
        return PatientParams(
            id=name,
            beta_threshold=self.beta.default_threshold,
            u_max=self.constraints.u_max,
        )


# Singleton instance (cached)
_config: Optional[DeviceConfig] = None


def get_device_config(reload: bool = False) -> DeviceConfig:
    """Get the device configuration (cached singleton).

    Args:
        reload: If True, force reload from JSON file.

    Returns:
        DeviceConfig instance.

    Example:
        >>> from embedded_mpc.config.device_config import get_device_config
        >>> config = get_device_config()
        >>> print(config.constraints.u_max)
        0.03
    """
    global _config
    if _config is None or reload:
        _config = DeviceConfig.load()
    return _config
