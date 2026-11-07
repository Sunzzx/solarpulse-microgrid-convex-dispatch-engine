"""
Data models and physics parameters for microgrid nodes, battery storage, and energy resources.
"""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class BatteryStorage:
    """
    Electrochemical Energy Storage System (BESS) representation.

    Parameters:
        capacity_kwh: Total energy capacity (kWh)
        p_max_charge_kw: Maximum charging rate limit (kW)
        p_max_discharge_kw: Maximum discharging rate limit (kW)
        efficiency_charge: One-way charging Coulombic/thermal efficiency eta_c in (0, 1]
        efficiency_discharge: One-way discharging Coulombic/thermal efficiency eta_d in (0, 1]
        soc_min: Minimum allowable state of charge fraction in [0, 1]
        soc_max: Maximum allowable state of charge fraction in [0, 1]
        initial_soc: Initial state of charge in [soc_min, soc_max]
        target_soc: Terminal state of charge constraint fraction at horizon end
        degradation_cost_per_kwh: Cost coefficient for cycle life wear ($/kWh throughput)
    """
    capacity_kwh: float = 100.0
    p_max_charge_kw: float = 50.0
    p_max_discharge_kw: float = 50.0
    efficiency_charge: float = 0.95
    efficiency_discharge: float = 0.95
    soc_min: float = 0.10
    soc_max: float = 0.95
    initial_soc: float = 0.50
    target_soc: Optional[float] = 0.50
    degradation_cost_per_kwh: float = 0.02

    def __post_init__(self):
        if not (0.0 <= self.soc_min < self.soc_max <= 1.0):
            raise ValueError("Invalid SoC limits: ensure 0 <= soc_min < soc_max <= 1.0")
        if not (self.soc_min <= self.initial_soc <= self.soc_max):
            raise ValueError("initial_soc must fall within [soc_min, soc_max]")
        if not (0.0 < self.efficiency_charge <= 1.0 and 0.0 < self.efficiency_discharge <= 1.0):
            raise ValueError("Efficiencies must be strictly within (0, 1]")


@dataclass
class RenewableSource:
    """Photovoltaic or wind generation resource."""
    name: str
    rated_kw: float
    generation_profile_kw: np.ndarray
    curtailment_penalty_per_kwh: float = 0.01


@dataclass
class LoadProfile:
    """Electrical demand load profile with critical and deferrable splits."""
    critical_load_kw: np.ndarray
    curtailable_load_kw: Optional[np.ndarray] = None
    unserved_energy_penalty: float = 10.0  # High penalty ($/kWh) for shed load


@dataclass
class GridIntertie:
    """Point of Common Coupling (PCC) utility intertie connection."""
    p_max_import_kw: float = 200.0
    p_max_export_kw: float = 150.0
    tariff_import_per_kwh: np.ndarray = field(default_factory=lambda: np.array([]))
    tariff_export_per_kwh: np.ndarray = field(default_factory=lambda: np.array([]))
    co2_intensity_kg_per_kwh: Optional[np.ndarray] = None


@dataclass
class MicrogridNode:
    """
    Aggregated regional or single-bus microgrid node containing local generation, storage, load, and grid link.
    """
    node_id: str
    battery: BatteryStorage
    solar: RenewableSource
    load: LoadProfile
    grid: GridIntertie
