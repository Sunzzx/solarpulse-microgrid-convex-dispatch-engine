"""Unit, edge-case, and numerical invariance tests for SolarPulse."""
from __future__ import annotations

import numpy as np
import pytest

from solarpulse.models import (
    BatteryStorage,
    RenewableSource,
    LoadProfile,
    GridIntertie,
    MicrogridNode,
)
from solarpulse.solver import ConvexDispatchEngine, DispatchPlan
from solarpulse.dynamics import MicrogridSimulation


def _make_flat_node(
    T: int = 8,
    load: float = 40.0,
    pv: float = 30.0,
    s0: float = 0.5,
) -> MicrogridNode:
    batt = BatteryStorage(
        capacity_kwh=100.0,
        p_max_charge_kw=40.0,
        p_max_discharge_kw=40.0,
        efficiency_charge=0.95,
        efficiency_discharge=0.95,
        soc_min=0.1,
        soc_max=0.9,
        initial_soc=s0,
        target_soc=s0,
        degradation_cost_per_kwh=0.02,
    )
    solar = RenewableSource(
        name="pv",
        rated_kw=100.0,
        generation_profile_kw=np.full(T, pv),
        curtailment_penalty_per_kwh=0.01,
    )
    load_p = LoadProfile(
        critical_load_kw=np.full(T, load),
        unserved_energy_penalty=20.0,
    )
    grid = GridIntertie(
        p_max_import_kw=100.0,
        p_max_export_kw=80.0,
        tariff_import_per_kwh=np.full(T, 0.20),
        tariff_export_per_kwh=np.full(T, 0.05),
        co2_intensity_kg_per_kwh=np.full(T, 0.4),
    )
    return MicrogridNode(
        node_id="test-node",
        battery=batt,
        solar=solar,
        load=load_p,
        grid=grid,
    )


class TestBatteryStorage:
    def test_valid_defaults(self):
        b = BatteryStorage()
        assert 0.0 <= b.soc_min < b.soc_max <= 1.0
        assert b.soc_min <= b.initial_soc <= b.soc_max

    def test_invalid_soc_limits(self):
        with pytest.raises(ValueError):
            BatteryStorage(soc_min=0.9, soc_max=0.1)

    def test_initial_soc_out_of_range(self):
        with pytest.raises(ValueError):
            BatteryStorage(soc_min=0.2, soc_max=0.8, initial_soc=0.05)

    def test_invalid_efficiency(self):
        with pytest.raises(ValueError):
            BatteryStorage(efficiency_charge=0.0)
        with pytest.raises(ValueError):
            BatteryStorage(efficiency_discharge=1.5)


class TestConvexDispatchEngine:
    def test_solve_succeeds_flat_profiles(self):
        node = _make_flat_node(T=6, load=40.0, pv=25.0)
        plan = ConvexDispatchEngine(node, dt_hours=1.0).solve()
        assert plan.success is True
        assert plan.time_steps == 6
        assert plan.total_cost is not None

    def test_power_balance_residual(self):
        """Power balance must hold at every timestep (within numerical tol)."""
        node = _make_flat_node(T=12, load=50.0, pv=35.0)
        plan = ConvexDispatchEngine(node, dt_hours=1.0).solve()
        assert plan.success
        for t in range(plan.time_steps):
            lhs = (
                plan.p_grid_import[t]
                - plan.p_grid_export[t]
                + plan.p_discharge[t]
                - plan.p_charge[t]
                + plan.p_solar_used[t]
                + plan.p_load_shed[t]
            )
            rhs = plan.critical_load[t]
            assert abs(lhs - rhs) < 1e-5, f"balance fail at t={t}: {lhs} vs {rhs}"

    def test_soc_within_bounds(self):
        node = _make_flat_node(T=16, load=60.0, pv=20.0, s0=0.5)
        plan = ConvexDispatchEngine(node, dt_hours=1.0).solve()
        assert plan.success
        assert np.all(plan.soc >= node.battery.soc_min - 1e-6)
        assert np.all(plan.soc <= node.battery.soc_max + 1e-6)

    def test_nonnegativity_of_controls(self):
        node = _make_flat_node(T=8)
        plan = ConvexDispatchEngine(node).solve()
        assert plan.success
        for arr in (
            plan.p_charge,
            plan.p_discharge,
            plan.p_grid_import,
            plan.p_grid_export,
            plan.p_solar_curtailed,
            plan.p_load_shed,
        ):
            assert np.all(arr >= -1e-8)

    def test_curtailment_bounded_by_pv(self):
        node = _make_flat_node(T=6, load=10.0, pv=80.0)
        plan = ConvexDispatchEngine(node).solve()
        assert plan.success
        pv = node.solar.generation_profile_kw
        assert np.all(plan.p_solar_curtailed <= pv + 1e-6)

    def test_high_tariff_prefers_discharge(self):
        """When import tariff is high and PV is zero, battery should discharge."""
        T = 4
        batt = BatteryStorage(
            capacity_kwh=100.0,
            p_max_charge_kw=50.0,
            p_max_discharge_kw=50.0,
            initial_soc=0.8,
            target_soc=0.2,
            soc_min=0.1,
            soc_max=0.95,
            degradation_cost_per_kwh=0.001,
        )
        solar = RenewableSource("pv", 50.0, np.zeros(T))
        load = LoadProfile(critical_load_kw=np.full(T, 30.0), unserved_energy_penalty=50.0)
        grid = GridIntertie(
            p_max_import_kw=100.0,
            p_max_export_kw=50.0,
            tariff_import_per_kwh=np.full(T, 2.0),  # very expensive import
            tariff_export_per_kwh=np.full(T, 0.01),
            co2_intensity_kg_per_kwh=np.full(T, 0.5),
        )
        node = MicrogridNode("peak", batt, solar, load, grid)
        plan = ConvexDispatchEngine(node, carbon_weight_per_kg=0.0).solve()
        assert plan.success
        # Total discharge should be meaningful
        assert float(np.sum(plan.p_discharge)) > 1.0

    def test_to_dict_keys(self):
        node = _make_flat_node(T=4)
        plan = ConvexDispatchEngine(node).solve()
        d = plan.to_dict()
        assert d["success"] is True
        assert "total_cost" in d
        assert "total_carbon_emissions_kg" in d

    def test_horizon_truncation(self):
        node = _make_flat_node(T=10)
        plan = ConvexDispatchEngine(node).solve(horizon_steps=5)
        assert plan.success
        assert plan.time_steps == 5


class TestMicrogridSimulation:
    def test_synthetic_profiles_shapes(self):
        p = MicrogridSimulation.generate_synthetic_profiles(days=1, dt_hours=1.0, seed=1)
        assert p["solar_kw"].shape == (24,)
        assert p["load_kw"].shape == (24,)
        assert np.all(p["solar_kw"] >= 0)
        assert np.all(p["load_kw"] > 0)

    def test_synthetic_solar_night_zero(self):
        p = MicrogridSimulation.generate_synthetic_profiles(days=1, dt_hours=1.0, seed=2)
        # Hours 0-5 and 19-23 should be near-zero solar
        night = np.concatenate([p["solar_kw"][:5], p["solar_kw"][19:]])
        assert float(np.mean(night)) < 5.0

    def test_mpc_runs_and_soc_bounded(self):
        profiles = MicrogridSimulation.generate_synthetic_profiles(days=1, dt_hours=1.0, seed=3)
        T = 12  # shorter for speed
        batt = BatteryStorage(
            capacity_kwh=150.0,
            p_max_charge_kw=60.0,
            p_max_discharge_kw=60.0,
            initial_soc=0.5,
            target_soc=0.5,
            soc_min=0.1,
            soc_max=0.9,
        )
        node = MicrogridNode(
            node_id="mpc-test",
            battery=batt,
            solar=RenewableSource("pv", 120.0, profiles["solar_kw"][:T]),
            load=LoadProfile(critical_load_kw=profiles["load_kw"][:T]),
            grid=GridIntertie(
                p_max_import_kw=200.0,
                p_max_export_kw=150.0,
                tariff_import_per_kwh=profiles["tariff_import"][:T],
                tariff_export_per_kwh=profiles["tariff_export"][:T],
                co2_intensity_kg_per_kwh=profiles["co2_intensity"][:T],
            ),
        )
        sim = MicrogridSimulation(node, dt_hours=1.0)
        res = sim.run_receding_horizon_mpc(
            actual_pv=profiles["solar_kw"][:T],
            actual_load=profiles["load_kw"][:T],
            mpc_horizon_steps=6,
        )
        assert res.time_steps == T
        assert np.all(res.soc_history >= batt.soc_min - 1e-5)
        assert np.all(res.soc_history <= batt.soc_max + 1e-5)
        assert res.cumulative_cost == res.cumulative_cost  # not NaN


class TestNumericalInvariants:
    def test_zero_load_zero_pv_zero_flow(self):
        """With zero load and zero PV, optimal flows should be near zero (no free lunch)."""
        node = _make_flat_node(T=4, load=0.0, pv=0.0, s0=0.5)
        # target_soc = initial => no forced cycling
        plan = ConvexDispatchEngine(node, carbon_weight_per_kg=0.0).solve()
        assert plan.success
        assert float(np.sum(plan.p_charge + plan.p_discharge)) < 1e-3
        assert float(np.sum(plan.p_grid_import + plan.p_grid_export)) < 1e-3

    def test_energy_conservation_soc_dynamics(self):
        """SoC trajectory must match integrated charge/discharge dynamics."""
        node = _make_flat_node(T=10, load=45.0, pv=30.0, s0=0.55)
        engine = ConvexDispatchEngine(node, dt_hours=1.0)
        plan = engine.solve()
        assert plan.success
        batt = node.battery
        soc = batt.initial_soc
        for t in range(plan.time_steps):
            soc += (engine.dt / batt.capacity_kwh) * (
                batt.efficiency_charge * plan.p_charge[t]
                - (1.0 / batt.efficiency_discharge) * plan.p_discharge[t]
            )
            assert abs(soc - plan.soc[t]) < 1e-6
