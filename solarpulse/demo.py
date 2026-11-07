"""
Interactive CLI and ANSI visualization for SolarPulse Microgrid Convex Optimal Dispatch Engine.
"""
from __future__ import annotations

import sys
import numpy as np

from solarpulse.models import (
    MicrogridNode,
    BatteryStorage,
    RenewableSource,
    LoadProfile,
    GridIntertie,
)
from solarpulse.solver import ConvexDispatchEngine
from solarpulse.dynamics import MicrogridSimulation


def render_ascii_sparkline(arr: np.ndarray, width: int = 36) -> str:
    """Render a compact Unicode block sparkline for a 1-D series."""
    if arr.size == 0:
        return " " * width
    # Downsample if needed
    if arr.size > width:
        idx = np.linspace(0, arr.size - 1, width).astype(int)
        series = arr[idx]
    else:
        series = arr
        width = arr.size
    blocks = " ▁▂▃▄▅▆▇█"
    lo, hi = float(np.min(series)), float(np.max(series))
    span = hi - lo if hi > lo else 1.0
    out = []
    for v in series:
        level = int(round((float(v) - lo) / span * (len(blocks) - 1)))
        out.append(blocks[level])
    return "".join(out)


def render_soc_bar(soc: float, width: int = 24) -> str:
    filled = int(round(np.clip(soc, 0.0, 1.0) * width))
    return "[" + ("█" * filled) + ("░" * (width - filled)) + f"] {soc * 100:5.1f}%"


def build_demo_node(days: int = 1, dt_hours: float = 1.0) -> MicrogridNode:
    profiles = MicrogridSimulation.generate_synthetic_profiles(days=days, dt_hours=dt_hours, seed=7)
    battery = BatteryStorage(
        capacity_kwh=200.0,
        p_max_charge_kw=80.0,
        p_max_discharge_kw=80.0,
        efficiency_charge=0.95,
        efficiency_discharge=0.95,
        soc_min=0.15,
        soc_max=0.90,
        initial_soc=0.45,
        target_soc=0.45,
        degradation_cost_per_kwh=0.025,
    )
    solar = RenewableSource(
        name="rooftop-pv",
        rated_kw=150.0,
        generation_profile_kw=profiles["solar_kw"],
        curtailment_penalty_per_kwh=0.02,
    )
    load = LoadProfile(
        critical_load_kw=profiles["load_kw"],
        unserved_energy_penalty=25.0,
    )
    grid = GridIntertie(
        p_max_import_kw=250.0,
        p_max_export_kw=180.0,
        tariff_import_per_kwh=profiles["tariff_import"],
        tariff_export_per_kwh=profiles["tariff_export"],
        co2_intensity_kg_per_kwh=profiles["co2_intensity"],
    )
    return MicrogridNode(
        node_id="campus-microgrid-alpha",
        battery=battery,
        solar=solar,
        load=load,
        grid=grid,
    )


def print_header() -> None:
    print("=" * 72)
    print("  SolarPulse  ·  Convex Microgrid Optimal Battery Dispatch Engine")
    print("  Decentralized renewable microgrids · LP/MPC · Carbon-aware ToU")
    print("=" * 72)
    print()


def run_open_loop_demo() -> None:
    print_header()
    print("[1/2] Open-loop finite-horizon convex economic dispatch (24h)…")
    node = build_demo_node(days=1, dt_hours=1.0)
    engine = ConvexDispatchEngine(node, dt_hours=1.0, carbon_weight_per_kg=0.08)
    plan = engine.solve()

    if not plan.success:
        print(f"  ✗ Solver failed: {plan.message}")
        sys.exit(1)

    print(f"  ✓ Optimal plan converged  |  objective cost = ${plan.total_cost:,.2f}")
    print()
    print("  Trajectory summary (kWh / kg):")
    print(f"    Solar generated : {plan.total_solar_generated_kwh:8.1f} kWh")
    print(f"    Solar curtailed : {plan.total_solar_curtailed_kwh:8.1f} kWh")
    print(f"    Grid import     : {plan.total_grid_import_kwh:8.1f} kWh")
    print(f"    Grid export     : {plan.total_grid_export_kwh:8.1f} kWh")
    print(f"    Battery thruput : {plan.total_battery_throughput_kwh:8.1f} kWh")
    print(f"    Load shed       : {plan.total_load_shed_kwh:8.1f} kWh")
    print(f"    CO₂ emissions   : {plan.total_carbon_emissions_kg:8.1f} kg")
    print()
    print("  ASCII trajectories (24h, left→right = midnight→midnight):")
    print(f"    PV gen     {render_ascii_sparkline(node.solar.generation_profile_kw)}")
    print(f"    Load       {render_ascii_sparkline(plan.critical_load)}")
    print(f"    Charge     {render_ascii_sparkline(plan.p_charge)}")
    print(f"    Discharge  {render_ascii_sparkline(plan.p_discharge)}")
    print(f"    Grid imp   {render_ascii_sparkline(plan.p_grid_import)}")
    print(f"    Grid exp   {render_ascii_sparkline(plan.p_grid_export)}")
    print(f"    SoC        {render_ascii_sparkline(plan.soc)}")
    print()
    print(f"  Final SoC bar: {render_soc_bar(float(plan.soc[-1]))}")
    print()


def run_mpc_demo() -> None:
    print("[2/2] Closed-loop receding-horizon MPC (48h, N=12)…")
    node = build_demo_node(days=2, dt_hours=1.0)
    sim = MicrogridSimulation(node, dt_hours=1.0)
    profiles = {
        "solar_kw": node.solar.generation_profile_kw,
        "load_kw": node.load.critical_load_kw,
    }
    results = sim.run_receding_horizon_mpc(
        actual_pv=profiles["solar_kw"],
        actual_load=profiles["load_kw"],
        mpc_horizon_steps=12,
    )
    print(f"  ✓ MPC closed-loop complete  |  realized cost = ${results.cumulative_cost:,.2f}")
    print(f"    CO₂ emissions (realized) : {results.total_carbon_emissions_kg:8.1f} kg")
    print(f"    Final SoC                : {render_soc_bar(float(results.soc_history[-1]))}")
    print()
    print("  MPC trajectories (48h):")
    print(f"    PV gen     {render_ascii_sparkline(results.actual_pv, width=48)}")
    print(f"    Load       {render_ascii_sparkline(results.actual_load, width=48)}")
    print(f"    Discharge  {render_ascii_sparkline(results.dispatched_discharge, width=48)}")
    print(f"    Charge     {render_ascii_sparkline(results.dispatched_charge, width=48)}")
    print(f"    SoC        {render_ascii_sparkline(results.soc_history, width=48)}")
    print()
    print("=" * 72)
    print("  Demo complete. SolarPulse is ready for campus / islanded microgrids.")
    print("=" * 72)


def main() -> None:
    run_open_loop_demo()
    run_mpc_demo()


if __name__ == "__main__":
    main()
