"""
Stochastic Microgrid Real-Time Simulation and Model Predictive Control (MPC) Receding Horizon.
"""
from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np

from .models import MicrogridNode, BatteryStorage, RenewableSource, LoadProfile, GridIntertie
from .solver import ConvexDispatchEngine, DispatchPlan


@dataclass
class SimulationResults:
    """Closed-loop simulation outcomes."""
    time_steps: int
    dt_hours: float
    actual_load: np.ndarray
    actual_pv: np.ndarray
    dispatched_charge: np.ndarray
    dispatched_discharge: np.ndarray
    soc_history: np.ndarray
    grid_import_history: np.ndarray
    grid_export_history: np.ndarray
    unserved_load_history: np.ndarray
    cumulative_cost: float
    total_carbon_emissions_kg: float


class MicrogridSimulation:
    """
    High-fidelity closed-loop dynamic simulator with MPC receding horizon optimization
    and stochastic renewable/load disturbances.
    """

    def __init__(self, node: MicrogridNode, dt_hours: float = 1.0):
        self.node = node
        self.dt = dt_hours

    @staticmethod
    def generate_synthetic_profiles(days: int = 1, dt_hours: float = 1.0, seed: int = 42) -> Dict[str, np.ndarray]:
        """
        Generates realistic diurnal solar irradiance, industrial/residential load patterns,
        time-of-use (ToU) electricity tariffs, and marginal grid CO2 emissions.
        """
        np.random.seed(seed)
        total_hours = int(days * 24 / dt_hours)
        t = np.linspace(0, 24 * days, total_hours, endpoint=False)
        hour_of_day = t % 24

        # Solar irradiance (bell curve peaking at solar noon ~12-13h)
        solar_peak = 120.0  # kW
        solar_raw = np.maximum(0.0, np.sin(np.pi * (hour_of_day - 6) / 12))
        solar_raw[(hour_of_day < 6) | (hour_of_day > 18)] = 0.0
        # Solar irradiance with cloud turbulence
        turbulence = 1.0 - 0.15 * np.sin(0.8 * t) + 0.05 * np.random.randn(total_hours)
        turbulence = np.clip(turbulence, 0.4, 1.1)
        solar_kw = solar_peak * (solar_raw ** 1.3) * turbulence
        solar_kw = np.maximum(0.0, solar_kw)

        # Dual-peak electric load: morning rise (08:00) + evening peak (19:00)
        base_load = 35.0
        morning_peak = 30.0 * np.exp(-((hour_of_day - 9) ** 2) / 8.0)
        evening_peak = 55.0 * np.exp(-((hour_of_day - 19) ** 2) / 6.0)
        noise = np.random.normal(0, 3.0, total_hours)
        load_kw = np.maximum(15.0, base_load + morning_peak + evening_peak + noise)

        # Time of Use (ToU) pricing ($/kWh)
        # Off-peak: $0.08, Mid-peak: $0.18, On-peak (16-21h): $0.42
        tariff_imp = np.full(total_hours, 0.08)
        tariff_imp[(hour_of_day >= 7) & (hour_of_day < 16)] = 0.18
        tariff_imp[(hour_of_day >= 16) & (hour_of_day <= 21)] = 0.42
        tariff_exp = tariff_imp * 0.45  # Feed-in tariff discount

        # Grid marginal CO2 intensity (kg CO2 / kWh)
        # Dirtier thermal generation dispatched during high-demand peaks
        co2_intensity = 0.30 + 0.35 * (tariff_imp / 0.42)

        return {
            "solar_kw": solar_kw,
            "load_kw": load_kw,
            "tariff_import": tariff_imp,
            "tariff_export": tariff_exp,
            "co2_intensity": co2_intensity,
        }

    def run_receding_horizon_mpc(
        self,
        actual_pv: np.ndarray,
        actual_load: np.ndarray,
        mpc_horizon_steps: int = 24,
    ) -> SimulationResults:
        """
        Executes closed-loop Model Predictive Control over the given simulation horizon.
        At each step t, an optimal plan is resolved over [t, t+N], the first control action
        is applied to the physical state, and the state progresses.
        """
        T = len(actual_pv)
        batt = self.node.battery
        cap = batt.capacity_kwh
        eta_c = batt.efficiency_charge
        eta_d = batt.efficiency_discharge

        p_ch_hist = np.zeros(T)
        p_dis_hist = np.zeros(T)
        soc_hist = np.zeros(T)
        p_imp_hist = np.zeros(T)
        p_exp_hist = np.zeros(T)
        p_shed_hist = np.zeros(T)

        current_soc = batt.initial_soc
        cumulative_cost = 0.0
        total_carbon = 0.0

        for t in range(T):
            # Lookahead window
            h = min(mpc_horizon_steps, T - t)
            forecast_pv = actual_pv[t : t + h]
            forecast_load = actual_load[t : t + h]
            forecast_tariff_imp = self.node.grid.tariff_import_per_kwh[t : t + h]
            forecast_tariff_exp = self.node.grid.tariff_export_per_kwh[t : t + h]
            forecast_co2 = (
                self.node.grid.co2_intensity_kg_per_kwh[t : t + h]
                if self.node.grid.co2_intensity_kg_per_kwh is not None
                else None
            )

            step_batt = BatteryStorage(
                capacity_kwh=batt.capacity_kwh,
                p_max_charge_kw=batt.p_max_charge_kw,
                p_max_discharge_kw=batt.p_max_discharge_kw,
                efficiency_charge=batt.efficiency_charge,
                efficiency_discharge=batt.efficiency_discharge,
                soc_min=batt.soc_min,
                soc_max=batt.soc_max,
                initial_soc=float(np.clip(current_soc, batt.soc_min, batt.soc_max)),
                target_soc=batt.target_soc,
                degradation_cost_per_kwh=batt.degradation_cost_per_kwh,
            )
            step_node = MicrogridNode(
                node_id=f"{self.node.node_id}_step_{t}",
                battery=step_batt,
                solar=RenewableSource(name="solar", rated_kw=self.node.solar.rated_kw, generation_profile_kw=forecast_pv),
                load=LoadProfile(critical_load_kw=forecast_load),
                grid=GridIntertie(
                    p_max_import_kw=self.node.grid.p_max_import_kw,
                    p_max_export_kw=self.node.grid.p_max_export_kw,
                    tariff_import_per_kwh=forecast_tariff_imp,
                    tariff_export_per_kwh=forecast_tariff_exp,
                    co2_intensity_kg_per_kwh=forecast_co2,
                ),
            )

            engine = ConvexDispatchEngine(step_node, dt_hours=self.dt)
            plan = engine.solve(horizon_steps=h)

            if plan.success and len(plan.p_charge) > 0:
                u_ch = plan.p_charge[0]
                u_dis = plan.p_discharge[0]
                u_imp = plan.p_grid_import[0]
                u_exp = plan.p_grid_export[0]
                u_shed = plan.p_load_shed[0]
            else:
                # Robust fallback heuristic if LP is singular/infeasible
                net_power = forecast_load[0] - forecast_pv[0]
                u_ch = 0.0
                u_dis = max(0.0, min(batt.p_max_discharge_kw, net_power))
                u_imp = max(0.0, net_power - u_dis)
                u_exp = max(0.0, -net_power)
                u_shed = 0.0

            # Advance physical state
            d_soc = (self.dt / cap) * (eta_c * u_ch - (1.0 / eta_d) * u_dis)
            current_soc = np.clip(current_soc + d_soc, batt.soc_min, batt.soc_max)

            p_ch_hist[t] = u_ch
            p_dis_hist[t] = u_dis
            soc_hist[t] = current_soc
            p_imp_hist[t] = u_imp
            p_exp_hist[t] = u_exp
            p_shed_hist[t] = u_shed

            step_imp_cost = forecast_tariff_imp[0] * u_imp * self.dt
            step_exp_rev = forecast_tariff_exp[0] * u_exp * self.dt
            step_deg = batt.degradation_cost_per_kwh * (u_ch + u_dis) * self.dt
            step_carbon = (forecast_co2[0] if forecast_co2 is not None else 0.4) * u_imp * self.dt
            cumulative_cost += (step_imp_cost - step_exp_rev + step_deg)
            total_carbon += step_carbon

        return SimulationResults(
            time_steps=T,
            dt_hours=self.dt,
            actual_load=actual_load,
            actual_pv=actual_pv,
            dispatched_charge=p_ch_hist,
            dispatched_discharge=p_dis_hist,
            soc_history=soc_hist,
            grid_import_history=p_imp_hist,
            grid_export_history=p_exp_hist,
            unserved_load_history=p_shed_hist,
            cumulative_cost=float(cumulative_cost),
            total_carbon_emissions_kg=float(total_carbon),
        )
