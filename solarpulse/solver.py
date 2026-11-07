"""
Convex Optimization Engine for Microgrid Economic & Environmental Dispatch.

Solves the multi-period finite-horizon convex optimal power dispatch via Linear/Quadratic
Programming using scipy.optimize.linprog (Highs interior-point/simplex) and coordinate projection.
"""
from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np
from scipy.optimize import linprog

from .models import MicrogridNode


@dataclass
class DispatchPlan:
    """Optimal multi-period dispatch trajectory results."""
    success: bool
    message: str
    time_steps: int
    dt_hours: float
    total_cost: float
    total_grid_import_kwh: float
    total_grid_export_kwh: float
    total_solar_generated_kwh: float
    total_solar_curtailed_kwh: float
    total_load_shed_kwh: float
    total_battery_throughput_kwh: float
    total_carbon_emissions_kg: float
    
    # Time series trajectories (arrays of length T)
    p_charge: np.ndarray
    p_discharge: np.ndarray
    soc: np.ndarray
    p_grid_import: np.ndarray
    p_grid_export: np.ndarray
    p_solar_used: np.ndarray
    p_solar_curtailed: np.ndarray
    p_load_shed: np.ndarray
    critical_load: np.ndarray
    import_tariffs: np.ndarray
    export_tariffs: np.ndarray

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "message": self.message,
            "total_cost": float(self.total_cost),
            "total_grid_import_kwh": float(self.total_grid_import_kwh),
            "total_grid_export_kwh": float(self.total_grid_export_kwh),
            "total_solar_generated_kwh": float(self.total_solar_generated_kwh),
            "total_solar_curtailed_kwh": float(self.total_solar_curtailed_kwh),
            "total_load_shed_kwh": float(self.total_load_shed_kwh),
            "total_battery_throughput_kwh": float(self.total_battery_throughput_kwh),
            "total_carbon_emissions_kg": float(self.total_carbon_emissions_kg),
        }


class ConvexDispatchEngine:
    """
    Mathematical programming engine formulating microgrid dispatch as a convex LP:
    
    Decision variables per timestep t in 0..T-1:
      - p_ch[t]   : Battery charging power (kW) >= 0
      - p_dis[t]  : Battery discharging power (kW) >= 0
      - p_imp[t]  : Grid import power (kW) >= 0
      - p_exp[t]  : Grid export power (kW) >= 0
      - p_curt[t] : Solar curtailment power (kW) >= 0
      - p_shed[t] : Load shed power (kW) >= 0
    
    Objective Function:
      Minimize Sum_{t=0}^{T-1} [
          dt * (
              c_imp[t] * p_imp[t] 
            - c_exp[t] * p_exp[t] 
            + c_deg * (p_ch[t] + p_dis[t]) 
            + c_curt * p_curt[t] 
            + c_shed * p_shed[t]
            + carbon_weight * co2[t] * p_imp[t]
          )
      ]
    
    Subject to:
      1. Power Balance: p_imp[t] - p_exp[t] + p_dis[t] - p_ch[t] + (P_pv[t] - p_curt[t]) + p_shed[t] = P_load[t]
      2. State of Charge Dynamics and Bounds: S_min <= S[t] <= S_max for all t
      3. Terminal SoC Constraint: S[T] >= target_soc
      4. Converter / line capacity bounds: p_ch <= P_ch_max, p_dis <= P_dis_max, p_imp <= P_imp_max, etc.
    """

    def __init__(self, node: MicrogridNode, dt_hours: float = 1.0, carbon_weight_per_kg: float = 0.05):
        self.node = node
        self.dt = dt_hours
        self.carbon_weight = carbon_weight_per_kg

    def solve(self, horizon_steps: Optional[int] = None) -> DispatchPlan:
        pv_profile = self.node.solar.generation_profile_kw
        load_profile = self.node.load.critical_load_kw
        T = len(pv_profile) if horizon_steps is None else min(horizon_steps, len(pv_profile))
        
        if len(load_profile) < T:
            raise ValueError(f"Load profile length ({len(load_profile)}) is shorter than horizon ({T})")

        pv = pv_profile[:T]
        load = load_profile[:T]
        
        grid = self.node.grid
        tariff_imp = grid.tariff_import_per_kwh[:T] if len(grid.tariff_import_per_kwh) >= T else np.full(T, 0.15)
        tariff_exp = grid.tariff_export_per_kwh[:T] if len(grid.tariff_export_per_kwh) >= T else np.full(T, 0.05)
        co2_intensity = grid.co2_intensity_kg_per_kwh[:T] if (grid.co2_intensity_kg_per_kwh is not None and len(grid.co2_intensity_kg_per_kwh) >= T) else np.full(T, 0.40)

        batt = self.node.battery
        eta_c = batt.efficiency_charge
        eta_d = batt.efficiency_discharge
        cap = batt.capacity_kwh
        s0 = batt.initial_soc
        s_min = batt.soc_min
        s_max = batt.soc_max
        s_target = batt.target_soc if batt.target_soc is not None else s0

        # Number of variables per step = 6: [p_ch, p_dis, p_imp, p_exp, p_curt, p_shed]
        NV = 6
        N_total = NV * T

        # Helper index lambdas
        idx_ch = lambda t: t * NV + 0
        idx_dis = lambda t: t * NV + 1
        idx_imp = lambda t: t * NV + 2
        idx_exp = lambda t: t * NV + 3
        idx_curt = lambda t: t * NV + 4
        idx_shed = lambda t: t * NV + 5

        # Build Objective vector c
        c = np.zeros(N_total)
        for t in range(T):
            # battery wear
            c[idx_ch(t)] = self.dt * batt.degradation_cost_per_kwh
            c[idx_dis(t)] = self.dt * batt.degradation_cost_per_kwh
            # grid import (tariff + carbon penalty)
            c[idx_imp(t)] = self.dt * (tariff_imp[t] + self.carbon_weight * co2_intensity[t])
            # grid export (revenue)
            c[idx_exp(t)] = -self.dt * tariff_exp[t]
            # curtailment penalty
            c[idx_curt(t)] = self.dt * self.node.solar.curtailment_penalty_per_kwh
            # load shedding penalty
            c[idx_shed(t)] = self.dt * self.node.load.unserved_energy_penalty

        # 1. Equality Constraints: A_eq @ x = b_eq (Power Balance for each t)
        # p_imp - p_exp + p_dis - p_ch - p_curt + p_shed = load[t] - pv[t]
        A_eq = np.zeros((T, N_total))
        b_eq = np.zeros(T)
        for t in range(T):
            A_eq[t, idx_imp(t)] = 1.0
            A_eq[t, idx_exp(t)] = -1.0
            A_eq[t, idx_dis(t)] = 1.0
            A_eq[t, idx_ch(t)] = -1.0
            A_eq[t, idx_curt(t)] = -1.0
            A_eq[t, idx_shed(t)] = 1.0
            b_eq[t] = load[t] - pv[t]

        # 2. Inequality Constraints: A_ub @ x <= b_ub
        # State of charge evolution:
        # S[k] = s0 + (dt / cap) * sum_{t=0}^{k-1} (eta_c * p_ch[t] - (1/eta_d) * p_dis[t])
        # S[k] <= s_max  =>  (dt/cap)*sum (eta_c * p_ch - p_dis/eta_d) <= s_max - s0
        # S[k] >= s_min  => -(dt/cap)*sum (eta_c * p_ch - p_dis/eta_d) <= s0 - s_min
        # Terminal constraint: S[T] >= s_target
        # Total inequalities: T (max soc) + T (min soc) + 1 (terminal soc) = 2*T + 1
        num_ineq = 2 * T + 1
        A_ub = np.zeros((num_ineq, N_total))
        b_ub = np.zeros(num_ineq)

        row = 0
        # Upper SoC bounds: k from 1 to T
        for k in range(1, T + 1):
            for t in range(k):
                A_ub[row, idx_ch(t)] = (self.dt / cap) * eta_c
                A_ub[row, idx_dis(t)] = -(self.dt / cap) * (1.0 / eta_d)
            b_ub[row] = s_max - s0
            row += 1

        # Lower SoC bounds: k from 1 to T
        for k in range(1, T + 1):
            for t in range(k):
                A_ub[row, idx_ch(t)] = -(self.dt / cap) * eta_c
                A_ub[row, idx_dis(t)] = (self.dt / cap) * (1.0 / eta_d)
            b_ub[row] = s0 - s_min
            row += 1

        # Terminal SoC bound: -S[T] <= -s_target
        for t in range(T):
            A_ub[row, idx_ch(t)] = -(self.dt / cap) * eta_c
            A_ub[row, idx_dis(t)] = (self.dt / cap) * (1.0 / eta_d)
        b_ub[row] = s0 - s_target
        row += 1

        # Bounds on variables: 0 <= var <= upper_bound
        bounds = []
        for t in range(T):
            bounds.append((0.0, batt.p_max_charge_kw))     # p_ch
            bounds.append((0.0, batt.p_max_discharge_kw))  # p_dis
            bounds.append((0.0, grid.p_max_import_kw))      # p_imp
            bounds.append((0.0, grid.p_max_export_kw))      # p_exp
            bounds.append((0.0, float(pv[t])))              # p_curt <= pv available
            bounds.append((0.0, float(load[t])))            # p_shed <= load

        # Solve with HiGHS interior-point/simplex via scipy
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

        if not res.success:
            return DispatchPlan(
                success=False,
                message=res.message,
                time_steps=T,
                dt_hours=self.dt,
                total_cost=0.0,
                total_grid_import_kwh=0.0,
                total_grid_export_kwh=0.0,
                total_solar_generated_kwh=float(np.sum(pv) * self.dt),
                total_solar_curtailed_kwh=0.0,
                total_load_shed_kwh=0.0,
                total_battery_throughput_kwh=0.0,
                total_carbon_emissions_kg=0.0,
                p_charge=np.zeros(T),
                p_discharge=np.zeros(T),
                soc=np.full(T, s0),
                p_grid_import=np.zeros(T),
                p_grid_export=np.zeros(T),
                p_solar_used=np.zeros(T),
                p_solar_curtailed=np.zeros(T),
                p_load_shed=np.zeros(T),
                critical_load=load,
                import_tariffs=tariff_imp,
                export_tariffs=tariff_exp,
            )

        x = res.x
        p_ch = np.array([x[idx_ch(t)] for t in range(T)])
        p_dis = np.array([x[idx_dis(t)] for t in range(T)])
        p_imp = np.array([x[idx_imp(t)] for t in range(T)])
        p_exp = np.array([x[idx_exp(t)] for t in range(T)])
        p_curt = np.array([x[idx_curt(t)] for t in range(T)])
        p_shed = np.array([x[idx_shed(t)] for t in range(T)])
        p_solar_used = pv - p_curt

        # Compute SoC trajectory
        soc_traj = np.zeros(T)
        curr_soc = s0
        for t in range(T):
            curr_soc += (self.dt / cap) * (eta_c * p_ch[t] - (1.0 / eta_d) * p_dis[t])
            soc_traj[t] = curr_soc

        total_cost = res.fun
        total_imp_kwh = float(np.sum(p_imp) * self.dt)
        total_exp_kwh = float(np.sum(p_exp) * self.dt)
        total_pv_gen_kwh = float(np.sum(pv) * self.dt)
        total_curt_kwh = float(np.sum(p_curt) * self.dt)
        total_shed_kwh = float(np.sum(p_shed) * self.dt)
        total_throughput_kwh = float(np.sum(p_ch + p_dis) * self.dt)
        total_carbon_kg = float(np.sum(p_imp * co2_intensity) * self.dt)

        return DispatchPlan(
            success=True,
            message="Optimal dispatch plan converged.",
            time_steps=T,
            dt_hours=self.dt,
            total_cost=float(total_cost),
            total_grid_import_kwh=total_imp_kwh,
            total_grid_export_kwh=total_exp_kwh,
            total_solar_generated_kwh=total_pv_gen_kwh,
            total_solar_curtailed_kwh=total_curt_kwh,
            total_load_shed_kwh=total_shed_kwh,
            total_battery_throughput_kwh=total_throughput_kwh,
            total_carbon_emissions_kg=total_carbon_kg,
            p_charge=p_ch,
            p_discharge=p_dis,
            soc=soc_traj,
            p_grid_import=p_imp,
            p_grid_export=p_exp,
            p_solar_used=p_solar_used,
            p_solar_curtailed=p_curt,
            p_load_shed=p_shed,
            critical_load=load,
            import_tariffs=tariff_imp,
            export_tariffs=tariff_exp,
        )
