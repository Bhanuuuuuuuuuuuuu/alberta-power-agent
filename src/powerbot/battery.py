"""Stage 3: the battery and how it decides when to charge and discharge.

Battery (similar to small Alberta grid batteries):
    10 MWh energy, 5 MW power (2 hours to fill or empty), 90% round trip
    efficiency, and a wear cost for every MWh it pushes out.

Each midnight the optimizer gets 24 price forecasts and solves a linear program:

    maximize   sum_t  price_t * (discharge_t - charge_t)  -  wear * discharge_t
    subject to 0 <= charge_t, discharge_t <= 5 MW
               soc_t = soc_{t-1} + eta * charge_t - discharge_t / eta
               0 <= soc_t <= 10 MWh
               soc at end of day = soc at start of day = empty (no free energy,
               and each day is settled on its own)
               energy discharged per day <= 1 full cycle (warranty limit)

eta = sqrt(0.9), so a full charge then discharge loses 10%.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog


@dataclass(frozen=True)
class Battery:
    capacity_mwh: float = 10.0
    power_mw: float = 5.0
    round_trip_eff: float = 0.90
    wear_cost: float = 10.0       # $ per MWh discharged (battery degradation)
    start_soc_mwh: float = 0.0    # start and end each day empty
    max_cycles_per_day: float = 1.0  # warranty style limit on battery use

    @property
    def eta(self) -> float:
        return float(np.sqrt(self.round_trip_eff))


def optimize_day(prices: np.ndarray, bat: Battery = Battery(),
                 charge_cap: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Best charge and discharge plan for one day of (forecast) prices.

    charge_cap optionally limits charging per hour (the desk uses 0 to block
    buying in hours with high spike risk).
    Returns (charge, discharge, soc), each length 24, in MW / MWh.
    Variables are stacked as x = [charge(24), discharge(24), soc(24)].
    """
    T = len(prices)
    p = np.asarray(prices, dtype=float)
    eta = bat.eta

    # linprog minimizes, so flip the sign of profit.
    c = np.concatenate([p, -(p - bat.wear_cost), np.zeros(T)])

    # Energy balance: soc_t - soc_{t-1} - eta*charge_t + discharge_t/eta = 0
    A_eq = np.zeros((T + 1, 3 * T))
    b_eq = np.zeros(T + 1)
    for t in range(T):
        A_eq[t, t] = -eta
        A_eq[t, T + t] = 1 / eta
        A_eq[t, 2 * T + t] = 1
        if t == 0:
            b_eq[t] = bat.start_soc_mwh
        else:
            A_eq[t, 2 * T + t - 1] = -1
    A_eq[T, 3 * T - 1] = 1          # end of day soc
    b_eq[T] = bat.start_soc_mwh

    # Cycle limit: total energy drawn from the battery per day <= max_cycles * capacity
    A_ub = np.zeros((1, 3 * T))
    A_ub[0, T:2 * T] = 1 / eta
    b_ub = np.array([bat.max_cycles_per_day * bat.capacity_mwh])

    caps = np.full(T, bat.power_mw) if charge_cap is None else np.minimum(charge_cap, bat.power_mw)
    bounds = ([(0, float(cap)) for cap in caps] + [(0, bat.power_mw)] * T
              + [(0, bat.capacity_mwh)] * T)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(res.message)
    x = res.x
    return x[:T], x[T:2 * T], x[2 * T:]


def rule_day(hours_charge: list[int], hours_discharge: list[int], bat: Battery = Battery()):
    """Simple fixed schedule: always charge and discharge at the same hours."""
    T = 24
    charge = np.zeros(T)
    discharge = np.zeros(T)
    assert max(hours_charge) < min(hours_discharge), "charge before discharging"
    # Fill the battery, then empty it, so the day ends where it started.
    fill = bat.capacity_mwh - bat.start_soc_mwh
    charge[hours_charge] = min(bat.power_mw, fill / bat.eta / len(hours_charge))
    stored = bat.eta * charge.sum()
    discharge[hours_discharge] = min(bat.power_mw, stored * bat.eta / len(hours_discharge))
    soc = bat.start_soc_mwh + np.cumsum(bat.eta * charge - discharge / bat.eta)
    return charge, discharge, soc


def settle(actual_prices: np.ndarray, charge: np.ndarray, discharge: np.ndarray,
           bat: Battery = Battery()) -> float:
    """Real profit for the day: plan is paid at ACTUAL prices, not forecasts."""
    a = np.asarray(actual_prices, dtype=float)
    return float(np.sum(a * (discharge - charge)) - bat.wear_cost * discharge.sum())
