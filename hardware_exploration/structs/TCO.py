from dataclasses import dataclass, replace
from typing import Optional

from .Base import Base
from .Constants import TCOConstants, TCOConstantsCommon

@dataclass
class TCO(Base):
    server_tdp: float
    server_cost: float
    server_life: float

    dc_amortization: Optional[float] = None
    dc_interest: Optional[float] = None
    dc_opex: Optional[float] = None
    srv_power: Optional[float] = None
    pue_overhead: Optional[float] = None
    srv_amortization: Optional[float] = None
    srv_interest: Optional[float] = None
    srv_opex: Optional[float] = None

    total: Optional[float] = None
    capex: Optional[float] = None
    opex: Optional[float] = None # depends on the real utilization (power)

    constants: TCOConstants = TCOConstantsCommon

    def update(self) -> None:
        # Fixed parts of TCO
        self.dc_amortization = self.constants.DCCapex * self.server_tdp * self.server_life / self.constants.DCAmortPeriod
        self.dc_interest = evalTotalInterestPaid(self.constants.DCCapex, self.constants.InterestRate, self.constants.DCAmortPeriod) \
              * self.server_tdp * self.server_life / self.constants.DCAmortPeriod
        self.dc_opex = self.constants.DCOpex * self.server_tdp * self.server_life * 12

        # Electricity cost is in $/KWh, so 1000 is for K, and it calculates for a month
        self.srv_power = self.constants.ElectricityCost * self.constants.SrvAvgPwr / 1000 * 24 * 30 \
              * self.server_tdp * self.server_life * 12
        self.pue_overhead = self.srv_power * (self.constants.PUE - 1.0) * self.server_tdp * self.server_life * 12

        # TCO portions dependent on inputs
        self.srv_amortization = self.server_cost
        self.srv_opex  = self.srv_amortization * self.constants.SrvOpexRate
        self.srv_interest = evalTotalInterestPaid(self.server_cost, self.constants.InterestRate, self.server_life)

        # Accumulate costs
        self.total = self.dc_amortization + self.dc_interest + self.dc_opex \
                   + self.srv_amortization + self.srv_opex + self.srv_interest \
                   + self.srv_power + self.pue_overhead

def evalTotalInterestPaid(total_borrowing, annual_interest_rate, loan_period_year):
    num_of_payments = loan_period_year * 12
    monthly_r = annual_interest_rate / 12

    monthly_payment = monthly_r * total_borrowing * (1 + monthly_r)**num_of_payments / ((1 + monthly_r)**num_of_payments - 1)

    return monthly_payment * num_of_payments - total_borrowing