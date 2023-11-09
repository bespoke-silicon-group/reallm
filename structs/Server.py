from dataclasses import dataclass
from typing import Optional

from .Base import Base
from .IO import IO
from .Package import Package
from .Heatsink import Heatsink
from .TCO import TCO
from .Constants import ServerConstants, ServerConstantsCommon

@dataclass
class Server(Base):
    package: Package
    packages_per_lane: int # number of packages per lan
    io: IO # server to server links
    num_lanes: int = 8 # number of lanes

    num_packages: Optional[int] = None # number of packages per server
    num_chips: Optional[int] = None # number of chips per server
    core_tdp: Optional[float] = None # core tdp including chips and memories
    # other_tdp: Optional[float] = None # other parts tdp
    tdp: Optional[float] = None # total tdp
    hs: Optional[Heatsink] = None # heatsinks used, per package

    cost: Optional[float] = None # server cost
    tco: Optional[TCO] = None # server tco at tdp

    perf: Optional[int] = None # #OPS
    sram: Optional[int] = None # Byte
    dram: Optional[int] = None # Byte

    valid: Optional[bool] = None
    invalid_reason: Optional[str] = None

    tops: Optional[float] = None
    sram_mb: Optional[float] = None # MByte
    total_mem: Optional[int] = None # sram and dram, Byte

    constants: ServerConstants = ServerConstantsCommon

    # for test
    cost_dcdc: float = 0
    cost_psu:  float = 0

    def update(self) -> None:
        if self.check_area():
            self.num_packages = self.num_lanes * self.packages_per_lane
            self.num_chips = self.num_packages * self.package.num_chips
            self.core_tdp = self.num_packages * self.package.tdp
            self.hs = Heatsink(heatsource_length=self.package.heatsource_length, 
                               heatsource_width=self.package.heatsource_width, 
                               packages_per_lane=self.packages_per_lane)

            if self.check_thermal():
                self.valid = True
                self.perf = self.package.perf * self.num_packages
                self.sram = self.package.sram * self.num_packages
                self.dram = self.package.dram * self.num_packages
                self.tdp = self._get_tdp()
                self.cost = self._get_cost()
                self.tco = TCO(self.tdp, self.cost, self.constants.SrvLife)
                self.tops = self.perf / 1e12
                self.sram_mb = self.sram / 1e6
                self.total_mem = self.sram + self.dram
            else:
                self.valid = False
        else:
            self.valid = False
    
    def check_area(self) -> bool:
        silicon_per_lane = (self.package.heatsource_length * self.package.heatsource_width) * self.packages_per_lane 
        if silicon_per_lane > self.constants.LaneAreaMax:
            self.invalid_reason = f'Lane silicon area {silicon_per_lane} too large'
            return False
        if silicon_per_lane < self.constants.LaneAreaMin:
            self.invalid_reason = f'Lane silicon area {silicon_per_lane} too small'
            return False
        return True

    def check_thermal(self) -> bool:
        if not self.hs.valid:
            self.invalid_reason = "Can't find valid heatsink configuration"
            return False
        if self.hs.max_power < self.package.tdp:
            self.invalid_reason = f"Heatsink {self.hs.max_power} can't cool the package tdp {self.package.tdp}"
            return False
        if self.constants.SrvMaxPower < self.core_tdp:
            self.invalid_reason = f"Server core power {self.core_tdp} is too large"
            return False
        return True

    def _get_tdp(self) -> float:
        w_fan = self.constants.FanPower * self.num_lanes
        # This is a bug in the original code, 0.5 is the w_IO per chip. It should instead be
        # w_dcdc = (self.package.tdp * self.num_packages + self.constants.APPPower) / self.constants.DCDCEfficiency
        w_dcdc = (self.package.tdp * self.num_packages + self.constants.APPPower + 0.5) / self.constants.DCDCEfficiency
        output_of_psu = w_dcdc + w_fan
        server_power = output_of_psu / self.constants.PSUEfficiency

        return server_power

    def _get_cost(self) -> float:
        c_dcdc = self.constants.APPPower / 1.0
        c_dcdc += (self.package.chip.core_tdp / self.package.chip.vdd) * self.num_chips
        if self.package.mem_3d:
            c_dcdc += (self.package.mem_3d.tdp / self.package.mem_3d.vdd) * self.num_chips
        if self.package.mem_side:
            c_dcdc += (self.package.mem_side.tdp / self.package.mem_side.vdd) * self.package.num_mem_side * self.num_packages
        cost_dcdc = self.constants.DCDCCostPerAmp * c_dcdc
        cost_psu = self.tdp * self.constants.PSUCostPerW

        cost_per_package = self.package.cost
        cost_all_package = cost_per_package * self.num_packages

        cost_all_heatsinks = self.hs.cost * self.num_packages

        cost_all_fans = self.constants.FanCost * self.num_lanes
        cost_all_ethernet = self.constants.EthernetCost

        pcb_parts_cost = self.constants.PCBPartsCost
        chassis_cost = self.constants.ChassisCost
        pcb_cost = self.constants.PCBCost * 2

        system_cost = pcb_cost + pcb_parts_cost + chassis_cost

        server_cost = cost_all_package + \
                      cost_all_heatsinks + \
                      cost_all_fans + \
                      cost_all_ethernet + \
                      cost_dcdc + \
                      cost_psu + \
                      system_cost
        self.cost_dcdc = cost_dcdc
        self.cost_psu = cost_psu
        
        return server_cost

