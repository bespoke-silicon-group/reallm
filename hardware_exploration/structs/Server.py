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

    num_packages: Optional[int] = None # number of packages per server
    num_chips: Optional[int] = None # number of chips per server
    core_tdp: Optional[float] = None # core tdp including chips and memories
    # other_tdp: Optional[float] = None # other parts tdp
    tdp: Optional[float] = None # total tdp
    hs: Optional[Heatsink] = None # heatsinks used, per package

    cost: Optional[float] = None # server cost
    tco: Optional[TCO] = None # server tco at 

    perf: Optional[float] = None # #OPS
    sram: Optional[float] = None # Byte
    dram: Optional[float] = None # Byte

    valid: Optional[bool] = None

    constants: ServerConstants = ServerConstantsCommon

    def update(self) -> None:
        self.num_packages = self.constants.SrvLanes * self.packages_per_lane
        self.num_chips = self.num_packages * self.package.num_chips
        self.core_tdp = self.num_packages * self.package.tdp
        self.perf = self.package.perf * self.num_packages
        self.sram = self.package.sram * self.num_packages
        self.dram = self.package.dram * self.num_packages
        self.tdp = self._get_tdp()
        self.hs = Heatsink(heatsource_length=self.package.heatsource_length, 
                           heatsource_width=self.package.heatsource_width, 
                           packages_per_lane=self.packages_per_lane)

        self.cost = self._get_cost()
        self.tco = TCO(self.tdp, self.cost, self.constants.SrvLife)

        if not self.too_hot():
            self.valid = True
        else:
            self.valid = False

    def too_hot(self) -> bool:
        return (not self.hs.valid) or (self.hs.max_power < self.package.tdp) or (self.constants.SrvMaxPower < self.core_tdp)

    def _get_tdp(self) -> float:
        w_fan = self.constants.FanPower * self.constants.SrvLanes
        w_dcdc = (self.package.tdp * self.num_packages + self.constants.APPPower) / self.constants.DCDCEfficiency
        output_of_psu = w_dcdc + w_fan
        server_power = output_of_psu / self.constants.PSUEfficiency

        return server_power

    def _get_cost(self) -> float:
        c_dcdc = self.constants.APPPower / 1.0
        c_dcdc += (self.package.chip.tdp / self.package.chip.vdd) * self.num_chips
        if self.package.mem_3d:
            c_dcdc += (self.package.mem_3d.tdp / self.package.mem_3d.vdd) * self.num_chips
        if self.package.mem_side:
            c_dcdc += (self.package.mem_side.tdp / self.package.mem_side.vdd) * self.package.num_mem_side * self.num_packages
        pass
        cost_dcdc = self.constants.DCDCCostPerAmp * c_dcdc
        cost_psu = self.tdp * self.constants.PSUCostPerW

        cost_per_package = self.package.cost
        cost_all_package = cost_per_package * self.num_packages

        cost_all_heatsinks = self.hs.cost * self.num_packages

        cost_all_fans = self.constants.FanCost * self.constants.SrvLanes
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
        
        return server_cost

