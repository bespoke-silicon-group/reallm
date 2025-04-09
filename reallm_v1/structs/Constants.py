from dataclasses import dataclass
from typing import Optional


@dataclass
class ChipConstants():
    # SRAM density mm2/MB
    sram_density: float
    # MACs density mm2/Tera BF16 ops
    macs_density: float
    # Power Model, W/Tera BF16 ops
    w_per_tops: float
    max_power_density: float # Watts/mm2
    padring_width: float # mm

    max_die_area: float # mm2
    D0: float # defects/mm2 = defects/cm2/100
    alpha: float # critical level
    wafer_diameter: int # in mm
    wafer_dicing_gap: float # in mm
    wafer_cost: float # in $
    testing_cost_overhead: float # testing add 0.01 cost per die

    hbm_phy_ctrl_area_per_channel: float # mm2
    mem_3d_tsv_ctrl_area_per_vault: Optional[float] = None # mm2
    mem_3d_ctrl_area_per_vault: Optional[float] = None # mm2
    mem_3d_area_per_tsv: Optional[float] = None # mm2
    mem_3d_test_area_per_vault: Optional[float] = None # mm2

@dataclass
class PackageConstants():
    # max die area for a package, mm2
    max_die_area: float
    max_power_density: float # Watts/mm2

    # Package cost model, based on Chiplet Actuary
    # Paper: https://arxiv.org/abs/2203.12268
    # Github: https://github.com/Yinxiao-Feng/DAC2022
    # Organic Substrate
    os_area_scale_factor: float
    os_cost_per_mm2: float # $/mm2
    # if there's more than one chips in a package, the cost will be multiplied by the factor
    # which depends on the package area: {area_threshold: cost_factor}
    os_layer_scale_factor: dict
    os_bonding_yield: float
    c4_bump_cost_per_mm2: float
    # Silicon Interposer
    si_area_scale_factor: float
    si_wafer_cost: float # $/wafer
    si_bonding_yield: float
    si_D0: float # defects/mm2 = defects/cm2/100
    si_alpha: float # critical level


@dataclass
class ServerConstants():
    PCBCost: float            # $/each (parts not included)x2 if there is DRAM
    PCBPartsCost: float       # (w/o DCDC, michael's sheet)
    DCDCCostPerAmp: float     # $/amp
    DCDCMaxCurrent: float     # Amps
    DCDCEfficiency: float     # 5% loss
    MaxDCDCCost: float        # $
    FanPower: float           # W/each
    FanCost: float
    APPPower: float           # W (Application Processor + Srv DRAM), on average
    APPCost: float            # $ (Application Processor + Srv DRAM)
    # PSU
    PSUCostPerW: float        # $/W
    PSUEfficiency: float      # 5% loss
    PSUOutputVoltage: float   # V
    MaxPSUCost: float         # $
    # Chassis
    ChassisCost: float        # $
    # Ethernet
    EthernetCost: float       # $
    SrvLife: float            # years
    SrvMaxPower: float        # W
    LaneAreaMin: float        # mm2
    LaneAreaMax: float        # mm2

@dataclass
class TCOConstants():
    ElectricityCost: float     # $/kWh
    InterestRate: float        # Annual Interest
    DCCapex: float             # $/W
    DCAmortPeriod: float       # amortizatio period in year
    DCOpex: float              # $/kW/month
    PUE: float
    SrvOpexRate: float         # % of server amortization
    SrvAvgPwr: float           # Server Average Power Relative to Peak,,

@dataclass
class EnergyConstants:
    # all in pJ
    # per byte
    sram_wgt: float # large SRAM, for weight
    sram_act: float # small SRAM, for activation, from https://gwern.net/doc/ai/scaling/hardware/2021-jouppi.pdf
    dram: float
    hbm2: float
    stacked_dram: float
    fma_fp16: float
