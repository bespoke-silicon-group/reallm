from dataclasses import dataclass, field

@dataclass
class ChipConstants():
    # SRAM density mm2/MB, data from real implementations
    # sram_density: float = 0.45
    # MACs density mm2/Tera BF16 ops
    # macs_density: float = 2.65 # data from whole chip implementations
    # IPU: 215mm2 tile logic for 250TOPS --> 0.86mm2/TOPS
    # TPUv4i: 100mm2 MXU for 138TOPS --> 0.72mm2/TOPS
    # macs_density: float = 1.0
    # Power Model, W/Tera BF16 ops
    # w_per_tops: float = 1.3

    hbm_phy_ctrl_area_per_channel: float = 1.04 # mm2
    mem_3d_tsv_ctrl_area_per_vault: float = 0.5 # mm2, double check this number
    mem_3d_ctrl_area_per_vault: float = 0.15 # 0.62 mm2 in 28nm (https://past.date-conference.com/proceedings-archive/2015/pdf/0054.pdf, Sec. IV) 
                                             # Since CPP of 28nm is 117nm,  of 7nm is 57nm, the area should be 0.62/(117/57)^2 = 0.15 mm2 
                                             # https://teamvlsi.com/2021/09/tsmc-7nm-16nm-and-28nm-technology-node-comparisons.html
    mem_3d_area_per_tsv: float = 0.0097 # mm2

    max_die_area: int = 900
    D0: float = 0.001 # defects/mm2 = defects/cm2/100
    alpha: float = 10.0 # critical level
    wafer_diameter: int = 300 # in mm
    wafer_dicing_gap: float = 0.1 # in mm
    wafer_cost: int = 10000 # in $
    testing_cost_overhead: float = 0.01 # testing add 0.01 cost per die

    max_power_density: float = 1.1 # Watts/mm2

@dataclass
class PackageConstants():
    max_die_area: float = 1400 # A100 is around 1400mm2
    # organic substrate
    os_area_scale_factor: float = 4.0
    os_cost_per_mm2: float = 0.005
    # if there's more than one chips in a package, the cost will be multiplied by the factor, 
    # which depends on the package area: {area_threshold: cost_factor}
    os_layer_scale_factor: dict = field(default_factory=lambda: {30*30: 2, 17*17: 1.75, 0: 1.0})
    os_bonding_yield: float = 0.99
    c4_bump_cost_per_mm2: float = 0.005
    # silicon interposer
    si_area_scale_factor: float = 1.1
    si_wafer_cost: float = 1937.0 # $/wafer for 55nm technology
    si_bonding_yield: float = 0.95
    si_D0: float = 0.0007 # defects/mm2 = defects/cm2/100
    si_alpha: float = 6.0 # critical level

    max_power_density: float = 1.0 # Watts/mm2

@dataclass
class ServerConstants():
    WaferCost = 12000.0
    PCBCost = 50.0            # $/each (parts not included)x2 if there is DRAM
    PCBPartsCost = 50.0       # (w/o DCDC, michael's sheet)
    DCDCCostPerAmp = 0.33     # $/amp
    DCDCMaxCurrent = 30.0     # Amps
    DCDCEfficiency = 0.95     # 5% loss
    FanPower = 7.4            # W/each
    FanCost = 15.0
    APPPower = 50.0           # W (Application Processor + Srv DRAM), on average
    APPCost = 200.0           # $ (Application Processor + Srv DRAM)
    # PSU
    PSUCostPerW = 0.13        # $/W
    PSUEfficiency = 0.95      # 5% loss
    PSUOutputVoltage = 12.0
    # Chassis
    ChassisCost = 30.0        # $
    # Ethernet
    # EthernetCost = 10.0     # $ for 1 GigE
    # EthernetCost = 100.0    # $ for 10 GigE
    EthernetCost = 450.0      # $ for 100 GigE
    SrvLife = 1.5             # years
    SrvMaxPower = 2000.0      # W
    LaneAreaMin = 100.0       # mm2
    LaneAreaMax = 6000.0      # mm2

@dataclass
class TCOConstants():
    ElectricityCost = 0.067    # $/kWh
    InterestRate = 0.08        # Annual Interest
    DCCapex = 10.0             # $/W
    DCAmortPeriod = 12.0       # amortizatio period in year
    DCOpex = 0.04              # $/kW/month
    PUE = 1.5
    SrvOpexRate = 0.05         # % of server amortization
    SrvAvgPwr = 1.0            # Server Average Power Relative to Peak,,

Joules = float
PicoJoules = float
@dataclass
class EnergyConstants:
    # per byte
    sram_wgt: PicoJoules = 1.25 # large SRAM, for weight
    sram_act: PicoJoules = 7.5/8 # small SRAM, for activation, from https://gwern.net/doc/ai/scaling/hardware/2021-jouppi.pdf
    dram: PicoJoules = 80.0
    hbm2: PicoJoules = 31.2
    stacked_dram: PicoJoules = 18.72

    # fma_fp16: PicoJoules = 2.75
    # fma_fp16: PicoJoules = 0.16 + 0.34 # from https://gwern.net/doc/ai/scaling/hardware/2021-jouppi.pdf
    fma_fp16: PicoJoules = 1.3 # from slides 17 of https://hc33.hotchips.org/assets/program/conference/day2/HC2021.Graphcore.SimonKnowles.v04.pdf

ChipConstants7nm = ChipConstants()
PackageConstantsCommon = PackageConstants()
ServerConstantsCommon = ServerConstants()
TCOConstantsCommon = TCOConstants()