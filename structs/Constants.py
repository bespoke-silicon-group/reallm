from dataclasses import dataclass

@dataclass
class ChipConstants():
    # SRAM density mm2/MB, data from real implementations
    sram_density: float = 0.45
    # MACs density mm2/Tera BF16 ops, data from real implementations
    macs_density: float = 2.65
    # Power Model, W/Tera BF16 ops
    w_per_tops: float = 1.3

    max_die_area: int = 900
    D0: float = 0.001 # defects/mm2 = defects/cm2/100
    alpha: float = 26.0
    wafer_diameter: int = 300 # in mm
    wafer_dicing_gap: float = 0.1 # in mm
    wafer_cost: int = 10000 # in $
    testing_cost_overhead: float = 0.01 # testing add 0.01 cost per die

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
    APPPower = 10.0           # W (Application Processor + DRAM)
    APPCost = 10.0            # $ (Application Processor + DRAM)
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
    SrvLanes = 8              # lanes per server

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



ServerConstantsCommon = ServerConstants()
TCOConstantsCommon = TCOConstants()
ChipConstants7nm = ChipConstants()