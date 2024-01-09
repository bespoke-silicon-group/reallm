import math
from dataclasses import dataclass
from typing import Optional
from .Constants import ChipConstants, ChipConstants7nm
from .Base import Base
from .IO import IO
from micro_arch_sim.design_memory import design_memory
from micro_arch_sim.design_memory_return_area import design_memory_return_area
from micro_arch_sim.magic_numbers import available_srams_7nm, vlsi_7nm

@dataclass
class Chip(Base):
   chip_id: int | str
   pkg2pkg_io: IO  # I/Os to the other package
   chip2chip_io: Optional[IO] = None # I/Os to the other chips in the same package

   # To define a chip, you should either give the perf, sram and bandwidth, or area and mac_ratio and operational intensity
   perf: Optional[float] = None # flops per sec
   sram: Optional[float] = None # byte
   sram_bw: Optional[int] = None # byte/s

   area: Optional[float] = None # mm2
   mac_ratio: Optional[float] = None # (0, 1.0), mac_area / (sram_area + mac_area)
   operational_intensity: Optional[float] = None # ops per sram read

   hbm_channels: int = 0 # number of HBM channels, each channel is 128 bit

   tech: str = '7nm'
   # MACs density mm2/Tera BF16 ops
   macs_density: float = 2.65 # data from whole chip implementations
   # IPU: 215mm2 tile logic for 250TOPS --> 0.86mm2/TOPS
   # TPUv4i: 100mm2 MXU for 138TOPS --> 0.72mm2/TOPS
   # macs_density: float = 1.0
   # Power Model, W/Tera BF16 ops
   w_per_tops: float = 1.3
   padring_width: float = 0.35
   core_area_ratio: float = 0.7
   other_area: float = 0.0 # chip area except SRAM and compute units
   acc_depth: int = int(1e100) # accumulation depth for systolic array, now assume it's infinite
   freq: float = 1e9 # Hz
   bytes_per_word: int = 2 # bfloat or fp16
   constants: ChipConstants = ChipConstants7nm

   valid: Optional[bool] = None
   invalid_reason: Optional[str] = None

   num_sa: Optional[int] = 1 # number of systolic arrays, now assume it's 1
   sa_width: Optional[int] = None # systolic array width, height is the same as width

   tdp: Optional[float] = None # Watts
   cost: Optional[float] = None # $
   power_density: Optional[float] = None # Watts/mm2

   io_area: Optional[float] = None
   sram_area: Optional[float] = None
   mac_area: Optional[float] = None

   die_yield: Optional[float] = 0.0
   dpw: Optional[int] = None # dies per wafer   

   tops: Optional[float] = None
   sram_mb: Optional[float] = None
   sram_bw_TB_per_sec: Optional[float] = None
   core_tdp: Optional[float] = None
   x: Optional[float] = None # mm
   y: Optional[float] = None # mm

   vdd: float = 0.8

   def update(self) -> None:
      if self.chip2chip_io:
         self.io_area = self.pkg2pkg_io.area + self.chip2chip_io.area
      else:
         self.io_area = self.pkg2pkg_io.area
      
      # HBM PHY and controller area
      self.other_area += self.hbm_channels * self.constants.hbm_phy_ctrl_area_per_channel

      if self.perf and self.sram:
         self.update_using_perf_sram()
      elif self.area and self.mac_ratio:
         self.update_using_area_ratio()
      else:
         self.valid = False
         self.invalid_reason = 'Wrong chip input configuration'
      
      total_fma = math.floor(self.perf / self.freq / 2) # 2 flops per MAC
      self.sa_width = math.floor(math.sqrt(total_fma / self.num_sa))
      
      if self.sram:
         if self.check_area():
            self.tdp = self._get_tdp()
            self.power_density = self.tdp / self.area
            if self.check_thermal():
               self.valid = True
               self.cost = self._get_cost()
               self.tops = self.perf / 1e12
               self.sram_bw_TB_per_sec = self.sram_bw / 1e12
               # assume the chip is a square
               self.x = math.sqrt(self.area)
               self.y = self.x

   def update_using_area_ratio(self) -> None:
      side = math.sqrt(self.area)
      core_side = side - self.padring_width
      core_area = core_side * core_side
      mac_sram_area = core_area * self.core_area_ratio - self.io_area - self.other_area
      if mac_sram_area < 0.1:
         # it generates some unreasonable designs when this is too small
         self.invalid_reason = f'Die size {self.area} mm2 too small! Not enough area for MACs and SRAM'
         self.valid = False
      else:
         self.sram_area = mac_sram_area * (1 - self.mac_ratio)
         self.mac_area = mac_sram_area - self.sram_area
         self.perf = self.mac_area / self.macs_density * 1e12
         # operational_intensity is the num of MACs per weight
         # 1 MAC = 2 FLOPS, 1 weight = 2 byte
         self.sram_bw = self.perf / 2 / self.operational_intensity * self.bytes_per_word
         # byte/s to bit/cycle, 1 byte is 8 bits
         sram_bw_bit_per_cycle = math.ceil(self.sram_bw * 8 / self.freq)
         sram_area_um2 = self.sram_area * 1e6
         self.sram_mb = design_memory(sram_area_um2, sram_bw_bit_per_cycle, vlsi_params=vlsi_7nm, available_srams=available_srams_7nm)
         if self.sram_mb:
            self.sram = self.sram_mb * 1e6
         else:
            self.invalid_reason = f'Can not find a valid SRAM design for {self.sram_area} mm2 and {self.sram_bw/1e12} TB/s'
            self.valid = False

   def update_using_perf_sram(self) -> None:
      self.sram_mb = self.sram / 1e6
      sram_bw_bit_per_cycle = math.ceil(self.sram_bw * 8 / self.freq)
      sram_area_um2 = design_memory_return_area(self.sram_mb, sram_bw_bit_per_cycle, vlsi_params=vlsi_7nm, available_srams=available_srams_7nm)
      self.sram_area = sram_area_um2 / 1e6
      self.mac_area = self.perf / 1e12 * self.macs_density
      mac_sram_area = self.sram_area + self.mac_area
      self.mac_ratio = self.mac_area / mac_sram_area
      core_area = (mac_sram_area + self.io_area + self.other_area) / self.core_area_ratio
      core_side = math.sqrt(core_area)
      side = core_side + self.padring_width
      self.area = side * side

   def check_area(self) -> bool:
      if self.area <= self.constants.max_die_area:
         return True
      else:
         self.valid = False
         self.invalid_reason = f'Chip area {self.area} is too large'
         return False

   def check_thermal(self) -> bool:
      if self.power_density <= self.constants.max_power_density:
         return True
      else:
         self.valid = False
         self.invalid_reason = f'Chip power density {self.power_density} W/mm2 is too high'
         return False
   
   def _get_tdp(self) -> float:
      tdp = self.perf / 1e12 * self.w_per_tops
      self.core_tdp = tdp
      # for reproduction, should remove this later
      tdp += self.pkg2pkg_io.tdp
      return tdp

   def _get_cost(self) -> float:
      self.die_yield = get_die_yield(self.area, self.constants.D0, self.constants.alpha)
      self.dpw = dies_per_wafer(self.area, self.constants.wafer_diameter, self.constants.wafer_dicing_gap)
      die_cost = (self.constants.wafer_cost * 1.0 / self.dpw) / self.die_yield
      testing_cost_per_die = die_cost * self.constants.testing_cost_overhead
  
      return die_cost + testing_cost_per_die
   
#################################################
# Yield and DPW calculation, from ASIC Cloud    #
#################################################

def get_die_yield(area: float, D0: float, alpha: float) -> float:
   if (area == 0):
      return 1.0
   else:
      # Negative binomial model for yield
      return math.pow((1.0 + (D0 * area / alpha)), alpha * -1)

# def dies_per_wafer(wafer_d) -> int:
def dies_per_wafer(die_area: float, wafer_diameter: float, wafer_dicing_gap: float) -> int:

   # given circle radious and square edge length, and distance between
   # diameter and center row of squares (longest one), calculates the 
   # number of squares that can be fited
   def max_fit(r, a, d) -> int:
      D1 = a / 2.0 + d
      # we don't need to calculate the longest line close to diameter
      D2 = (3.0 * a) / 2.0 - d
      R2 = r * r
      summ = 0
      while (D1 < r):
         l = 2.0 * math.sqrt(R2 - D1 * D1)
         summ += math.floor(l / a)
         D1 += a
      while (D2 < r):
         l = 2.0 * math.sqrt(R2 - D2 * D2)
         summ += math.floor(l / a)
         D2 += a
      return int(summ)
   
   # we want to find the optimal value by binary searching the different values
   # for center row distance to diameter. The optimal solution would have 
   # the longest row farthest from the center, having two long lines is the best if possible
   def max_square(r, a) -> int:
      start = 0.0
      end = a / 2.0
      max_start = max_fit(r, a, start)
      max_end = max_fit(r, a, end)
      # if we can fit two long rows, it's the optimal point
      if max_end >= max_start:
         return max_end	
      step = a / 8.0
      end = end / 2.0
      max_end = max_fit(r, a, end)
      while (max_start != max_end):
         if (max_start > max_end):
            end -= step
         else:
            start = end
            max_start = max_end
            end += step

         max_end = max_fit(r,a,end)
         step = step / 2.0
      return max_end

   # Converts the problem of dies per wafer to squares per circle
   # by adding the dicing gap to die width as square edge and calculate
   # radious instead of diameter
   return max_square(wafer_diameter / 2.0, math.sqrt(die_area) + wafer_dicing_gap)


