import math
from dataclasses import dataclass
from typing import Optional
from .Constants import ChipConstants, ChipConstants7nm
from .Base import Base
from .IO import IO
from micro_arch_sim.design_memory import design_memory
from micro_arch_sim.magic_numbers import available_srams_7nm, vlsi_7nm

@dataclass
class Chip(Base):
   chip_id: int
   pkg2pkg_io: IO  # I/Os to the other package
   chip2chip_io: Optional[IO] = None # I/Os to the other chips in the same package

   # To define a chip, you should either give the perf, sram and bandwidth, or area and mac_ratio and operational intensity
   perf: Optional[float] = None # #OPS
   sram: Optional[float] = None # byte
   sram_bw: Optional[int] = None # byte/s

   area: Optional[float] = None # mm2
   mac_ratio: Optional[float] = None # (0, 1.0), mac_area / (sram_area + mac_area)
   operational_intensity: Optional[float] = None # ops per sram read

   tech: str = '7nm'
   freq: float = 1e9 # Hz
   bytes_per_word: int = 2 # bfloat or fp16
   constants: ChipConstants = ChipConstants7nm
   padring_width: float = 0.35
   core_area_ratio: float = 0.7

   valid: Optional[bool] = None

   tdp: Optional[float] = None # Watts
   cost: Optional[float] = None # $
   power_density: Optional[float] = None # Watts/mm2

   other_area: float = 0.0 # chip area except SRAM and compute units
   io_area: Optional[float] = None
   sram_area: Optional[float] = None
   mac_area: Optional[float] = None

   die_yield: Optional[float] = 0.0
   dies_per_wafer: Optional[int] = None

   vdd: float = 0.8

   def update(self) -> None:
      if self.chip2chip_io:
         self.io_area = self.pkg2pkg_io.area + self.chip2chip_io.area
      else:
         self.io_area = self.pkg2pkg_io.area

      if self.perf and self.sram:
         self.update_using_perf_sram()
      elif self.area and self.mac_ratio:
         self.update_using_area_ratio()
      else:
         print('To define a Chip class, please either input perf and sram, or area and mac_ratio')
         self.valid = False
         
      if self.sram and not self.too_hot() and not self.too_big():
         self.valid = True
      else:
         self.valid = False

   def update_using_area_ratio(self) -> None:
      side = math.sqrt(self.area)
      core_side = side - self.padring_width
      core_area = core_side * core_side
      mac_sram_area = core_area * self.core_area_ratio - self.io_area - self.other_area
      if mac_sram_area < 0.1:
         # it generates some unreasonable designs when this is too small
         print('Die size too small! Not enough area for MACs and SRAM')
      else:
         self.sram_area = mac_sram_area * (1 - self.mac_ratio)
         self.mac_area = mac_sram_area - self.sram_area
         self.perf = self.mac_area / self.constants.macs_density * 1e12
         # operational_intensity is the num of MACs per weight
         # 1 MAC = 2 FLOPS, 1 weight = 2 byte
         self.sram_bw = self.perf / 2 / self.operational_intensity * self.bytes_per_word
         # byte/s to bit/cycle, 1 byte is 8 bits
         sram_bw_bit_per_cycle = math.ceil(self.sram_bw * 8 / self.freq)
         sram_area_um2 = self.sram_area * 1e6
         self.sram = design_memory(sram_area_um2, sram_bw_bit_per_cycle, vlsi_params=vlsi_7nm, available_srams=available_srams_7nm)

      self.tdp = self._get_tdp()
      self.cost = self._get_cost()
      self.power_density = self.tdp / self.area
      
   def update_using_perf_sram(self) -> None:
      # TODO: update the area and bw estimates using the micro_arch_sim
      self.sram_area = self.sram * self.constants.sram_density
      self.mac_area = self.perf * self.constants.macs_density
      mac_sram_area = self.sram_area + self.mac_area
      self.mac_ratio = self.mac_area / mac_sram_area
      core_area = (mac_sram_area + self.io_area + self.other_area) / self.core_area_ratio
      core_side = math.sqrt(core_area)
      side = core_side + self.padring_width
      self.area = side * side

      self.tdp = self._get_tdp()
      self.cost = self._get_cost()
      self.power_density = self.tdp / self.area

   def too_hot(self) -> bool:
      return (self.power_density > self.constants.max_power_density)

   def too_big(self) -> bool:
      return (self.area > self.constants.max_die_area)

   def _get_tdp(self) -> float:
      tdp = self.perf / 1e12 * self.constants.w_per_tops
      # for reproduction, should remove this later
      tdp += self.pkg2pkg_io.tdp
      return tdp

   def _get_die_yield(self) -> float:
      if (self.area == 0):
         return 1.0
      else:
         # Negative binomial model for yield
         return math.pow((1.0 + (self.constants.D0 * self.area / self.constants.alpha)), 
                         self.constants.alpha * -1)

   def _get_cost(self) -> float:
      self.die_yield = self._get_die_yield()
      self.dies_per_wafer = self._dies_per_wafer()
      die_cost = (self.constants.wafer_cost * 1.0 / self.dies_per_wafer) / self.die_yield
      testing_cost_per_die = die_cost * self.constants.testing_cost_overhead
  
      return die_cost + testing_cost_per_die
   
   ################################################
   # Calculate dies per wafer, from ASIC Cloud    #
   ################################################
   def _dies_per_wafer(self) -> int:

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
      return max_square(self.constants.wafer_diameter / 2.0,
                        math.sqrt(self.area) + self.constants.wafer_dicing_gap)


