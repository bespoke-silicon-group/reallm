from dataclasses import dataclass


@dataclass
class SRAM:
    process: int  # tech node (nm)
    depth: int  # number of words
    width: int  # bits per word
    x: float  # um
    y: float  # um

    bits: int = None  # total number of bits
    area: float = None  # um^2
    density: float = None  # bits / um^2

    def __post_init__(self) -> None:
        self.bits = self.width * self.depth
        self.area = self.x * self.y
        self.density = self.bits / self.area


@dataclass
class MEMORY:
    base: SRAM  # base SRAM used to build the memory
    banks_x: int  # number of SRAMs in the logical x dim
    banks_y: int  # number of SRAMs in the logical y dim

    total_banks: int = None  # total number of tiles
    bits: int = None  # total number of bits
    area: float = None  # um^2
    bandwidth: int = None  # bits per cycle

    fp_x: int = None
    fp_y: int = None
    x: float = None
    y: float = None

    channels: int = None

    def __post_init__(self) -> None:
        self.total_banks = self.banks_x * self.banks_y
        self.bits = self.total_banks * self.base.bits
        self.area = self.total_banks * self.base.area
        self.bandwidth = self.banks_x * self.base.width

        self.fp_x = self.banks_x
        self.fp_y = self.banks_y
        self.x = self.fp_x * self.base.x
        self.y = self.fp_y * self.base.y

        while True:
            new_fp_y = self.fp_y * 2
            new_fp_x = self.fp_x // 2
            new_x = new_fp_x * self.base.x
            new_y = new_fp_y * self.base.y
            if abs(self.x - self.y) < abs(new_x - new_y):
                break
            self.fp_x, self.fp_y, self.x, self.y = new_fp_x, new_fp_y, new_x, new_y

        self.channels = self.fp_y // self.banks_y
