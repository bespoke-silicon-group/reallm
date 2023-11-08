from structs.HardwareConfig import *

# Chip Configurations
chip_area_options = [20, 30, 40, 50, 60, 80, 100, 120, 140, 160, 180, 200, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680, 720, 760, 800]
mac_area_ratio_options = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.16, 0.20, 0.24, 0.28, 0.32, 0.40, 0.50]
operational_intensity_options = [1.0, 1.25, 1.5, 1.75, 2.0, 4.0, 8.0, 16.0, 32.0]
chip_io = IO(io_type='p2p', num=4, bandwidth_per_io=12.5e9, 
             area_per_io=0.3, tdp_per_io=0.125)
# Package Configurations
num_chips_options = [1]
# Server Configurations
packages_per_lane_options = [*range(3, 21)]
server_io = IO(io_type='s2s', num=2, bandwidth_per_io=12.5e9)

chip_area_config = ChipAreaConfig(area=chip_area_options, 
                                  mac_area_ratio=mac_area_ratio_options, 
                                  operational_intensity=operational_intensity_options)
chip_config = ChipConfig(core_config=chip_area_config, 
                         pkg_io=chip_io)
package_config = PackageConfig(num_chips=num_chips_options)
server_config = ServerConfig(packages_per_lane=packages_per_lane_options,
                             server_io=server_io)
