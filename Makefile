.DEFAULT_GOAL=build

BSG_CADENV_DIR 				 = $(abspath ./bsg_cadenv)
BSG_OUT_DIR            = $(abspath ./out)
SIMV  	   		         = $(abspath ./out/simv)
SIMV_DEBUG  	   		   = $(abspath ./out/simv-debug)

include $(BSG_CADENV_DIR)/cadenv.mk

# Repository setup
export BSG_DESIGNS_DIR = $(abspath .)
export BSG_DESIGNS_TARGET_DIR = $(BSG_DESIGNS_DIR)
export BASEJUMP_STL_DIR  = $(BSG_DESIGNS_DIR)/basejump_stl


export TESTING_BSG_DESIGNS_DIR        = $(BSG_OUT_DIR)/chiplet_simulator_designs
export TESTING_BSG_DESIGNS_TARGET_DIR = $(BSG_OUT_DIR)/chiplet_simulator_designs
export TESTING_BASEJUMP_STL_DIR 			= $(BSG_OUT_DIR)/basejump_stl


########################################
## VCS OPTIONS
########################################

# Common VCS Options (will be used most of the time by all corners)
VCS_OPTIONS := -full64
VCS_OPTIONS += -notice
VCS_OPTIONS += -V
VCS_OPTIONS += +v2k
VCS_OPTIONS += -sverilog -assert svaext
VCS_OPTIONS += +noportcoerce
VCS_OPTIONS += +vc
VCS_OPTIONS += +vcs+loopreport
VCS_OPTIONS += -timescale=1ps/1ps
VCS_OPTIONS += -diag timescale
VCS_OPTIONS += -Mdir=$(BSG_OUT_DIR)
VCS_OPTIONS += -top bsg_config bsg_config.v
# VCS_OPTIONS += -top test test.v
VCS_OPTIONS += +warn=all,noOPD,noTMR
VCS_OPTIONS += -l out/vcs.log
VCS_OPTIONS += +lint=all,noSVA-UA,noSVA-NSVU,noVCDE,noNS

VCS_OPTIONS += +notimingcheck
VCS_OPTIONS += +nospecify

########################################
## Chip and Testing Filelists and Liblists
########################################

BSG_TOP_SIM_MODULE = chiplets_array_tb
BSG_TOP_INSTANCE_PATH = chiplets_array_tb.inst

VCS_OPTIONS += +define+BSG_TOP_SIM_MODULE=$(BSG_TOP_SIM_MODULE)
VCS_OPTIONS += +define+BSG_TOP_INSTANCE_PATH=$(BSG_TOP_INSTANCE_PATH)

export BSG_CHIP_LIBRARY_NAME = bsg_chip
export BSG_CHIP_FILELIST = $(BSG_OUT_DIR)/$(BSG_CHIP_LIBRARY_NAME).filelist
export BSG_CHIP_LIBRARY = $(BSG_OUT_DIR)/$(BSG_CHIP_LIBRARY_NAME).library

VCS_OPTIONS += +define+BSG_CHIP_LIBRARY_NAME=$(BSG_CHIP_LIBRARY_NAME)
VCS_OPTIONS += -f $(BSG_CHIP_FILELIST)
VCS_OPTIONS += -libmap $(BSG_CHIP_LIBRARY)

export BSG_DESIGNS_TESTING_LIBRARY_NAME = bsg_design_testing
export BSG_DESIGNS_TESTING_FILELIST = $(BSG_OUT_DIR)/$(BSG_DESIGNS_TESTING_LIBRARY_NAME).filelist
export BSG_DESIGNS_TESTING_LIBRARY = $(BSG_OUT_DIR)/$(BSG_DESIGNS_TESTING_LIBRARY_NAME).library

VCS_OPTIONS += +define+BSG_DESIGNS_TESTING_LIBRARY_NAME=$(BSG_DESIGNS_TESTING_LIBRARY_NAME)
VCS_OPTIONS += -f $(BSG_DESIGNS_TESTING_FILELIST)
VCS_OPTIONS += -libmap $(BSG_DESIGNS_TESTING_LIBRARY)


########################################
## Run Targets
########################################

build: $(SIMV) $(SIMV_DEBUG)

simv: $(SIMV)

$(SIMV): filelist 
	$(VCS) $(VCS_OPTIONS) -o $@

$(SIMV_DEBUG): filelist 
	$(VCS) $(VCS_OPTIONS) -debug_pp +vcs+vcdpluson -o $@
	$(SIMV_DEBUG)

filelist: test_repo
	mkdir -p out/
	/usr/bin/tclsh bsg_config.tcl

test_repo:
	mkdir -p out/
	ln -nsf $(BASEJUMP_STL_DIR) $(TESTING_BASEJUMP_STL_DIR)
	# ln -nsf $(BSG_DESIGNS_DIR) $(TESTING_BSG_DESIGNS_DIR)
	ln -nsf $(BSG_DESIGNS_TARGET_DIR) $(TESTING_BSG_DESIGNS_TARGET_DIR)

dve:
	$(DVE) -full64 -vpd vcdplus.vpd

clean:
	rm -rf $(BSG_OUT_DIR)
	rm -rf DVEfiles
	rm -rf stack.info.*
	rm -f  vc_hdrs.h
	rm -f  vcdplus.vpd
	rm -f  inter.vpd
	rm -f  ucli.key
	rm -f  *.log

test:
	echo $(BSG_DESIGNS_TARGET_DIR)
