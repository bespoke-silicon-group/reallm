import os
import sys
import numpy as np
import pandas as pd

import utils
import dfplot
import DieCost
import CONSTANTS
from HTMLTable import df2html

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from matplotlib import rcParams
rcParams['text.usetex'] = True

#

mpw_area_min =      1.0 # mpw area per reticle
mpw_area_max = 100000.0 # limited by the foundry reticle size (eSilicon: 768)
wafers_min   =      1.0 # minimum wafer orders
wafers_max   = 100000.0 # maximum wafer orders (eSilicon: 100)
dpr_min      =      1.0 # minimum dies per reticle
cuts_max     =     10.0 # maximum saw cuts within MPW area
blocks_min   =      1.0 # miminum blocks per reticle

# Cost parameters
# nm   : technology node in nanometers
# nre  : full-reticle mask cost (with the nominal metal layers shown below)
# pkg  : packaging NRE cost
# bs   : foundry block size in mm^2
# rpw  : reticles per wafer
# wd   : wafer diameter
# fmwc : full-mask wafer cost
# flat : flat cost ($)
# cpb  : cost ($) per block
# cpc  : cost ($) per cut
# cpw  : cost ($) per wafer
# cpp  : cost ($) per package, not exactly characterized, can use 28.41 temporatily.
# mrk  : pyplot marker symbol: 'o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', 'D', 'd', '8'

# Note:
# For 28nm the 'nre' was 2159700 according to online queries but 4450000 was used by ISCA.

# Full-reticle cost data is collected from eSilicon using 1K units, First-available Metallization,
# 1mm x 1mm die area, Empty Options, No Packaging, and No Testing.
#
# The 'fmwc' is obtained from eSilicon and linearly scaled to make the 28nm cost consistent with the ISCA paper.
#       Metal  eSlcn
# 16nm    9    11121              = 17626
# 28nm    9     7575  --> * 1.585 = 12000 (be consistent with ISCA)
# 40nm    9     4835              =  7663
# 65nm    9     3277              =  5194
# 90nm    9     3223              =  5108
# 130nm   9     2955              =  4683
# 180nm   6      792              =  1255
# 250nm   5      723              =  1146

CostBook = {
  "7nm"   : {"nre": 7688289.0, "pkg": 105000.0, "nm":   7.0, "bs":  3.0, "rpw": 100.0, "wd": 300.0, "fmwc": 20000.0, "flat": 4400.0, "cpb": 251477.28, "cpc": 568.185, "cpw": 64920.45, "cpp": 0, "mrk": '8'},
  "10nm"  : {"nre": 7000000.0, "pkg": 105000.0, "nm":  10.0, "bs":  3.0, "rpw": 100.0, "wd": 300.0, "fmwc": 15000.0, "flat": 4400.0, "cpb": 201477.28, "cpc": 568.185, "cpw": 54920.45, "cpp": 0, "mrk": 'd'},
  "16nm"  : {"nre": 5688289.0, "pkg": 105000.0, "nm":  16.0, "bs":  4.0, "rpw": 100.0, "wd": 300.0, "fmwc": 11121.0, "flat": 4400.0, "cpb": 151477.28, "cpc": 568.185, "cpw": 44920.45, "cpp": 0, "mrk": 'o'},
  "22nm"  : {"nre": 3459700.0, "pkg": 105000.0, "nm":  22.0, "bs":  5.0, "rpw": 100.0, "wd": 300.0, "fmwc":  8000.0, "flat": 4400.0, "cpb": 123409.10, "cpc": 568.185, "cpw": 34920.45, "cpp": 0, "mrk": '+'},
  "28nm"  : {"nre": 2159700.0, "pkg": 105000.0, "nm":  28.0, "bs":  6.0, "rpw": 100.0, "wd": 300.0, "fmwc":  7575.0, "flat": 4400.0, "cpb": 103409.10, "cpc": 568.185, "cpw": 24920.45, "cpp": 0, "mrk": 'v'},
  "40nm"  : {"nre": 1246723.0, "pkg": 105000.0, "nm":  40.0, "bs":  9.0, "rpw": 100.0, "wd": 300.0, "fmwc":  4835.0, "flat": 4400.0, "cpb":  56015.91, "cpc": 568.185, "cpw": 10629.55, "cpp": 0, "mrk": '^'},
  "65nm"  : {"nre":  697027.0, "pkg": 105000.0, "nm":  65.0, "bs": 12.0, "rpw": 100.0, "wd": 300.0, "fmwc":  3277.0, "flat": 4400.0, "cpb":  48652.32, "cpc": 568.185, "cpw":  7954.53, "cpp": 0, "mrk": '<'},
  "90nm"  : {"nre":  556906.0, "pkg": 105000.0, "nm":  90.0, "bs": 16.0, "rpw": 100.0, "wd": 300.0, "fmwc":  3223.0, "flat": 4400.0, "cpb":  45736.32, "cpc": 568.185, "cpw":  5798.87, "cpp": 0, "mrk": '>'},
  "130nm" : {"nre":  293416.0, "pkg": 105000.0, "nm": 130.0, "bs": 25.0, "rpw": 100.0, "wd": 300.0, "fmwc":  2955.0, "flat": 4400.0, "cpb":  35739.75, "cpc": 568.185, "cpw":  4109.08, "cpp": 0, "mrk": 's'},
  "180nm" : {"nre":  105698.0, "pkg": 105000.0, "nm": 180.0, "bs": 25.0, "rpw":  40.0, "wd": 200.0, "fmwc":   792.0, "flat": 4400.0, "cpb":  17226.25, "cpc": 227.275, "cpw":  1223.84, "cpp": 0, "mrk": 'p'},
  "250nm" : {"nre":   63952.0, "pkg": 105000.0, "nm": 250.0, "bs": 25.0, "rpw":  40.0, "wd": 200.0, "fmwc":   723.0, "flat": 4400.0, "cpb":   9314.75, "cpc": 227.275, "cpw":  1139.79, "cpp": 0, "mrk": '*'},
}

# --- #

#
# Format value with comma-separated segments.
#
def comma_format(value):
  if (value == float("Infinity")):
    strval = "Infinity"
  elif (value == float("-Infinity")):
    strval = "-Infinity"
  else:
    strval = format(int(value), ",d")

  return strval

#
# Return the number of dies per wafer given die area and wafer diameter.
#
def dpw(die_area, wd):
  return DieCost.dies_per_wafer(die_area, wd)

#
# Return the number of saw cuts to get the given number of dies.
#
def saw_cuts(dies):
  sqrt_dies = np.floor(np.sqrt(dies))

  even_cuts = (sqrt_dies - 1) * 2 # even number of cuts give a perfect square number of dies
  even_cuts_dies = (even_cuts / 2 + 1) * (even_cuts / 2 + 1)

  odd_cuts = even_cuts + 1
  odd_cuts_dies = (even_cuts / 2 + 1) * (even_cuts / 2 + 2)

  if (dies == even_cuts_dies):
    cuts = even_cuts
  elif (dies <= odd_cuts_dies):
    cuts = odd_cuts
  else:
    cuts = odd_cuts + 1

  return cuts

#
# Create line plots with markers: the first arg is X data, and the second one is a list of Y data.
#
def plot_lines(xdata, l_ydata, l_yname, xlabel, ylabel, fillstyle=None, keyloc="upper right", marker=None, figsize=(10,8), fig=None, ax=None):
  dfplot.plot_config()

  if (figsize == None):
    figsize = (10, 8)

  if (ax == None):
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, squeeze=True, figsize=figsize)

  dfplot.axis_config(ax)

  # Autoscale visual area with graph content
  plt.autoscale(enable=True, axis="both", tight=None)

  l_markers = ['o', 'v', '8', '^', 's', '<', 'p', '*', '>', 'h', 'H', 'D', 'd', '1', '2', '3', '4', 'x', '+']

  if fillstyle != None:
    fs = fillstyle
  else:
    fs = 'full'

  l_handles = []
  for idx, ydata in enumerate(l_ydata):
    xpos = xdata[:len(ydata)] # making X-Y data have the same dimension
    ypos = ydata              # only considering len(xdata) >= len(ydata) case.
    if marker != None:
      mkr = marker
    else:
      mkr = l_markers[idx % len(l_markers)]
    handle, = ax.plot(xpos, ypos, marker=mkr,
                      markersize=7, fillstyle=fs, linestyle="", label=l_yname[idx])
    l_handles.append(handle)

  ax.legend(handles=l_handles, loc=keyloc, prop=dict(family='monospace', size=12))
  ax.grid(True)
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.grid(True, which="majorminor", linestyle="-", color='0.65')
  # Set minor tick locator for both axes.
  ax.xaxis.set_major_locator(matplotlib.ticker.AutoLocator())
  ax.yaxis.set_major_locator(matplotlib.ticker.AutoLocator())
  ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
  ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

  ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: utils.float_format(x, dtz=True,  dd=0)))
  ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: utils.float_format(x, dtz=False, dd=2)))

  ax.legend(numpoints=1, prop=dict(family='monospace', size=12))

  # Do show() and savefig() after post-function-call adjustments with (fig, ax) handles.
  # plt.show() # In notebook with inline matplotlib there is no need to call this function.
  # fig.savefig("figure.pdf", bbox_inches = 'tight')

  return ax

#
# Return cost in dollars given technology, die area, and the die volume.
#
# Cost function: Price = nre + fmwc * #wafers
def fms_cost(tech, die_area, dies, verbose):
  if not CostBook.has_key(tech):
    print "Error:", tech, "is not defined in the CostBook!"
    sys.exit(1)

  nre = CostBook[tech]["nre"]
  wd = CostBook[tech]["wd"]
  fmwc = CostBook[tech]["fmwc"]
  wafers = np.ceil(dies / dpw(die_area, wd))
  wcost = fmwc * wafers

  cost = nre + wcost

  if (verbose):
    print "The cost of manufacturing %d dies of %.2f mm2 in %s technology is $%s." % (dies, die_area, tech, comma_format(cost))
    print "NRE Cost:\t $%s"      % comma_format(nre)
    print "Wafer Cost:\t $%s"    % comma_format(wcost)
    print "Dies per Wafer:\t %s" % comma_format(dpw(die_area, wd))
    print "Wafer Count:\t %s"    % comma_format(wafers)

  return int(cost)

#
# Return cost in dollars given technology, mpw die area, and the number of wafers.
#
# Cost function: Price = flat + cpb * #blocks + cpw * (#wafers - 1) + cpc * #cuts * #wafers
#
# Parameters with '#' are variables.
# #blocks is per reticle, or per mpw die area.
#
# #block value is minimum 1.0 and it doesn't have to be integer.
#
def mpw_cost(tech, mpw_area, cuts, wafers, verbose):
  cost = 0.0
  if not CostBook.has_key(tech):
    print "Error:", tech, "is not defined in the CostBook!"
    sys.exit(1)

  if (verbose):
    if (mpw_area < mpw_area_min) or (mpw_area > mpw_area_max):
      print "Warning: MPW die area", mpw_area, "is clamped to range [%d..%d]." % (mpw_area_min, mpw_area_max)
    if (wafers < wafers_min) or (wafers > wafers_max):
      print "Warning: #Wafers", wafers, "is clamped to range [%d..%d]." % (wafers_min, wafers_max)

  # value clamps
  mpw_area = min(max(mpw_area_min, mpw_area), mpw_area_max)
  wafers = min(max(wafers_min, wafers), wafers_max)
  # The number of wafers should be integer.
  wafers = np.ceil(wafers)

  blocks = max(blocks_min, mpw_area / CostBook[tech]["bs"])

  fixed = CostBook[tech]["flat"]
  mcost = CostBook[tech]["cpb"] * blocks
  wcost = CostBook[tech]["cpw"] * (wafers - 1)
  ccost = CostBook[tech]["cpc"] * cuts * wafers

  # total = fixed + mpw mask cost + wafer cost + cut cost
  cost = fixed + mcost + wcost + ccost

  if (verbose):
    print "The cost of ordering %.2f mm2 MPW area with %d cuts and %d wafers in %s technology is $%s." % (mpw_area, cuts, wafers, tech, comma_format(cost))
    print "Fixed Cost:\t $%s"    % comma_format(fixed)
    print "MPW Mask Cost:\t $%s" % comma_format(mcost)
    print "Wafer Cost:\t $%s"    % comma_format(wcost)
    print "Cut Cost:\t $%s"      % comma_format(ccost)

  return int(cost), int(mcost), int(wcost)

#
# Optimizing cost given technology, die area, and the desired number of dies.
#
# Parameters with '#' are variables to tweak.
#
# dpr: dies per reticle
# Cost function: Price = cpb * (#dpr * die_area / bs) + cpw * (#wafers - 1) + flat
#
# Constraints:   #wafers * rpw * #dpr * die_area >= dies * die_area = total_area
#            --> #wafers * rpw * #dpr >= dies
#                #dpr and #wafers are integers
#                dpr_min <= #dpr <= dpr_max = floor(mpw_area_max / die_area)
#                wafers_min <= #wafers <= wafers_max
#
# Cost function derivative with respect to #dpr: Price' = cpb * die_area / bs - cpw * dies / (rpw * #dpr^2)
#
# Minimum cost is reached when Price' = 0 --> #dpr = sqrt(cpw * dies * bs / (rpw * cpb * die_area))
#
def opt_cost(tech, die_area, dies, verbose, plot):
  # Die area shouldn't be larger than mpw_area_max.
  if (die_area > mpw_area_max):
    if (verbose):
      print "\nWarning: In %s technology, the die area %.2f exceeds the maximum MPW area %d!" % (tech, die_area, mpw_area_max)
    return (0.0, 0.0, float("Infinity"))

  if not CostBook.has_key(tech):
    print "Error:", tech, "is not defined in the CostBook!"
    sys.exit(1)

  bs  = CostBook[tech]["bs"]
  rpw = CostBook[tech]["rpw"]
  cpb = CostBook[tech]["cpb"]
  cpw = CostBook[tech]["cpw"]
  cpp = CostBook[tech]["cpp"]

  total_area = die_area * dies

  if (total_area > mpw_area_max * rpw * wafers_max):
  # if (verbose):
  #   print "Warning: In %s technology, the total area %.2f exceeds the MPW capacity %d!" % (tech, total_area, mpw_area_max * rpw * wafers_max)

  # dies = np.floor(mpw_area_max * rpw * wafers_max / die_area)
  # total_area = die_area * dies
  # if (verbose):
  #   print "         The number of dies is clamped to %d, and thus total area is %d!" % (dies, total_area)
    if (verbose):
      print "\nWarning: In %s technology, the total area %.2f exceeds the maximum MPW capacity %d!" % (total_area, mpw_area_max * rpw * wafers_max)
    return (0.0, 0.0, float("Infinity"))

  if (verbose):
    print "\n[%s]: Search for the minimum fabrication cost for %d design dies with %.3f mm^2 die area." % (tech, dies, die_area)

  dpr_max = np.floor(mpw_area_max / die_area)

  # Solve optimal (#dpr, #wafers) in floating-point values.
  dpr_opt = np.sqrt(cpw * dies * bs / (rpw * cpb * die_area))
  wafers_opt = dies / (rpw * dpr_opt)
  if (verbose):
    print "Floating-point (#dpr, #wafers) solution for minimal cost: (%.2f, %.2f)" % (dpr_opt, wafers_opt)

  # Round-up and clamp values (they can't both exceed their maximum now due to the total_area clamping above).
  dpr_opt = min(dpr_max, np.ceil(dpr_opt))
  wafers_opt = min(wafers_max, np.ceil(dies / (rpw * dpr_opt)))

  # The optimal cost can't be more than this value, and so the integer search space can be reduced.
  # Search Constraints: wafers_min <= #wafers <= wafers_opt
  #                     dpr_min <= #dpr <= dpr_opt
  #                     #wafers * rpw * #dpr >= dies
  cuts_opt = saw_cuts(dpr_opt)
  cost_opt = mpw_cost(tech, dpr_opt * die_area, cuts_opt, wafers_opt, False)[0] + int(dies * cpp)

  # Search for the optimal #dpr and #wafers in integer space.
  for dpr in range(int(dpr_min), int(dpr_opt)): # 1, 2, ..., (dpr_opt - 1)
    cuts = saw_cuts(dpr)
    wafers = np.ceil(dies / (rpw * dpr))
    if (wafers <= wafers_max):
      cost = mpw_cost(tech, dpr * die_area, cuts, wafers, False)[0] + int(dies * cpp)
      if (cost < cost_opt):
        dpr_opt = dpr
        cuts_opt = cuts
        wafers_opt = wafers
        cost_opt = cost
  for wafers in range(int(wafers_min), int(wafers_opt)): # 1, 2, ..., (wafers_opt - 1)
    dpr = np.ceil(dies / (rpw * wafers))
    cuts = saw_cuts(dpr)
    if (dpr <= dpr_max):
      cost = mpw_cost(tech, dpr * die_area, cuts, wafers, False)[0] + int(dies * cpp)
      if (cost < cost_opt):
        dpr_opt = dpr
        cuts_opt = cuts
        wafers_opt = wafers
        cost_opt = cost

  # Adjust mpw die area and corresponding number of blocks.
  # No need to clamp mpw_area because it should be in range.
  mpw_area_opt = dpr_opt * die_area
  blocks_opt = max(blocks_min, mpw_area_opt / bs)

  if (verbose):
    print "Dies per Reticle: \t", int(dpr_opt)
    print "Saw Cuts: \t\t", int(cuts_opt)
    print "MPW Die Area: \t\t%.3f mm^2, or %.2f blocks (block size is %d mm^2)" % (mpw_area_opt, blocks_opt, bs)
    print "Reticles per Wafer: \t", int(rpw), "(foundry fixed)"
    print "Number of Wafers: \t", int(wafers_opt)
    print "Total Reticle Dies: \t", int(wafers_opt * rpw), "(before slicing)"
    print "Total Design Dies: \t", int(wafers_opt * rpw * dpr_opt)
    print "Price: \t\t\t$%s" % comma_format(cost_opt)

  if (plot):
    l_dpr = []
    l_mpw_area = []
    l_wafers = []
    for dpr in range(int(max(dpr_min, dpr_opt - 10)), int(max(dpr_min, dpr_opt - 10)) + 20): # 20 dpr points for graphing
      mpw_area = dpr * die_area
      wafers = np.ceil(dies / (dpr * rpw))
      if (mpw_area == min(max(mpw_area_min, mpw_area), mpw_area_max)) and (wafers == min(max(wafers_min, wafers), wafers_max)):
        l_dpr.append(dpr)
        l_mpw_area.append(mpw_area)
        l_wafers.append(wafers)
    l_cuts = map(lambda x: saw_cuts(x), l_dpr)
    l_pkg_cost = map(lambda x, y: x * y * cpp, l_dpr, l_wafers)
    l_cost = map(lambda x, y, z, p: mpw_cost(tech, x, y, z, False)[0] + int(p), l_mpw_area, l_cuts, l_wafers, l_pkg_cost)
    l_mcost = map(lambda x, y, z, p: mpw_cost(tech, x, y, z, False)[1] + int(p), l_mpw_area, l_cuts, l_wafers, l_pkg_cost)
    l_wcost = map(lambda x, y, z, p: mpw_cost(tech, x, y, z, False)[2] + int(p), l_mpw_area, l_cuts, l_wafers, l_pkg_cost)

    ax = plot_lines(l_dpr, [l_mcost, l_wcost, l_cost], ["Mask Cost", "Wafer Cost", "Total Cost"], xlabel="RCAs per Die", ylabel="Manufacturing Cost (\$)")
    fig = ax.get_figure()
    imf = os.environ.get('IMAGE_FORMAT')
    fig.savefig(tech + "_MPWOpt."+imf, bbox_inches = 'tight')

  return (dpr_opt, wafers_opt, cost_opt)

#
# Optimizing cost given technology and the total silicon area to be fabricated.
#
def opt_cost_given_total_area(tech, total_area, verbose, plot):
  if CostBook.has_key(tech):
    bs  = CostBook[tech]["bs"]
    rpw = CostBook[tech]["rpw"]
    if (total_area > mpw_area_max * rpw * wafers_max):
      if (verbose):
        print "\nWarning: In %s technology, the total area %.2f exceeds the maximum MPW capacity %d!" % (total_area, mpw_area_max * rpw * wafers_max)
      return (0.0, 0.0, float("Infinity"))

    die_area = mpw_area_min
    l_die_area = []
    l_dies = []
    l_cost =[]
    (die_area_opt, dies_opt, dpr_opt, wafers_opt, cost_opt) = (die_area, 0.0, 0.0, 0.0, float("Infinity"))

    while True:
      dies = np.ceil(total_area / die_area)
      (dpr, wafers, cost) = opt_cost(tech, die_area, dies, False, False)
      l_die_area.append(die_area)
      l_dies.append(dies)
      l_cost.append(cost)
      if ( (cost < cost_opt) or ((cost == cost_opt) and (die_area > die_area_opt)) ):
        (die_area_opt, dies_opt, dpr_opt, wafers_opt, cost_opt) = (die_area, dies, dpr, wafers, cost)
      die_area *= 2
      if (die_area > min(mpw_area_max, total_area)):
        break

    # Adjust mpw die area and corresponding number of blocks.
    # No need to clamp mpw_area because it should be in range.
    mpw_area_opt = dpr_opt * die_area_opt
    blocks_opt = max(blocks_min, mpw_area_opt / bs)

    if (verbose):
      print "\n[%s]: The optimal die area to fabricate %d mm^2 silicon in total is %.2f." % (tech, total_area, die_area_opt)
      print "Dies per Reticle: \t", int(dpr_opt)
      print "MPW Die Area: \t\t%.3f mm^2, or %.2f blocks (block size is %d mm^2)" % (mpw_area_opt, blocks_opt, bs)
      print "Reticles per Wafer: \t", int(rpw), "(foundry fixed)"
      print "Number of Wafers: \t", int(wafers_opt)
      print "Total Reticle Dies: \t", int(wafers_opt * rpw), "(before slicing)"
      print "Total Design Dies: \t", int(wafers_opt * rpw * dpr_opt)
      print "Total Area (adjusted): \t", int(die_area_opt * dpr_opt * rpw * wafers_opt)
      print "Price: \t\t\t$%s" % comma_format(cost_opt)

    if (plot):
      handle, = plt.semilogx(l_die_area, l_cost, marker=CostBook[tech]["mrk"], label=tech, basex=2)
      plt.legend(handles=[handle], loc=0, prop=dict(family='monospace', size=12))
      plt.grid(True)
      plt.xlabel("Die Area (mm^2)")
      plt.ylabel("Cost (\$)")
      plt.show()
  else:
    print "Error:", tech, "is not defined in the CostBook!"
    sys.exit(1)

  return (die_area_opt, dies_opt, dpr_opt, wafers_opt, cost_opt)

def opt_cost_test():
  opt_cost("16nm", 1.024, 50, True, True)
  opt_cost("16nm", 1.024, 500, True, True)
  opt_cost("16nm", 1.024, 1000, True, True)
  opt_cost("28nm", 2, 500, True, True)
  opt_cost("28nm", 500, 100, True, True)
  opt_cost("28nm", 10, 5000, True, True)
  opt_cost("28nm", 4, 355, True, True)
  opt_cost("28nm", 3, 2000, True, True)
  opt_cost("65nm", 10, 2048, True, True)
  opt_cost("90nm", 10, 5000, True, True)
  opt_cost("90nm", 100, 1955, True, True)

def volume_cost(tech, tech_norm, die_area, l_dies, invr, mpw_opt):

  valid_invr = ["trans", "perf"]
  if not (invr in valid_invr):
    print "Error: Invariant '%s' is not a valid input!" % invr
    sys.exit(1)

# f = open("volume_cost.txt", 'w')
# map(lambda x: f.write('\t' + str(x)), l_dies)

  nm  = CostBook[tech]["nm"]
  nre = CostBook[tech]["nre"]
  rpw = CostBook[tech]["rpw"]
  cpp = CostBook[tech]["cpp"]

  if not (tech_norm == None):
    nm_norm = CostBook[tech_norm]["nm"]
    sf = nm / nm_norm # Denard's scaling factor
  else:
    sf = 1.0

  if (invr == "perf"): # same performance (throughput)
    area_sf = sf*sf*sf
  else: # (invr == "trans"): # same transistor counts
    area_sf = sf*sf

# print die_area*area_sf

  if (mpw_opt):
    l_mpw_cost = map(lambda x: opt_cost(tech, die_area*area_sf, x, False, False)[2], l_dies)
  else:
    l_mpw_cost = map(lambda x: mpw_cost(tech, die_area*area_sf, 0, np.ceil(x/rpw), False)[0] + int(x * cpp), l_dies)

  l_fms_cost = map(lambda x: fms_cost(tech, die_area*area_sf, x, False), l_dies)

  l_cost = map(lambda x, y: min(x, y), l_mpw_cost, l_fms_cost)

  # l_cpd = map(lambda x, y: x/y, l_cost, l_dies)

  if nre in l_cost:
    breakeven = l_dies[l_cost.index(nre)] # This is an approx. break even volume, not exactly accurate.
  else:
    breakeven = 9999999

# f.write("\n" + tech)
# map(lambda x: f.write('\t' + str(x)), l_cost)

# f.close()

  return l_cost

#
# Given technology and accelerator unit area, explore cost values as a function of the replicated #units per die (1 die per reticle).
# Or, given technology and die area, explore cost values as a function of the replicated dies per reticle.
#
# Mode:
#   die: area = die area
#   rca: area = replicable compute accelerator (RCA) area
#
def replica_cost(tech, mode, area, l_replica, volume):
  valid_modes = ["rca", "die"]
  if not (mode in valid_modes):
    print "Error: Mode '%s' is not a valid input!" % mode
    sys.exit(1)

  if not CostBook.has_key(tech):
    print "Error:", tech, "is not defined in the CostBook!"
    sys.exit(1)

  rpw = CostBook[tech]["rpw"]
  cpp = CostBook[tech]["cpp"]

# f = open(mode + "_replica_cost_" + tech + ".txt", 'w')
# f.write("Copies per Reticle")
# map(lambda x: f.write('\t' + str(x)), l_replica)

  l_wafers = map(lambda x: np.ceil(volume / (x * rpw)), l_replica)
  if 1 in l_wafers: # When #wafers is 1, there is no need to increase copies per reticle
    l_replica_truc = l_replica[0:l_wafers.index(1)+1]
    l_wafers_truc = l_wafers[0:l_wafers.index(1)+1]
  else:
    l_replica_truc = l_replica
    l_wafers_truc = l_wafers

  if (mode == "die"):
  # f.write("\n" + comma_format(volume) + " Dies")
    l_cost = map(lambda x, y: mpw_cost(tech, area * x, saw_cuts(x), y, False)[0] + int(x * cpp), l_replica_truc, l_wafers_truc)
  elif (mode == "rca"):
  # f.write("\n" + comma_format(volume) + " RCAs")
    l_cost = map(lambda x, y: mpw_cost(tech, area * x, 0, y, False)[0] + int(cpp), l_replica_truc, l_wafers_truc)
# map(lambda x: f.write('\t' + str(x)), l_cost)

# f.close()

  return l_cost

def const_df():
  df = pd.DataFrame(CostBook, columns=["16nm", "28nm", "40nm", "65nm", "90nm", "130nm", "180nm", "250nm"],
                              index=["nre", "nm", "bs", "rpw", "wd", "fmwc", "flat", "cpb", "cpc", "cpw", "cpp"]).transpose()

  df.columns = ["Full Mask NRE ($)", "Feature Size (nm)", "MPW Block Size (mm2)", "Reticles per Wafer",
                "Wafer Diameter (mm)", "Full-Mask Wafer Cost ($)", "MPW Flat Cost ($)", "MPW Cost per Block ($)",
                " MPW Cost per Cut ($)", "MPW Cost per Wafer ($)", "Cost per Package ($)"]
  return df

def fr_vs_mpw():
  techs = ["28nm", "40nm", "65nm", "90nm", "130nm", "180nm", "250nm"]
  tech_norm = '250nm'

  df = pd.DataFrame(index=techs)

  df['nm'] = [CostBook[tech]['nm'] for tech in techs]
  df['vdd'] = [CONSTANTS.TechData.at[tech, 'CoreVdd'] for tech in techs]
  df['fr_nre'] = [CostBook[tech]['nre'] for tech in techs]
  df['fr_wc'] = [CostBook[tech]['fmwc'] for tech in techs]
  df['wd'] = [CostBook[tech]['wd'] for tech in techs]
  df['fr_wa'] = np.pi * (df.wd/2.0) ** 2
  df['fr_cost_mm2'] = df.fr_wc / df.fr_wa

  df['mpw_bs'] = [CostBook[tech]['bs'] for tech in techs]
  df['mpw_cpb'] = [CostBook[tech]['cpb'] for tech in techs]
  df['mpw_wc'] = [CostBook[tech]['cpw'] for tech in techs]
  df['mpw_rpw'] = [CostBook[tech]['rpw'] for tech in techs]
  df['mpw_wa'] = df.mpw_bs * df.mpw_rpw
  df['mpw_cost_mm2'] = df.mpw_wc / df.mpw_wa

  df['norm_joules_per_op'] = (df.nm/df.nm[tech_norm])*(df.vdd/df.vdd[tech_norm])**2
  df['density'] = (df.nm[tech_norm]/df.nm)**2
  df['frequency'] = (df.nm[tech_norm]/df.nm)
  df['perf_mm2'] = df.density * df.frequency


  df['fr_cost_per_perf']      = df.fr_cost_mm2 / df.perf_mm2
  df['mpw_1b_cost_per_perf']  = df.mpw_cost_mm2 / df.perf_mm2
  df['mpw_2b_cost_per_perf']  = df.mpw_1b_cost_per_perf / 2.0
  df['mpw_5b_cost_per_perf']  = df.mpw_1b_cost_per_perf / 5.0
  df['mpw_10b_cost_per_perf'] = df.mpw_1b_cost_per_perf / 10.0
  df['mpw_13b_cost_per_perf'] = df.mpw_1b_cost_per_perf / 13.0

  df['mpw_2b_cost'] = df.mpw_cpb * 2.0
  df['mpw_5b_cost'] = df.mpw_cpb * 5.0
  df['mpw_10b_cost'] = df.mpw_cpb * 10.0
  df['mpw_13b_cost'] = df.mpw_cpb * 13.0

  dfx = df[['fr_cost_per_perf', 'mpw_1b_cost_per_perf', 'mpw_2b_cost_per_perf', 'mpw_5b_cost_per_perf', 'mpw_10b_cost_per_perf', 'mpw_13b_cost_per_perf']].T
  dfy = df[['fr_nre','mpw_cpb','mpw_2b_cost','mpw_5b_cost','mpw_10b_cost','mpw_13b_cost']].T

  xs = dfx.iloc[:, 0].tolist()
  ys = dfy.iloc[:, 0].tolist()
  frlist = [(xs[0], ys[0])]
  ax = plot_lines(xs, [ys], [dfx.columns[0]], xlabel="Cost (\$) per OP/s", ylabel="Mask Cost (\$)", keyloc="best")
  markers = ['o', 'v', '8', '^', 's', '<', 'p', '*', '>', 'h', 'H', 'D', 'd', '1', '2', '3', '4', 'x', '+']
  for i in range(1, len(techs)):
    xs = dfx.iloc[:, i].tolist()
    ys = dfy.iloc[:, i].tolist()
    frlist.append( (xs[0], ys[0]) )
    ax = plot_lines(xs, [ys], [dfx.columns[i]], xlabel="Cost (\$) per OP/s", ylabel="Mask Cost (\$)", keyloc="best", marker=markers[i], ax=ax)

# for fr in frlist:
#   ax.annotate('fr', xy=(fr[0], fr[1]), xycoords='data', size=10, bbox=dict(boxstyle='circle', fc='none'))

  ax.set_xscale("log", nonposx='clip')
  ax.set_yscale("log", nonposy='clip')
  ax.set_xlim(1e-4, 2)
  ax.set_ylim(5e3, 10e6)
  ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: utils.float_format(x, dd=5)))
  ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: utils.float_format(x, dd=1)))
  handles, labels = ax.get_legend_handles_labels()

  labelwidth = max(map(len, labels)) # for equalizing label text width
  labels = map(lambda label: '{:>{w}}'.format(label, w=labelwidth), labels)

  ax.legend(handles, labels, numpoints=1, prop=dict(family='monospace', size=12))
  fig = ax.get_figure()
  imf = os.environ.get('IMAGE_FORMAT')
  fig.savefig('CostPerOP_NRE.'+imf, bbox_inches = 'tight')

  ax = dfplot.line(df, x='norm_joules_per_op', y=  'fr_nre',     loglog=True, linestyle='', marker='o', alpha=0.8, keyloc='best', xlabel='Normalized Energy per OP', ylabel='Mask Cost (\$)')
  ax = dfplot.line(df, x='norm_joules_per_op', y= 'mpw_cpb',     loglog=True, linestyle='', marker='^', alpha=0.8, keyloc='best', xlabel='Normalized Energy per OP', ylabel='Mask Cost (\$)', ax=ax)
  ax = dfplot.line(df, x='norm_joules_per_op', y= 'mpw_2b_cost', loglog=True, linestyle='', marker='s', alpha=0.8, keyloc='best', xlabel='Normalized Energy per OP', ylabel='Mask Cost (\$)', ax=ax)
  ax = dfplot.line(df, x='norm_joules_per_op', y= 'mpw_5b_cost', loglog=True, linestyle='', marker='p', alpha=0.8, keyloc='best', xlabel='Normalized Energy per OP', ylabel='Mask Cost (\$)', ax=ax)
  ax = dfplot.line(df, x='norm_joules_per_op', y='mpw_10b_cost', loglog=True, linestyle='', marker='d', alpha=0.8, keyloc='best', xlabel='Normalized Energy per OP', ylabel='Mask Cost (\$)', ax=ax)
  ax = dfplot.line(df, x='norm_joules_per_op', y='mpw_13b_cost', loglog=True, linestyle='', marker='*', alpha=0.8, keyloc='best', xlabel='Normalized Energy per OP', ylabel='Mask Cost (\$)', ax=ax)
  handles, labels = ax.get_legend_handles_labels()
  labels = ['Full Reticle',
            'MPW  1 Block',
            'MPW  2 Blocks',
            'MPW  5 Blocks',
            'MPW 10 Blocks',
            'MPW 13 Blocks']
  ax.legend(handles, labels, numpoints=1, prop=dict(family='monospace', size=12))
  ax.set_xscale("log", nonposx='clip')
  ax.set_yscale("log", nonposy='clip')
  ax.set_xlim(1e-2, 2)
  ax.set_ylim(5e3, 10e6)
  ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: utils.float_format(x, dd=5)))
  ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: utils.float_format(x, dd=1)))
  fig = ax.get_figure()
  imf = os.environ.get('IMAGE_FORMAT')
  fig.savefig('EnergyPerOP_NRE.'+imf, bbox_inches = 'tight')

  return df

def fabcost(trans, tech, dies=1e3, mpw=False):
  area = trans*CONSTANTS.TechData.at[tech, 'TrSize']*1e-6 # um2 -> mm2
  if not mpw:
    if area > 26*33: # max full reticle die size
      cost = float('nan')
    else:
      diecost, _ = DieCost.die_cost_calc(area, 1.0, tech, 0, False)
      cost = CostBook[tech]['nre'] + diecost * dies
  else:
    if area > 768: # max mpw die size
      cost = float('nan')
    else:
      if area <= CostBook[tech]['bs']:
        cost = CostBook[tech]['cpb']
      else:
        diecost, _ = DieCost.die_cost_calc(area, 1.0, tech, 0, True)
        cost = CostBook[tech]['cpb'] * (area/CostBook[tech]['bs']) + diecost * dies

  return cost

# only consider mask cost here because our nre model depends on specific applications
# to get the ip, labor, and cad costs.
def transistors_nre():
  techs = ["28nm", "40nm", "65nm", "90nm", "130nm", "180nm", "250nm"]
  trs = np.logspace(6, 12, base=10, num=50) # transistor counts
  frdf = pd.DataFrame(columns=['trs','nre','tech'])
  mpwdf = pd.DataFrame(columns=['trs','nre','tech'])
  for tech in techs:
    for tr in trs:
      row = pd.Series([tr, fabcost(tr, tech, dies=1e3, mpw=False)/tr, ' FR '+tech], frdf.columns)
      frdf = frdf.append(row, ignore_index=True)
      # --- #
      row = pd.Series([tr, fabcost(tr, tech, dies=1e3, mpw=True)/tr, 'MPW '+tech], mpwdf.columns)
      mpwdf = mpwdf.append(row, ignore_index=True)

  ax = dfplot.line(frdf, x='trs', y='nre', key='tech', keyloc='best', xlabel='Transistors', loglog=True, ylabel='Mask Cost (\$)', fillstyle='none', linestyle='-')
  ax.set_prop_cycle(None)
  ax = dfplot.line(mpwdf, x='trs', y='nre', key='tech', keyloc='best', xlabel='Transistors', loglog=True, ylabel='Mask Cost (\$)', markersize=5, ax=ax)
  fig = ax.get_figure()
  imf = os.environ.get('IMAGE_FORMAT')
  fig.savefig('Transistors_NRE.'+imf, bbox_inches = 'tight')

def mpw_plots():

  mpw_cost("40nm", 10, 0, 8, True)
  opt_cost("40nm", 3, 100000, True, True)

  # --- #

  tech = "40nm"
  mode = "rca"
  area = 1.35 # mm2
  l_replica = range(1, 11) + range(15, 101, 5) # copies per reticle

  plotYdata = []
  plotYname = []
  for volume in [100, 1000, 10000, 50000, 100000]:
      l_cost = replica_cost(tech, mode, area, l_replica, volume)
      plotYdata.append(l_cost)
      plotYname.append("Total " + str(volume) + " RCAs")

  ax = plot_lines(l_replica, plotYdata, plotYname, xlabel="RCAs per Reticle", ylabel="Cost (\$)", keyloc="lower right")
  fig = ax.get_figure()
  ax.set_yscale('log')

  imf = os.environ.get('IMAGE_FORMAT')
  fig.savefig("RCA_Replica."+imf, bbox_inches = 'tight')

  # --- #

  tech = "40nm"
  die_area = 6 # mm2
  l_dies = range(100, 100000, 2000)

  plotYdata = []
  plotYname = []

  l_cost = volume_cost(tech, None, die_area, l_dies, "trans", False)
  plotYdata.append(l_cost)
  plotYname.append("One Die per Reticle")

  l_cost = volume_cost(tech, None, die_area, l_dies, "trans", True)
  plotYdata.append(l_cost)
  plotYname.append("Optimal Dies per Reticle")

  ax = plot_lines(l_dies, plotYdata, plotYname, xlabel="Die Volume", ylabel="Cost (\$)", keyloc="lower right")
  ax.legend(loc='lower right', prop=dict(family='monospace', size=12))
  fig = ax.get_figure()
  imf = os.environ.get('IMAGE_FORMAT')
  fig.savefig("Volume_Cost."+imf, bbox_inches = 'tight')

  # --- #

  l_tech = ["28nm", "40nm", "65nm", "180nm"]

  die_area = 1.0 # mm2 in 28nm

  l_dies = range(100, 50000, 800) # range(100, 100000, 3000)

  fig, ax = plt.subplots(sharex=False, sharey=False, squeeze=True, figsize=(20, 6))

  invrs = ["trans", "perf"] # invariable options
  for idx, invr in enumerate(invrs):

      plotYdata = []
      plotYname = []
      for tech in l_tech:
          l_cost = volume_cost(tech, l_tech[0], die_area, l_dies, invr, False)
          plotYdata.append(l_cost)
          plotYname.append(tech)

      ax = plt.subplot(1, len(invrs), idx+1) # axis handle of subplots
      plot_lines(l_dies, plotYdata, plotYname, xlabel="Die Volume", ylabel="Cost (\$)", keyloc="upper left", fig=fig, ax=ax)
      fig = ax.get_figure()
      ax.legend(numpoints=1, loc='upper left', prop=dict(family='monospace', size=12))

  imf = os.environ.get('IMAGE_FORMAT')
  fig.savefig("Tech_Scaling."+imf, bbox_inches = 'tight')

def fr_plots():
  imf = os.environ.get('IMAGE_FORMAT')
 #techs = ["28nm", "40nm", "65nm", "90nm", "130nm", "180nm", "250nm"]
  techs = ["250nm", "180nm", "130nm", "90nm", "65nm", "40nm", "28nm", "16nm"]
  tech_norm = '250nm'

  df = pd.DataFrame(index=techs)

  df['tech'] = [tech for tech in techs]
  df['nm'] = [CostBook[tech]['nm'] for tech in techs]
  df['vdd'] = [CONSTANTS.TechData.at[tech, 'CoreVdd'] for tech in techs]
  df['tr_size'] = [CONSTANTS.TechData.at[tech, 'TrSize']*1e-6 for tech in techs] # mm2
  df['wc'] = [CostBook[tech]['fmwc'] for tech in techs]
  df['wd'] = [CostBook[tech]['wd'] for tech in techs]
  df['wa'] = np.pi * (df.wd/2.0) ** 2

  df['nm_norm'] = df.nm[tech_norm] / df.nm
  xs = df.nm_norm.tolist()

  df['die_trs'] = 26*33 / df.tr_size # 26 33 is the max reticle size from eSilicon.
  # df.die_trs = df.die_trs / df.die_trs[tech_norm] # relative available transistors
  ax = dfplot.line(df, x='nm_norm', y='die_trs', key='tech', xlabel='', ylabel='\#Transistors per Die', ybold=True, xbold=True,
                   keyloc="upper left", legend='reverse', loglog=True, figsize=(10,4.5))
  lastx = xs[0]
  lasty = df.die_trs[0]
  for i, x in enumerate(xs[1:]):
    y = df.die_trs[i+1]
    ax.annotate('', xy=(x, lasty), xycoords='data', xytext=(x, y), textcoords='data',
                arrowprops=dict(arrowstyle='<->', lw=2, connectionstyle='arc', shrinkA=0, shrinkB=0, color='royalblue'))
    ax.annotate('{:0,.2f}'.format(y/lasty), xy=(x, lasty), xycoords='data',
                xytext=(0, -13), textcoords='offset points',
                rotation=0, size=16, va='center', ha='center',
                bbox=dict(boxstyle='round', fc='white'))
    lastx, lasty = x, y
  ax.set_xlim(0.7, 20)
  ax.set_ylim(200e6, 1e12)
  xticks = [0.8]+range(1,11)+[20]
  plt.xticks(xticks, [str(x) for x in xticks])
  yticks = [1e9,5e9,10e9,50e9,100e9,500e9,1e12]
  plt.yticks(yticks, [str(y) for y in yticks])
  ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: utils.float_format(x, dd=1)))
  ax.get_figure().savefig('Die_Transistors.' + imf, bbox_inches = 'tight')

  df['freq'] = df.nm[tech_norm] / df.nm # relative freq
  # print df.freq
  ax = dfplot.line(df, x='nm_norm', y='freq', key='tech', xlabel='Relative Feature Size (250nm/X)', ylabel='Frequency\nRelative to 250nm', ybold=True, xbold=True,
                   keyloc="upper left", legend='reverse', loglog=True, figsize=(10,4.5))
  lastx = xs[0]
  lasty = df.freq[0]
  for i, x in enumerate(xs[1:]):
    y = df.freq[i+1]
    ax.annotate('', xy=(x, lasty), xycoords='data', xytext=(x, y), textcoords='data',
                arrowprops=dict(arrowstyle='<->', lw=2, connectionstyle='arc', shrinkA=0, shrinkB=0, color='royalblue'))
    ax.annotate('{:0,.2f}'.format(y/lasty), xy=(x, lasty), xycoords='data',
                xytext=(0, -13), textcoords='offset points',
                rotation=0, size=16, va='center', ha='center',
                bbox=dict(boxstyle='round', fc='white'))
    lastx, lasty = x, y
  ax.set_xlim(0.7, 20)
  ax.set_ylim(0.6, 20)
  xticks = [0.8]+range(1,11)+[20]
  plt.xticks(xticks, [str(x) for x in xticks])
  yticks = [0.7,1,2,3,4,5,7,10,20]
  plt.yticks(yticks, [str(y) for y in yticks])
  ax.get_figure().savefig('Norm_Frequency.' + imf, bbox_inches = 'tight')

  df['maskcost'] = [CostBook[tech]['nre'] for tech in techs]
  df.maskcost = df.maskcost / df.maskcost[tech_norm]
  ax = dfplot.line(df, x='nm_norm', y='maskcost', key='tech', xlabel='', ylabel='Mask Cost\nRelative to 250nm', ybold=True, xbold=True,
                   keyloc="upper left", legend='reverse', loglog=True, figsize=(10,4.5))
  lastx = xs[0]
  lasty = df.maskcost[0]
  for i, x in enumerate(xs[1:]):
    y = df.maskcost[i+1]
    if (y/lasty) > 1.6:
      ax.annotate('', xy=(x, lasty), xycoords='data', xytext=(x, y), textcoords='data',
                  arrowprops=dict(arrowstyle='<->', lw=2, connectionstyle='arc', shrinkA=0, shrinkB=0, color='royalblue'))
    else:
      ax.annotate('', xy=(x, lasty), xycoords='data', xytext=(x, y), textcoords='data',
                  arrowprops=dict(arrowstyle='<-', lw=2, connectionstyle='arc', shrinkA=0, shrinkB=0, color='royalblue'))
    ax.annotate('{:0,.1f}'.format(y/lasty), xy=(x, lasty), xycoords='data',
                xytext=(0, -13), textcoords='offset points',
                rotation=0, size=16, va='center', ha='center',
                bbox=dict(boxstyle='round', fc='white'))
    lastx, lasty = x, y
  ax.plot([df.nm_norm['250nm'], df.nm_norm['90nm']], [df.maskcost['250nm'], df.maskcost['90nm']], ls='-', lw=2, color='gray')
  ax.plot([df.nm_norm['90nm'], df.nm_norm['40nm']], [df.maskcost['90nm'], df.maskcost['40nm']], ls='--', lw=2, color='gray')
  ax.plot([df.nm_norm['40nm'], df.nm_norm['16nm']], [df.maskcost['40nm'], df.maskcost['16nm']], ls='-', lw=2, color='gray')
  ax.set_xlim(0.7, 20)
  ax.set_ylim(0.5, 200)
  xticks = [0.8]+range(1,11)+[20]
  plt.xticks(xticks, [str(x) for x in xticks])
  yticks = [1,2,5,10,20,50,100,200]
  plt.yticks(yticks, [str(y) for y in yticks])
  ax.get_figure().savefig('Norm_MaskCost.' + imf, bbox_inches = 'tight')
  # print df.maskcost

  df['energy_per_op'] = (df.nm/df.nm[tech_norm])*(df.vdd/df.vdd[tech_norm])**2
  ax = dfplot.line(df, x='nm_norm', y='energy_per_op', key='tech', xlabel='', ylabel='Energy/OP\nRelative to 250nm', ybold=True, xbold=True,
                   keyloc="lower left", loglog=True, figsize=(10,4.5))
  lastx = xs[0]
  lasty = df.energy_per_op[0]
  for i, x in enumerate(xs[1:]):
    y = df.energy_per_op[i+1]
    if (lasty/y) > 1.6:
      ax.annotate('', xy=(x, lasty), xycoords='data', xytext=(x, y), textcoords='data',
                  arrowprops=dict(arrowstyle='<->', lw=2, connectionstyle='arc', shrinkA=0, shrinkB=0, color='royalblue'))
    else:
      ax.annotate('', xy=(x, lasty), xycoords='data', xytext=(x, y), textcoords='data',
                  arrowprops=dict(arrowstyle='<-', lw=2, connectionstyle='arc', shrinkA=0, shrinkB=0, color='royalblue'))
    ax.annotate('{:0,.2f}'.format(lasty/y), xy=(x, lasty), xycoords='data',
                xytext=(0, 13), textcoords='offset points',
                rotation=0, size=16, va='center', ha='center',
                bbox=dict(boxstyle='round', fc='white'))
    lastx, lasty = x, y
  ax.plot([df.nm_norm['250nm'], df.nm_norm['90nm']], [df.energy_per_op['250nm'], df.energy_per_op['90nm']], ls='-', lw=2, color='gray')
  ax.plot([df.nm_norm['90nm'], df.nm_norm['16nm']], [df.energy_per_op['90nm'], df.energy_per_op['16nm']], ls='--', lw=2, color='gray')
  ax.set_xlim(0.7, 20)
  ax.set_ylim(0.005, 2)
  xticks = [0.8]+range(1,11)+[20]
  plt.xticks(xticks, [str(x) for x in xticks])
  yticks = [0.01,0.02,0.05,0.1,0.2,0.5,1.0,2.0]
  plt.yticks(yticks, [str(y) for y in yticks])
  ax.get_figure().savefig('Norm_EnergyPerOP.' + imf, bbox_inches = 'tight')
  # print df.energy_per_op

  df['cost_mm2'] = df.wc / df.wa
  df['density'] = (df.nm[tech_norm]/df.nm)**2
  df['perf_mm2'] = df.density * df.freq

  for tech in ['65nm', '40nm', '28nm', '16nm']:
    df.perf_mm2[tech] = df.perf_mm2['90nm'] * (df.energy_per_op['90nm'] / df.energy_per_op[tech])

  df['cost_per_perf'] = df.cost_mm2 / df.perf_mm2
  df.cost_per_perf = df.cost_per_perf / df.cost_per_perf[tech_norm]
  ax = dfplot.line(df, x='nm_norm', y='cost_per_perf', key='tech', xlabel='', ylabel='Cost (\$) per OP/s\nRelative to 250nm', ybold=True, xbold=True,
                   keyloc="lower left", legend=None, loglog=True, figsize=(10,4.5))
  lastx = xs[0]
  lasty = df.cost_per_perf[0]
  for i, x in enumerate(xs[1:]):
    y = df.cost_per_perf[i+1]
    if (lasty/y) > 1.6:
      ax.annotate('', xy=(x, lasty), xycoords='data', xytext=(x, y), textcoords='data',
                  arrowprops=dict(arrowstyle='<->', lw=2, connectionstyle='arc', shrinkA=0, shrinkB=0, color='royalblue'))
    else:
      ax.annotate('', xy=(x, lasty), xycoords='data', xytext=(x, y), textcoords='data',
                  arrowprops=dict(arrowstyle='<-', lw=2, connectionstyle='arc', shrinkA=0, shrinkB=0, color='royalblue'))

    if (lasty/y) > 1.0:
      labelshift = 13
    else:
      labelshift = -20
    ax.annotate('{:0,.2f}'.format(lasty/y), xy=(x, lasty), xycoords='data',
                xytext=(0, labelshift), textcoords='offset points',
                rotation=0, size=16, va='center', ha='center',
                bbox=dict(boxstyle='round', fc='white'))
    lastx, lasty = x, y
  ax.plot([df.nm_norm['250nm'], df.nm_norm['90nm']], [df.cost_per_perf['250nm'], df.cost_per_perf['90nm']], ls='-', lw=2, color='gray')
  ax.plot([df.nm_norm['90nm'], df.nm_norm['16nm']], [df.cost_per_perf['90nm'], df.cost_per_perf['16nm']], ls='--', lw=2, color='gray')
  ax.set_xlim(0.7, 20)
  ax.set_ylim(0.01, 3)
  xticks = [0.8]+range(1,11)+[20]
  plt.xticks(xticks, [str(x) for x in xticks])
  yticks = [0.01,0.02,0.05,0.1,0.2,0.5,1.0,2.0]
  plt.yticks(yticks, [str(y) for y in yticks])
  ax.get_figure().savefig('Norm_CostPerPerf.' + imf, bbox_inches = 'tight')
  # print df.cost_per_perf

  print df[['die_trs', 'freq', 'maskcost', 'energy_per_op', 'cost_per_perf']]

if __name__ == '__main__':

  fr_plots()

  # --- #

  HTMLBody  = '<img src="40nm_MPWOpt.svg" height="650px", width="650px"/>'
  HTMLBody += '<img src="RCA_Replica.svg" height="650px", width="650px"/>'
  HTMLBody += '<img src="Volume_Cost.svg" height="650px", width="650px"/>'
  HTMLBody += '<img src="Tech_Scaling.svg" height="650px", width="1300px"/>'
  HTMLBody += '<img src="CostPerOP_NRE.svg" height="650px", width="650px"/>'

  CostDf = const_df()
  df2html(CostDf, 'FabCost', 'Regular', '<b>Fab Cost Constants</b>', HTMLBody)
