import utils
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import collections

#
# Stack lists
#
def lstack(*arg):
  return reduce(lambda x, y: map(add, x, y), list(arg))

#
# Plot Configuration
# Call this function before plt.plot() or plt.subplots().
#
def plot_config(style='Default'):
  if (style == 'Default'):
    matplotlib.rcParams['font.size'] = 17
    matplotlib.rcParams['legend.handlelength'] = 1.5 # short handle lines in legend
    plt.style.use('seaborn-muted')
    plt.rcParams.update({"text.usetex": False})

#
# Axis Configuration
# Call this function after the axis handle is created.
#
def axis_config(ax):
  ax.set_axisbelow(True)

  ax.grid(b=True, which='major', color='gray',      linestyle='-')
  ax.grid(b=True, which='minor', color='lightgray', linestyle='-')

  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)

  ax.tick_params(which='both',  width  = 2)
  ax.tick_params(which='major', length = 7)
  ax.tick_params(which='minor', length = 4, color='gray')

#
# legend locator
#
def legend_location(keyloc, **kwargs):
  ax = kwargs.get('ax', None)
  # df.plot defaults legend to True
  legend = kwargs.get('legend', True)
  if (legend != None) and (legend != False):
    handles, labels = ax.get_legend_handles_labels()
    if (legend == 'reverse'):
      handles.reverse()
      labels.reverse()
    if (keyloc == 'outside right'): # this is not a standard location
      box = ax.get_position()
      # Shrink current axis by 20%.
      ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
      # Move legend to the right of the current axis.
      ax.legend(handles, labels, numpoints=1, prop=dict(family='monospace', size=16), loc='center left', bbox_to_anchor=(1, 0.5))
    elif (keyloc == 'pareto'): # for pareto plots
      ax.legend(handles, labels, numpoints=1, prop=dict(family='monospace', size=16), loc='lower center', bbox_to_anchor=(0.7, 0.01))
    elif (keyloc == 'normalize'): # for tco normalize
      ax.legend(handles, labels, numpoints=1, prop=dict(family='monospace', size=12), loc='upper center', ncol = 8, mode = 'expand', borderaxespad = 1)
    else:
      ax.legend(handles, labels, numpoints=1, prop=dict(family='monospace', size=16), loc=keyloc)

#
# Solve the intersection of two straight lines: y = a1 * x + b1; y = a2 * x + b2
#
def line_cross((a1, b1), (a2, b2)):
  if a1 == a2:
    xc, yc = None, None
  else:
    xc = (b2 - b1) / (a1 - a2)
    yc = a1 * xc + b1
  return xc, yc

#
# Create line plots with markers.
#
def line(dataframe, x, y,
         key=None, sort_key=False, keyloc='outside right',
         xlabel='X Axis', xbold=False, ylabel='Y Axis', ybold=False, datalabel=False, **kwargs):

  plot_config()

  # check df.plot args on http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.plot.html
  # kwargs.get('key', default_value)
  ax = kwargs.get('ax', None)
  if ax == None:
    figsize = kwargs.get('figsize', (10, 8))
    sharex = kwargs.get('sharex', False)
    sharey = kwargs.get('sharey', False)
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=sharex, sharey=sharey, figsize=figsize, squeeze=True)
    kwargs['ax'] = ax

  axis_config(ax)

  # autoscale visual area with graph content
  ax.autoscale(enable=True, axis='both', tight=None)

  # Dropping many not make sense if some irrelvant columns contain NaN.
  # dataframe = dataframe.dropna() # drop rows that contain NaN

  kwargs['markersize'] = kwargs.get('markersize', 12)
  kwargs['fillstyle']  = kwargs.get('fillstyle',  'full')
  kwargs['linestyle']  = kwargs.get('linestyle',  ':')


  if key != None:
    keys = dataframe[key].unique().tolist()
    if sort_key:
      keys.sort() # can do reversed sorting if legend=='reverse'
    labeltext = map(str, keys)
    labelwidth = max(map(len, labeltext)) # for equalizing label text width
    labels = map(lambda label: '{:>{w}}'.format(label, w=labelwidth), labeltext)

    kwargs.pop('marker', None)
    kwargs.pop('label', None)
    markers = itertools.cycle(['o', 'v', 's', '^', 'd', '<', 'p', '8', '>', 'h', 'H', '*', 'd', '1', '2', '3', '4', 'x', '+'])
    for k, l in zip(keys, labels):
      df = dataframe[dataframe[key] == k]
      if ( df[x].min() == df[x].max() ): # fallback to ax.plot to avoid "UserWarning: Attempting to set identical left==right results"; should be a bug in Pandas
        ax.plot(df[x].tolist(), df[y].tolist(), marker=markers.next(), markersize=kwargs['markersize'],
                fillstyle=kwargs['fillstyle'], linestyle=kwargs['linestyle'], label=l)
      else:
        df.plot.line(x, y, marker=markers.next(), label=l, **kwargs)
  else:
    kwargs['marker'] = kwargs.get('marker', 'o')
    if ( dataframe[x].min() == dataframe[x].max() ): # fallback to ax.plot to avoid "UserWarning: Attempting to set identical left==right results"; should be a bug in Pandas
      ax.plot(dataframe[x].tolist(), dataframe[y].tolist(), marker=kwargs['marker'], markersize=kwargs['markersize'],
              fillstyle=kwargs['fillstyle'], linestyle=kwargs['linestyle'], label=y)
    else:
      dataframe.plot.line(x, y, **kwargs)

  if datalabel:
    df = dataframe.sort_values(by=y, ascending=True)
    xs = df[x].tolist()
    ys = df[y].tolist()
    lastx, lasty = 0, 0
    for xy in zip(xs, ys):
      ytext = -12
      if (xy[0] < 10*lastx) and (xy[1] < 1.5*lasty): # avoid text overlapping
        ytext = 12
      xstr = utils.float_format(xy[0])
      ystr = utils.float_format(xy[1])
      ax.annotate('({}, {})'.format(xstr, ystr), xy=xy, xycoords='data', xytext=(0, ytext), textcoords='offset points', va='center', ha='center')
      lastx, lasty = xy[0], xy[1]

  if xbold is True:
    ax.set_xlabel(xlabel, fontweight='bold')
  else:
    ax.set_xlabel(xlabel)

  if ybold is True:
    ax.set_ylabel(ylabel, fontweight='bold')
  else:
    ax.set_ylabel(ylabel)

  # Set major tick locator for both axes; note this might move already-placed major ticks.
  ax.xaxis.set_major_locator(matplotlib.ticker.AutoLocator())
  ax.yaxis.set_major_locator(matplotlib.ticker.AutoLocator())
  # Set minor tick locator for both axes; note this might move already-placed minor ticks.
  ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
  ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

  if kwargs.get('logx', False):
    #  ax.set_xscale('log', nonposx='clip')
    ax.set_xscale('log')
  if kwargs.get('logy', False):
    #  ax.set_yscale('log', nonposx='clip')
    ax.set_yscale('log')
  if kwargs.get('loglog', False):
    #  ax.set_xscale('log', nonposx='clip')
    #  ax.set_yscale('log', nonposx='clip')
    ax.set_xscale('log')
    ax.set_yscale('log')

  ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: utils.float_format(x, dd=2)))
  ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: utils.float_format(x, dd=2)))

  #  legend_location(keyloc, **kwargs)

  # Do show() and savefig() after post-function-call adjustments with the ax handle.
  # plt.show()
  # fig = ax.get_figure()
  # fig.savefig('figure.pdf', bbox_inches = 'tight')

  return ax

#
# Create bar plots.
#
def bar(dataframe,
        keyloc='outside right',
        xlabel='X Axis', ylabel='Y Axis', **kwargs):

  plot_config()

  # check df.plot args on http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.plot.html
  # kwargs.get('key', default_value)
  ax = kwargs.get('ax', None)
  if ax == None:
    figsize = kwargs.get('figsize', (10, 8))
    sharex = kwargs.get('sharex', False)
    sharey = kwargs.get('sharey', False)
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=sharex, sharey=sharey, figsize=figsize, squeeze=True)
    kwargs['ax'] = ax

  axis_config(ax)

  dataframe.plot.bar(**kwargs)

  # bar plot doesn't need X grids
  ax.grid(b=False, axis='x', which='both')
  ax.tick_params(axis='x', which='both', length=0)

  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)

  ax.yaxis.set_major_locator(matplotlib.ticker.AutoLocator())
  ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

  if kwargs.get('logx', False):
    #  ax.set_xscale('log', nonposx='clip')
    ax.set_xscale('log')
  if kwargs.get('logy', False):
    #  ax.set_yscale('log', nonposx='clip')
    ax.set_yscale('log')
  if kwargs.get('loglog', False):
    #  ax.set_xscale('log', nonposx='clip')
    #  ax.set_yscale('log', nonposx='clip')
    ax.set_xscale('log')
    ax.set_yscale('log')

  ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: utils.float_format(x, dd=2)))
  # plt.setp(ax.get_xticklabels(), rotation=0)

  legend_location(keyloc, **kwargs)

  return ax

#
# Create multi keys line plots with markers.
#
def line_multi_keys(dataframe, x, y,
         key=None, sort_key=False, keyloc='outside right',
         xlabel='X Axis', xbold=False, ylabel='Y Axis', ybold=False, datalabel=False, **kwargs):

  plot_config()

  # check df.plot args on http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.plot.html
  # kwargs.get('key', default_value)
  ax = kwargs.get('ax', None)
  if ax == None:
    figsize = kwargs.get('figsize', (10, 8))
    sharex = kwargs.get('sharex', False)
    sharey = kwargs.get('sharey', False)
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=sharex, sharey=sharey, figsize=figsize, squeeze=True)
    kwargs['ax'] = ax

  axis_config(ax)

  # autoscale visual area with graph content
  ax.autoscale(enable=True, axis='both', tight=None)

  # Dropping many not make sense if some irrelvant columns contain NaN.
  # dataframe = dataframe.dropna() # drop rows that contain NaN

  kwargs['markersize'] = kwargs.get('markersize', 12)
  kwargs['fillstyle']  = kwargs.get('fillstyle',  'full')
  kwargs['linestyle']  = kwargs.get('linestyle',  ':')

  if len(key) == 2:
    keys0 = dataframe[key[0]].unique().tolist()
    keys1 = dataframe[key[1]].unique().tolist()
    if sort_key:
      keys0.sort() # can do reversed sorting if legend=='reverse'
      keys1.sort() # can do reversed sorting if legend=='reverse'
    labels0 = map(str, keys0)
    labels1 = map(str, keys1)
    kwargs.pop('marker', None)
    kwargs.pop('label', None)
    colors = itertools.cycle(['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray'])
    used_colors = {}
    used_markers = {}
    for k0, l0 in zip(keys0, labels0):
      color = colors.next()
      used_colors[int(l0[:-2])] = color
      markers = itertools.cycle(['+', 'x', 'o', 'v', 'd', '^', 's', '<', 'p', '8', '>', 'h', 'H', '*', 'd', '1', '2', '3', '4'])
      for k1, l1 in zip(keys1, labels1):
        marker = markers.next() 
        used_markers[int(l1[:-2])] = marker
        df0 = dataframe[dataframe[key[0]] == k0]
        df = df0[df0[key[1]] == k1]
        if not df.empty:
          df.plot.line(x, y, marker=marker, color=color, **kwargs)
  else:
    kwargs['marker'] = kwargs.get('marker', 'o')
    if ( dataframe[x].min() == dataframe[x].max() ): # fallback to ax.plot to avoid "UserWarning: Attempting to set identical left==right results"; should be a bug in Pandas
      ax.plot(dataframe[x].tolist(), dataframe[y].tolist(), marker=kwargs['marker'], markersize=kwargs['markersize'],
              fillstyle=kwargs['fillstyle'], linestyle=kwargs['linestyle'], label=y)
    else:
      dataframe.plot.line(x, y, **kwargs)

  used_colors = collections.OrderedDict(sorted(used_colors.items()))
  used_markers = collections.OrderedDict(sorted(used_markers.items()))
  f = lambda m,c: ax.plot([],[],marker=m, color=c, ls="none")[0]
  handles = []
  labels = []
  for l in used_colors:
    handles.append(f("s", used_colors[l]))
    labels.append(str(l)+'nm')
  for l in used_markers:
    handles.append(f(used_markers[l], 'k'))
    labels.append(str(l)+' PEs')

  if xbold is True:
    ax.set_xlabel(xlabel, fontweight='bold')
  else:
    ax.set_xlabel(xlabel)

  if ybold is True:
    ax.set_ylabel(ylabel, fontweight='bold')
  else:
    ax.set_ylabel(ylabel)

  # Set major tick locator for both axes; note this might move already-placed major ticks.
  ax.xaxis.set_major_locator(matplotlib.ticker.AutoLocator())
  ax.yaxis.set_major_locator(matplotlib.ticker.AutoLocator())
  # Set minor tick locator for both axes; note this might move already-placed minor ticks.
  ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
  ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

  if kwargs.get('logx', False):
    ax.set_xscale('log')
  if kwargs.get('logy', False):
    ax.set_yscale('log')
  if kwargs.get('loglog', False):
    ax.set_xscale('log')
    ax.set_yscale('log')

  ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: utils.float_format(x, dd=2)))
  ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: utils.float_format(x, dd=2)))

  legend_location(keyloc, **kwargs)

  return ax, handles, labels


if __name__ == '__main__':
  # df = pd.DataFrame([[1e3,2e3,1],[1e3,4e3,2],[2e3,6e3,3], [2e3,3e3,4]], columns=['xvals', 'yvals', 'keys'])
  # print df

  # ax = line(df, x='xvals', y='yvals', key='keys',  linestyle='-.', loglog=True)
  # ax.set_xlim(1e2, 1e4)
  # ax.set_ylim(1e2, 1e4)

  # df = pd.DataFrame([[1e3,2e3,3e3],[4e3,5e3,6e3],[7e3,8e3,9e3]], index=['x','y','z'], columns=['a', 'b', 'c'])
  # print df

  # ax = bar(df, rot=0, legend='reverse')

  print line_cross((1, 2), (2, 3))

  # plt.show()
