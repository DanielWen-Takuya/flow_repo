"""
trajectory_path - csv file
flow_params - merge
steps - ? - 1
title - ? - 'Time Space Diagram'
max_speed - ? - 8
min_speed - ? - 0
start - ? - 0

I need to print the args or flow_params
"""
from flow.networks import MergeNetwork
import argparse
from collections import defaultdict
try:
    from matplotlib import pyplot as plt
except ImportError:
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Rectangle
import matplotlib.colors as colors
import numpy as np
import pandas as pd

def import_data_from_trajectory(fp, params=dict()):
    # Read trajectory csv into pandas dataframe
    df = pd.read_csv(fp)

    # Convert column names for backwards compatibility using emissions csv
    column_conversions = {
        'time': 'time_step',
        'lane_number': 'lane_id',
    }
    df = df.rename(columns=column_conversions)
    
    # Compute line segment ends by shifting dataframe by 1 row
    df[['next_pos', 'next_time']] = df.groupby('id')[['distance', 'time_step']].shift(-1)

    # Remove nans from data
    df = df[df['next_time'].notna()]

    return df


def get_time_space_data(data, params):
    switcher = {
        MergeNetwork: _merge,
    }

    # Get the function from switcher dictionary
    func = switcher[params['network']]

    # Execute the function
    segs, data = func(data)

    return segs, data


def _merge(data):
    keep_edges = {'inflow_merge', 'bottom', ':bottom_0'}
    data = data[data['edge_id'].isin(keep_edges)]

    segs = data[['time_step', 'distance', 'next_time', 'next_pos']].values.reshape((len(data), 2, 2))

    return segs, data


def plot_tsd(ax, df, segs, args, lane=None, ghost_edges=None, ghost_bounds=None):
    norm = plt.Normalize(args.min_speed, args.max_speed)

    xmin, xmax = df['time_step'].min(), df['time_step'].max()
    xbuffer = (xmax - xmin) * 0.025  # 2.5% of range
    ymin, ymax = df['distance'].min(), df['distance'].max()
    ybuffer = (ymax - ymin) * 0.025  # 2.5% of range

    ax.set_xlim(xmin - xbuffer, xmax + xbuffer)
    ax.set_ylim(ymin - ybuffer, ymax + ybuffer)

    lc = LineCollection(segs, cmap=my_cmap, norm=norm)
    lc.set_array(df['speed'].values)
    lc.set_linewidth(1)
    ax.add_collection(lc)
    ax.autoscale()

    rects = []
    rects.append(Rectangle((xmin, ymin), args.start - xmin, ymax - ymin))
    if rects:
        pc = PatchCollection(rects, facecolor='grey', alpha=0.5, edgecolor=None)
        pc.set_zorder(20)
        ax.add_collection(pc)
    ax.set_title('Time-Space Diagram', fontsize=25)
    ax.set_ylabel('Position (m)', fontsize=20)
    ax.set_xlabel('Time (s)', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    cbar = plt.colorbar(lc, ax=ax, norm=norm)
    cbar.set_label('Velocity (m/s)', fontsize=20)
    cbar.ax.tick_params(labelsize=18)


module = __import__("examples.exp_configs.non_rl", fromlist=['merge'])
flow_params = getattr(module, 'merge').flow_params
# some plotting parameters
cdict = {
    'red': ((0, 0, 0), (0.2, 1, 1), (0.6, 1, 1), (1, 0, 0)),
    'green': ((0, 0, 0), (0.2, 0, 0), (0.6, 1, 1), (1, 1, 1)),
    'blue': ((0, 0, 0), (0.2, 0, 0), (0.6, 0, 0), (1, 0, 0))
}
my_cmap = colors.LinearSegmentedColormap('my_colormap', cdict, 1024)

# Read trajectory csv into pandas dataframe
traj_df = import_data_from_trajectory(args.trajectory_path, flow_params)

# Convert df data into segments for plotting
segs, traj_df = get_time_space_data(traj_df, flow_params)

# perform plotting operation
fig = plt.figure(figsize=(16, 9))
ax = plt.axes()
plot_tsd(ax, traj_df, segs, args)

plt.plot([df['time_step'].min(), df['time_step'].max()],
         [0, 0], linewidth=3, color="white")        #
plt.plot([df['time_step'].min(), df['time_step'].max()],
         [-0.1, -0.1], linewidth=3, color="white") 