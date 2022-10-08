import numpy as np
from skimage import io
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as mpe
from skimage.color import rgb2gray, rgba2rgb
from skimage.filters import threshold_otsu
import pandas as pd
import math

# styling
LINEWIDTH=6
EDGEWIDTH=3
CAPSTYLE="projecting"
COLORMAP="viridis"
ALPHA=1
FIRSTDAY=6 # 0=Mon, 6=Sun
MARGIN=4
OUTLIER=3 #add outlier with 3x y limit
N_RINGS = 10
PX_DIAMETER = 700 # rough pixels
REAL_LIMIT = 1000
MIN_RADIUS = LINEWIDTH + EDGEWIDTH
PX_RADIUS = PX_DIAMETER / 2
CORE_STOP = 1

def to_spiral(n_rings, in_d, out_d):
    return n_rings * np.pi * (in_d + out_d) / 2

def from_arc(arc, radius):
    radians = arc / radius
    return radians / (2 * np.pi)

def scale_1(n, h):
    return REAL_LIMIT * (n / h)

def scale_2(n, px):
    return n * px

def inverse_scale_2(n, px):
    return n / px

barchart = rgb2gray(rgba2rgb(io.imread("barchart.png")))
out = barchart > threshold_otsu(barchart)

deltas = [scale_1(np.count_nonzero(o), out.shape[0]) for o in np.transpose(out)[::2]]
maximum = scale_1(out.shape[0]*OUTLIER, out.shape[0])
deltas = ([maximum] + deltas)

px_ratio = to_spiral(N_RINGS, MIN_RADIUS, PX_DIAMETER) / sum(deltas)
arcs = [scale_2(d, px_ratio) for d in deltas]
real_limit = max(arcs[1:])
starts = [CORE_STOP]
for arc in arcs:
    last_turn = starts[-1]
    last_rad = PX_DIAMETER * last_turn / N_RINGS
    n_turns = from_arc(arc, last_rad)
    starts += [last_turn + n_turns]

first_trip = pd.to_datetime("2017-04-01 00:00:00")
origin = (first_trip - pd.to_timedelta(first_trip.weekday() - FIRSTDAY, unit='d')).replace(hour=0, minute=0, second=0)
weekdays = pd.date_range(origin, origin + np.timedelta64(1, 'W')).strftime("%a").tolist()[:-1]

N = len(arcs)
df = pd.DataFrame()
df["dist"] = arcs
df["start"] = starts[0:-1]
df["stop"] = starts[1:]
plt.style.use('dark_background')
fig = plt.figure(figsize=(16, 12))
ax = fig.gca(projection="polar")

core_nsamples = int(1000. * (CORE_STOP - 0))
core_t = np.linspace(0, CORE_STOP, core_nsamples)
core_theta = 2 * np.pi * core_t
ax.plot(core_theta, core_t, lw=LINEWIDTH, color="black", solid_capstyle=CAPSTYLE, alpha=ALPHA)

for idx, event in df.iterrows():
    # sample normalized distance from colormap
    ndist = event['dist'] / real_limit
    color = plt.cm.get_cmap(COLORMAP)(ndist)
    tstart, tstop = event.loc[['start', 'stop']]
    # timestamps are in week fractions, 2pi is one week
    nsamples = int(10000. * (tstop - tstart))
    t = np.linspace(tstart, tstop, nsamples)
    theta = 2 * np.pi * t
    arc, = ax.plot(theta, t, lw=LINEWIDTH, color=color, solid_capstyle=CAPSTYLE, alpha=ALPHA)
    if EDGEWIDTH > 0:
        arc.set_path_effects([mpe.Stroke(linewidth=LINEWIDTH+EDGEWIDTH, foreground='black'), mpe.Normal()])


# grid and labels
ax.set_rticks([])
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.set_xticks([])
ax.axis("off")

ax.grid(True)
# setup a custom colorbar, everything's always a bit tricky with mpl colorbars
vmin = df['dist'].min()
norm = mpl.colors.Normalize(vmin=vmin, vmax=real_limit)
sm = plt.cm.ScalarMappable(cmap=COLORMAP, norm=norm)
sm.set_array([])
ticks = np.linspace(vmin, real_limit, 10)
tick_labels = [f"{int(inverse_scale_2(t, px_ratio))}" for t in ticks]
tick_labels[-1] += "+"
cbar = plt.colorbar(
    sm, ticks=ticks, fraction=0.04, 
    aspect=60, pad=0.1, ax=ax
)
cbar.ax.set_yticklabels(tick_labels)
cbar.ax.tick_params(labelsize=15)
cbar.set_label(label='duration',size=15,weight='bold')

plt.show()
