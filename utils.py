import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib import dates, cm, colors
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
#from traffic.data import eurofirs
from cartopy.crs import *
from cartopy.feature import COASTLINE, BORDERS
from cartes.utils.features import countries, ocean, lakes
from cartopy import feature
import numpy as np
import glob
import contextlib
from PIL import Image
from matplotlib.patches import ConnectionPatch
import numpy as np
from scipy.special import binom

airports = pd.read_csv('airports.csv')

def get_paths(df):
    df['path_init'] = list(zip(df['dep_lat'], df['dep_lon']))
    df['path_end'] = list(zip(df['arr_lat'], df['arr_lon']))
    paths = df[['path_init', 'path_end']].values
    paths.sort(axis=1)
    paths = pd.DataFrame(paths, index=df.index, columns=['path_init', 'path_end'])
    paths['time'] = df['time']
    
    return paths

def bernstein_poly(i, n, t):
    return binom(n, i) * (t ** i) * ((1 - t) ** (n - i))

def bezier_curve(points, nTimes=100):
    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals
    
def hanging_line(point1, point2, height=0.01):
    # Example usage:
    # Control points for Bezier curve. You can add a middle point to control the height of the curve.
    control_points = [point1, [(point1[0] + point2[0]) / 2, max(point1[1], point2[1]) + 0.01], point2]
    xvals, yvals = bezier_curve(control_points, nTimes=100)
    return xvals, yvals
    
def plot_date(df, date=None, ax=None, cbar=True, vmin=False, vmax=False, projection=PlateCarree(), bounds=None, figsize=None, label_prop={'fontsize': 20}, arc_height = 0.1):
    if date:
        data = df[df['time'] == date]
    else:
        data = df
    paths = get_paths(data)
    data = paths.groupby(['path_init', 'path_end'], as_index=True).size().reset_index()
    data.columns = list(data.columns.values[:-1])+['c']
    data = data.sort_values('c')

    data['curve'] = data[['path_init', 'path_end']].apply(lambda row: hanging_line(row['path_init'], row['path_end'], arc_height), axis=1)

    if vmax == False:
        vmax = data['c'].max()
    if vmin == False:
        vmin = data['c'].min()

    norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.Reds)

    crs = PlateCarree()
    
    if not ax:
        if figsize:
            fig, ax = plt.subplots(1, 1, frameon=True, figsize=figsize, dpi=150, subplot_kw=dict(projection=projection))
        else:
            fig, ax = plt.subplots(1, 1, frameon=True, subplot_kw=dict(projection=projection))
    
    ax.add_feature(lakes(scale="50m"))
    ax.add_feature(ocean(scale="50m"))
    
    ax.add_feature(COASTLINE.with_scale("50m"), lw=0.5)
    ax.add_feature(BORDERS.with_scale("50m"), lw=0.3)

    ax.set_facecolor("#dedede")

    gl = ax.gridlines(crs=crs, draw_labels=True,
                      linewidth=.6, color='gray', alpha=0.5, linestyle='-.')
    
    if cbar:
        cb = plt.colorbar(mapper, ax=ax, orientation='horizontal', shrink=0.5, pad=0.04, aspect=40, label = '# of flights')

    for index, row in data.dropna().iterrows():
        ax.plot(row['curve'][0], row['curve'][1], c=mapper.to_rgba(row['c']), linewidth = 2, zorder=100+row['c'], transform=crs)
        ax.scatter([row['path_init'][0], row['path_end'][0]], [row['path_init'][1], row['path_end'][1]], s=np.log(row['c']+1)*10, c=mapper.to_rgba(row['c']), zorder=100+row['c'], transform=crs)

    text = AnchoredText(
        datetime.strftime(date, '%B %d, %Y'),
        loc=1,
        frameon=True,
        prop = label_prop
    )
    
    text.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(text)

    if not bounds:
        # lon0, lat0, lon1, lat1s
        lon0 = data[['dep_lon', 'arr_lon']].min().min()
        lat0 = data[['dep_lat', 'arr_lat']].min().min()
        lon1 = data[['dep_lon', 'arr_lon']].max().max()
        lat1 = data[['dep_lat', 'arr_lat']].max().max()
        bounds = [lat0-10, lat1+10, lon0-10, lon1+10]
    ax.set_extent(bounds, crs)
    return ax
    
def get_coor(coords, l):
    if not pd.isna(coords):
        if l == 'lat':
            return float(coords.split(',')[0])
        elif l == 'lon':
            return float(coords.split(',')[1])
    else:
        return None

def get_gif(fp_in, fp_out):
    with contextlib.ExitStack() as stack:
    
        # lazily load images
        imgs = (stack.enter_context(Image.open(f))
                for f in natsorted(glob.glob(fp_in)))
    
        # extract  first image from iterator
        img = next(imgs)
    
        # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
        img.save(fp=fp_out, format='GIF', append_images=imgs,
                 save_all=True, duration=100, loop=0)

def plot_tseries(data, freq='Y', ax = None, c='k', label=None, kwargs = {}):
    daily_counts = data.groupby('time').size()
    if not ax:
        ax = daily_counts.plot(c=c, figsize=(16, 4.5), label=label, **kwargs)
    else:
        daily_counts.plot(c=c, ax=ax, label=label,  **kwargs)
    if freq=='M':
        ax.xaxis.set_major_locator(dates.MonthLocator())
        ax.xaxis.set_major_formatter(dates.DateFormatter('%b\n%Y'))
    else:
        ax.xaxis.set_major_locator(dates.YearLocator())
        ax.xaxis.set_major_formatter(dates.DateFormatter('%Y'))
            
    labels = list(ax.get_xticklabels())
    jans = ([i for i in range(len(labels)) if 'Jan' in str(labels[i])])
    for i in range(len(labels)):
        text = labels[i].get_text()
        labels[i].set_ha('center')
        if not ('Jan' in text or i == 0):
            labels[i] = (text.split('\n')[0])
            
    ax.set_xticklabels(labels)
    plt.tight_layout()
    return ax


def get_dt(d, format='%Y-%m-%d'):
    if type(d) == int or type(d) == float:
        return datetime.fromtimestamp(d)
    elif type(d) == str:
        return datetime.strptime(d, format)
        
def get_ts(d, format='%Y-%m-%d'):
    if type(d) == str:
        d = datetime.strptime(d, format)
    return datetime.timestamp(d)

def add_label(ax, tt):
    text = AnchoredText(
        tt,
        loc=1,
        frameon=True,
    )
    
    text.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(text)
    return ax

def get_airports(s_countries):
    if type(s_countries) != list:
        s_countries = [s_countries]
        
    s_airports = airports[
        (airports['iso_country'].isin(s_countries)) &
        (
            (~airports['gps_code'].isna()) |
            (
                (airports['gps_code'].isna()) &  # needs filling nans manually
                (~airports['iata_code'].isna())
            )
        )
    ]
    
    return s_airports

def get_flights(data, s_countries):

    s_airports = get_airports(s_countries)
    
    s_flights = data[
        (~((data['estdepartureairport'].isna()) & (data['estarrivalairport'].isna()))) &
        (
            (data['estdepartureairport'].isin(s_airports['gps_code'].dropna())) | 
            (data['estarrivalairport'].isin(s_airports['gps_code'].dropna()))
        )
    ]

    return s_flights.reset_index(drop = True)

def featurize_flights(s_flights):
    s_flights['day'] = s_flights['day'].astype(int)
    s_flights = s_flights.sort_values(by = 'day')

    s_flights['time'] = s_flights['day'].apply(lambda x: datetime.fromtimestamp(x))

    linked_airports = airports[
        (airports['gps_code'].isin(s_flights['estarrivalairport'].dropna())) | 
        (airports['gps_code'].isin(s_flights['estdepartureairport'].dropna())) | 
        (airports['iata_code'].isin(s_flights['estarrivalairport'].dropna())) | 
        (airports['iata_code'].isin(s_flights['estdepartureairport'].dropna()))
    ][['iso_country', 'gps_code', 'iata_code', 'coordinates']]
    
    linked_airports['code'] = linked_airports['gps_code'].fillna(linked_airports['iata_code'])
    linked_airports = linked_airports.drop(columns = ['gps_code', 'iata_code'])

    s_flights = pd.merge(s_flights, linked_airports, left_on='estdepartureairport', right_on='code', how='left')
    s_flights = s_flights.rename(columns={'coordinates': 'dep_coor', 'iso_country': 'dep_country'}).drop(columns='code')
    
    s_flights = pd.merge(s_flights, linked_airports, left_on='estarrivalairport', right_on='code', how='left')
    s_flights = s_flights.rename(columns={'coordinates': 'arr_coor',  'iso_country': 'arr_country'}).drop(columns='code')
    
    s_flights['dep_lat'] = s_flights['dep_coor'].apply(lambda x: get_coor(x, 'lat'))
    s_flights['dep_lon'] = s_flights['dep_coor'].apply(lambda x: get_coor(x, 'lon'))
    s_flights['arr_lat'] = s_flights['arr_coor'].apply(lambda x: get_coor(x, 'lat'))
    s_flights['arr_lon'] = s_flights['arr_coor'].apply(lambda x: get_coor(x, 'lon'))
    s_flights = s_flights.drop(columns=['dep_coor', 'arr_coor'])

    return s_flights.reset_index(drop = True)

def shade_tb(ax, bounds, alpha=1, facecolor='#dedede'):
    x1, x2, y1, y2 = bounds
    rect1 = (x1, 0, x2 - x1, y1)
    rect2 = (x1, y2, x2 - x1, 5000)
    box = ax.indicate_inset(rect2, lw = 0, alpha=alpha, zorder=0, facecolor=facecolor)
    box = ax.indicate_inset(rect1, lw = 0, alpha=alpha, zorder=0, facecolor=facecolor)

def draw_box(ax, bounds, lw=2, alpha=1, zorder=0, facecolor='white', **box_kwargs):
    x1, x2, y1, y2 = bounds
    rect = (x1, y1, x2 - x1, y2 - y1)
    box = ax.indicate_inset(rect, alpha=alpha,lw=lw, zorder=zorder, facecolor=facecolor, **box_kwargs)
    
def zoom_window(ax, bounds, loc, con = {}, con_type = 'middle', box_kwargs = {}, con_kwargs = {}):
    lw = 2
    if 'lw' in box_kwargs:
        lw = box_kwargs['lw']
        del box_kwargs['lw']
    x1, x2, y1, y2 = bounds
    axins = ax.inset_axes(
        loc,
        zorder = 3,
        xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[])
    rect = (x1, y1, x2 - x1, y2 - y1)
    box = ax.indicate_inset(rect, alpha=1,lw=lw, zorder=0, facecolor='white', **box_kwargs)
    box = ax.indicate_inset(rect, alpha=1,lw=lw, zorder=1000, **box_kwargs)
    for c in con:
        if con_type == 'middle':
            if c == 0:
                Ax, Ay = ((x2+x1)/2, y2)
            elif c == 1:
                Ax, Ay = (x2, (y1+y2)/2)
            elif c == 2:
                Ax, Ay = ((x1+x2)/2, y1)
            elif c == 3:
                Ax, Ay = (x1, (y1+y2)/2)
            else:
                continue
                
            if con[c] == 0:
                Bx, By = (0.5, 1)
            elif con[c] == 1:
                Bx, By = (1, 0.5)
            elif con[c] == 2:
                Bx, By = (0.5, 0)
            elif con[c] == 3:
                Bx, By = (0, 0.5)
            else:
                continue
        
        elif con_type == 'corner':
            if c == 0:
                Ax, Ay = (x1, y2)
            elif c == 1:
                Ax, Ay = (x2, y2)
            elif c == 2:
                Ax, Ay = (x1, y1)
            elif c == 3:
                Ax, Ay = (x2, y1)
            else:
                continue
                
            ymin, ymax = 0, 1
            xmin, xmax = 0, 1
            if con[c] == 0:
                Bx, By = (xmin, ymax)
            elif con[c] == 1:
                Bx, By = (xmax, ymax)
            elif con[c] == 2:
                Bx, By = (xmin, ymin)
            elif con[c] == 3:
                Bx, By = (xmax, ymin)
            else:
                continue
        
        cp = ConnectionPatch(xyA=(Ax, Ay), xyB=(Bx, By), axesA=ax, axesB=axins,
                              coordsA="data", coordsB="axes fraction", lw=1, ls=":", zorder=1, **con_kwargs)
        ax.add_patch(cp)
    return axins

def add_ticks(ax, data, dates, labels, locator):
    if type(dates) != list:
        dates = [dates]
    if type(labels) != list:
        labels = [labels]
    if len(dates) != len(labels):
        raise "Not equal number of dates and labels"

    ticks = list(ax.get_xticks())
    labs = list(ax.get_xticklabels())

    for date, label in tuple(zip(dates, labels)):
        d = get_tt(data, date, locator=locator)
        ticks.append(d)
        labs.append(label)
    
        ax.set_xticks(ticks)
        ax.set_xticklabels(labs)
    return ax

def get_range(df, d1, d2):
    return df[
        (df['time'] >= d1) &
        (df['time'] <= d2)
    ]


def get_tt(data, date, dir=0, locator=dates.MonthLocator()):
    fig, ax = plt.subplots()
    daily_counts = data.groupby('time').size()
    daily_counts.plot(ax=ax)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(dates.DateFormatter('%Y-%m-%d'))
    ticks = list(ax.get_xticks())
    labels = [get_dt(label.get_text()) for label in ax.get_xticklabels()]
    min_close = (max([i for i, id_ in enumerate(labels) if get_ts(id_) <= get_ts(date)]))
    max_close = (min([i for i, id_ in enumerate(labels) if get_ts(id_) >= get_ts(date)]))
    date = get_ts(date)
    actual = ((date - get_ts(labels[0]))/(get_ts(labels[-1]) - get_ts(labels[0])))
    min_est = ((ticks[min_close] - ticks[0])/(ticks[-1] - ticks[0]))
    max_est = ((ticks[max_close] - ticks[0])/(ticks[-1] - ticks[0]))

    plt.close()
    if np.abs(actual-min_est) >= np.abs(actual-max_est):
        return ticks[max_close]
    else:
        return ticks[min_close]

