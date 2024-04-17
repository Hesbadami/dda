import pandas as pd
from utils import *

# read all data
s_flights = pd.read_csv('s_flights.csv')
s_flights['time'] = pd.to_datetime(s_flights['time'])
s_counts = pd.DataFrame(columns=['time', 'country'])
for c in [
    'UA',
    'PL',
    'RO',
    'HU',
    'SK',
    'BY',
    'MD'
]:
    bb = get_flights(s_flights, c)[['time']]
    bb['country'] = c
    s_counts = pd.concat([
        s_counts,
        bb
    ])

df = s_counts.groupby(['time', 'country'], as_index=False).size().sort_values(by='time')
df['time'] = df['time'].dt.date
df['time'] = pd.to_datetime(df['time'])

filler = pd.DataFrame(columns=['time', 'country'])
for c in [
    'UA',
    'PL',
    'RO',
    'HU',
    'SK',
    'BY',
    'MD'
]:
    tmp = pd.date_range(df['time'].min(), df['time'].max()).to_frame().reset_index(drop=True).rename(columns={0:'time'})
    tmp['country'] = c
    tmp = tmp[~tmp['time'].isin(df[df['country'] == c]['time'])]

    filler = pd.concat([filler, tmp])
filler['time'] = pd.to_datetime(filler['time'])
filler = filler.sort_values(by='time')
filler['size'] = np.nan
df2 = pd.concat([df, filler])
df2['time'] = pd.to_datetime(df2['time'])
df2 = df2.sort_values(by=['country', 'time'])
df2['size'] = df2['size'].interpolate()
df2['size'] = df2['size'].bfill().astype(int)
df2 = df2.sort_values(by=['time', 'country'])
df2.to_csv('flight_count.csv', index=False)
