# python 3.8

import pandas as pd
from lockdowndates.core import LockdownDates
keshvar_ha = {
    'UA': 'Ukraine',
    'PL': 'Poland',
    'RO': 'Romania',
    'HU': 'Hungary',
    'SK': 'Hungary',
    'BY': 'Belarus',
    'MD': 'Hungary'
}
lockdowns_dt = pd.DataFrame()
for c in keshvar_ha:
    ld = LockdownDates(keshvar_ha[c], df2['time'].min().strftime('%Y-%m-%d'), df2['time'].max().strftime('%Y-%m-%d'), ("stay_at_home","masks"))
    lockdown_dates = ld.dates().reset_index()
    lockdown_dates.columns = ['time', 'country', 'mask', 'lockdown']
    lockdown_dates['country'] = c
    lockdowns_dt = pd.concat([
        lockdowns_dt,
        lockdown_dates.drop(columns='mask')
    ])

lockdowns_dt['time'] = pd.to_datetime(lockdowns_dt['time'])
lockdowns_dt = lockdowns_dt.sort_values(by=['country', 'time'])
lockdowns_dt['lockdown'] = lockdowns_dt['lockdown'].ffill()
lockdowns_dt = lockdowns_dt.sort_values(by=['time', 'country'])
lockdowns_dt.to_csv('lockdowns.csv', index=False)
