# Python 3.11

import pandas as pd
from workalendar.europe import Ukraine, Poland, Hungary, Slovakia, Romania, Belarus
keshvar_ha = {
    'UA': Ukraine,
    'PL': Poland,
    'RO': Romania,
    'HU': Hungary,
    'SK': Slovakia,
    'BY': Belarus,
    'MD': Belarus
}
holidays_dt = {'time':[], 'country':[], 'name':[]}
for c in keshvar_ha:
    for t in list(df2['time'].dt.year.unique())+[2024]:
        for h in keshvar_ha[c]().holidays(year=t):
            holidays_dt['time'].append(h[0])
            holidays_dt['country'].append(c)
            holidays_dt['name'].append(h[1])

holidays_dt = pd.DataFrame(holidays_dt)
holidays_dt['time'] = pd.to_datetime(holidays_dt['time'])
holidays_dt = holidays_dt.sort_values(by=['time', 'country'])
holidays_dt.to_csv('holidays.csv', index=False)
