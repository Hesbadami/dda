import pandas as pd
# read all data
df = pd.read_csv('flight_count.csv')
holidays_dt = pd.read_csv('holidays.csv')
lockdowns_dt = pd.read_csv('lockdowns.csv')

df['time'] = pd.to_datetime(df['time'])
holidays_dt['time'] = pd.to_datetime(holidays_dt['time'])
lockdowns_dt['time'] = pd.to_datetime(lockdowns_dt['time'])
df['is_holiday'] = df.apply(lambda x: x['time'] in list(holidays_dt[holidays_dt['country'] == x['country']]['time']), axis=1).astype(int)
df['days_to_next_holiday'] = df.apply(lambda x: (holidays_dt[
                                                    (holidays_dt['time'] > x['time']) & 
                                                    (holidays_dt['country'] == x['country'])
                                                ].iloc[0]['time'] - x['time']).days, axis=1)

df['days_since_last_holiday'] = df.apply(lambda x: (x['time'] - holidays_dt[
                                                    (holidays_dt['time'] < x['time']) & 
                                                    (holidays_dt['country'] == x['country'])
                                                ].iloc[-1]['time']).days, axis=1)

df['business_days_in_month'] = df.apply(lambda x:
                                          x['time'].days_in_month - len(holidays_dt[(holidays_dt['time'].dt.strftime('%Y-%m') == x['time'].strftime('%Y-%m')) & (holidays_dt['country'] == x['country'])]['time'].unique())
                                         , axis=1)
df2 = pd.merge(df, lockdowns_dt, on=['time', 'country'], how='left')
df2['lockdown'] = df2['lockdown'].fillna(0)
df2.to_csv('flight_count_holiday_lockdown.csv', index=False)
