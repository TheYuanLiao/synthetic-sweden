import os
import sys
import subprocess
import pandas as pd
import sqlalchemy
from tqdm import tqdm
import numpy as np
import time


def get_repo_root():
    """Get the root directory of the repo."""
    dir_in_repo = os.path.dirname(os.path.abspath('__file__'))
    return subprocess.check_output('git rev-parse --show-toplevel'.split(),
                                   cwd=dir_in_repo,
                                   universal_newlines=True).rstrip()


ROOT_dir = get_repo_root()
sys.path.append(ROOT_dir)
sys.path.insert(0, ROOT_dir + '/lib')

import lib.dataworkers as dw
import lib.ev as ev


class EVSimStatsSpatiotemporal:
    def __init__(self, scenario=None):
        self.scenario = scenario
        self.df_trips = None
        self.agents = None

    def load_agents_trips(self):
        file_path2output_summary = ROOT_dir + f'/dbs/output_summary/{self.scenario}/'
        file = ROOT_dir + f'/results/{self.scenario}_stats_indi.csv'
        df_stats_indi = pd.read_csv(file)
        df_stats_indi.person = df_stats_indi.person.astype(int)
        self.agents = df_stats_indi.person.unique()
        self.df_trips = pd.read_csv(file_path2output_summary + 'charging_opportunity.csv')
        self.df_trips.PId = self.df_trips.PId.astype(int)
        self.df_trips = self.df_trips.loc[self.df_trips['PId'].isin(self.agents), :]

    def charging_points(self, paraset=None):
        user = dw.keys_manager['database']['user']
        password = dw.keys_manager['database']['password']
        port = dw.keys_manager['database']['port']
        db_name = dw.keys_manager['database']['name']
        engine = sqlalchemy.create_engine(f'postgresql://{user}:{password}@localhost:{port}/{db_name}')
        vars = ['person', 'purpose', 'car', 'seq', 'time', 'distance_driven', 'home_charger'] + \
               ['soc_0_' + str(x) for x in (1, 2, 3)] + \
               ['soc_' + str(x) for x in (1, 2, 3)] + \
               ['energy_' + str(x) for x in (1, 2, 3)] + ['charger_' + str(x) for x in (1, 2, 3)]
        vars_sql = ','.join(vars)
        start_time = time.time()
        df = pd.read_sql(f'''SELECT {vars_sql} FROM scenario_vg_car_ev_sim_paraset{paraset}_5days;''', con=engine)
        df.person = df.person.astype(int)
        print('Data loaded in %.2f minutes.'%((time.time() - start_time) / 60))

        # Get parking
        def get_parking(data):
            data = data.sort_values(by=['seq'])
            data.loc[:, 'deltaT'] = data.loc[:, 'time'].diff()
            data = data.loc[~data['purpose'].isnull(), :]
            return data

        start_time = time.time()
        tqdm.pandas()
        df_parking = df.groupby(['person']).progress_apply(get_parking).reset_index(drop=True)
        df_parking = df_parking.loc[(df_parking.energy_1 != 0) |
                                    (df_parking.energy_2 != 0) |
                                    (df_parking.energy_3 != 0), :]
        df_parking.loc[:, 'act_start'] = (df_parking.loc[:, 'time'] - df_parking.loc[:, 'deltaT']) / 60
        print('Got parking in %.2f minutes.' % ((time.time() - start_time) / 60))

        # Re-structure data
        start_time = time.time()
        vars = [x for x in df_parking.columns if ('1' not in x) & ('2' not in x) & ('3' not in x)]
        df_list = []
        for type in range(1, 4):
            vars_dict = {f'soc_{type}': 'soc', f'energy_{type}': 'energy', f'charger_{type}': 'charger'}
            temp = df_parking.loc[:, vars + [f'soc_0_{type}', f'soc_{type}', f'energy_{type}', f'charger_{type}']].rename(
                columns={f'soc_0_{type}': 'soc_baseline'})
            temp = temp.rename(columns=vars_dict)
            temp.loc[:, 'charging_type'] = type
            df_list.append(temp)
        df_parking = pd.concat(df_list)
        df_parking = df_parking.loc[df_parking.energy > 0, :]
        print('Restructured data in %.2f minutes.' % ((time.time() - start_time) / 60))

        # Find charging point
        start_time = time.time()
        def find_charging_point(data):
            act_id = []
            deso = []
            X = []
            Y = []
            act_start_ref = []
            for _, row in data.iterrows():
                act_start_0 = row['act_start']
                candidates = self.df_trips.loc[self.df_trips.PId == int(row['person']),
                                               ['act_id', 'deso', 'POINT_X', 'POINT_Y', 'act_start']]
                candidates.loc[:, 'deltaT'] = abs(candidates.loc[:, 'act_start'] - act_start_0)
                try:
                    charging_info = candidates.loc[candidates.deltaT == candidates.deltaT.min(),
                                                   ['act_id', 'deso', 'POINT_X', 'POINT_Y', 'act_start']].to_dict('records')[0]
                    act_id.append(charging_info['act_id'])
                    deso.append(charging_info['deso'])
                    X.append(charging_info['POINT_X'])
                    Y.append(charging_info['POINT_Y'])
                    act_start_ref.append(charging_info['act_start'])
                except:
                    act_id.append(999)
                    deso.append(999)
                    X.append(999)
                    Y.append(999)
                    act_start_ref.append(999)
            data.loc[:, 'act_id'] = act_id
            data.loc[:, 'deso'] = deso
            data.loc[:, 'POINT_X'] = X
            data.loc[:, 'POINT_Y'] = Y
            data.loc[:, 'act_start_ref'] = act_start_ref
            return data

        tqdm.pandas()
        df_parking = df_parking.groupby('person', as_index=False).progress_apply(find_charging_point).\
            reset_index(drop=True)
        print('Remove %.2f percent invalid charging points.'%(len(df_parking.loc[df_parking.act_id == 999, :]) / len(df_parking) * 100))
        df_parking = df_parking.loc[df_parking.act_id != 999, :]
        print('Found charging points in %.2f minutes.' % ((time.time() - start_time) / 60))

        # Get charging time
        start_time = time.time()
        def get_charging_time(row):
            soc_start = max(row['soc'] - row['energy'] / ev.car_dict[row['car']]['battery'], 0.01)
            if int(row['charger']) >= 50:
                soc_end = min(row['soc'], 0.8)
            else:
                soc_end = min(row['soc'], 1)
            time0 = ev.car_dict[row['car']]['sp_dict'][int(row['charger'])]['soc'](soc_start)
            time1 = ev.car_dict[row['car']]['sp_dict'][int(row['charger'])]['soc'](soc_end)
            charging_time = time1 - time0
            return charging_time / 60  # min

        tqdm.pandas()
        df_parking.loc[:, 'charging_time'] = df_parking.progress_apply(lambda row: get_charging_time(row), axis=1)
        print('Got charging time in %.2f minutes.' % ((time.time() - start_time) / 60))

        start_time = time.time()
        df_parking.loc[:, 'paraset'] = paraset
        df_parking.to_sql(name=self.scenario + '_charging_points_5days',
                          schema='sim_statistics',
                          con=engine,
                          index=False,
                          if_exists='append',
                          method='multi', chunksize=10000)
        print('Parking data dumped in %.2f minutes.' % ((time.time() - start_time) / 60))
        return df_parking

    def spatiotemporal_stats(self, df_parking=None, paraset=None):
        start_time = time.time()
        print('Get charging dynamics by DeSO zones...')
        user = dw.keys_manager['database']['user']
        password = dw.keys_manager['database']['password']
        port = dw.keys_manager['database']['port']
        db_name = dw.keys_manager['database']['name']
        engine = sqlalchemy.create_engine(f'postgresql://{user}:{password}@localhost:{port}/{db_name}')
        def minute_charging(data):
            temporal = np.zeros((len(data), 1440))
            ind = 0
            for _, row in data.iterrows():
                act_start = int(row['act_start'])
                charging_time = int(round(row['charging_time']))
                temporal[ind, act_start:(act_start + charging_time)] = 1
                ind += 1
            return pd.Series({'cars': temporal.sum(axis=0).tolist()})

        tqdm.pandas()
        df_charging = df_parking.groupby(['deso', 'purpose', 'charging_type', 'charger']).progress_apply(
            minute_charging).reset_index().explode('cars')
        df_charging.loc[:, 'minute'] = np.linspace(0, 1440, 1440, endpoint=False).tolist() * len(
            df_parking.groupby(['deso', 'purpose', 'charging_type', 'charger']))
        print('Charging dynamics created in %.2f minutes.' % ((time.time() - start_time) / 60))

        start_time = time.time()
        df_charging.loc[:, 'paraset'] = paraset
        df_charging.to_sql(name=self.scenario + '_charging_dynamics_5days',
                          schema='sim_statistics',
                          con=engine,
                          index=False,
                          if_exists='append',
                          method='multi', chunksize=10000)
        print('Charging dynamics dumped in %.2f minutes.' % ((time.time() - start_time) / 60))
        start_time = time.time()
        df_charger = df_charging.groupby(['deso', 'purpose', 'charging_type',
                                          'charger', 'paraset']).apply(lambda x: x.cars.max()).reset_index()
        df_charger.to_sql(name=self.scenario + '_charger_5days',
                           schema='sim_statistics',
                           con=engine,
                           index=False,
                           if_exists='append',
                           method='multi', chunksize=10000)
        print('# of charging points by DeSO zones dumped in %.2f minutes.' % ((time.time() - start_time) / 60))


if __name__ == '__main__':
    stats_aggregator = EVSimStatsSpatiotemporal(scenario='scenario_vg_car')
    stats_aggregator.load_agents_trips()
    for p in [0, 1, 2, 3]:
        print(f'Processing parameter set {p}...')
        df_parking = stats_aggregator.charging_points(paraset=p)
        stats_aggregator.spatiotemporal_stats(df_parking=df_parking, paraset=p)
