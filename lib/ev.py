import os
import subprocess
import numpy as np
from tqdm import tqdm
import geopandas as gpd
import pandas as pd
import sqlalchemy
import glob
from scipy.stats import skewnorm
from sklearn.preprocessing import minmax_scale


def get_repo_root():
    """Get the root directory of the repo."""
    dir_in_repo = os.path.dirname(os.path.abspath('__file__'))
    return subprocess.check_output('git rev-parse --show-toplevel'.split(),
                                   cwd=dir_in_repo,
                                   universal_newlines=True).rstrip()


import lib.dataworkers as workers

ROOT_dir = get_repo_root()

# Data locations of BEV energy efficiency
car_dict = dict(B=dict(table='CityCarMap.csv', battery=40, speed_max=33.4,
                       sp=workers.load_eff_lookup_table('CityCarMap.csv'), sp_dict=workers.load_charging_table(40)),
                C=dict(table='MidCarMap.csv', battery=60, speed_max=33.5,
                       sp=workers.load_eff_lookup_table('MidCarMap.csv'), sp_dict=workers.load_charging_table(60)),
                D=dict(table='LargeCarMap.csv', battery=100, speed_max=33.5,
                       sp=workers.load_eff_lookup_table('LargeCarMap.csv'), sp_dict=workers.load_charging_table(100)))

# Add stage information to MATSim events output
stage_dict = {x: 0 for x in ['actend', 'departure', 'PersonEntersVehicle', 'vehicle enters traffic']}
stage_dict.update({x: 1 for x in ['left link', 'entered link', 'travelled']})
stage_dict.update({x: 2 for x in ['arrival', 'actstart', 'vehicle leaves traffic', 'PersonLeavesVehicle']})

user = workers.keys_manager['database']['user']
password = workers.keys_manager['database']['password']
port = workers.keys_manager['database']['port']
db_name = workers.keys_manager['database']['name']
engine = sqlalchemy.create_engine(f'postgresql://{user}:{password}@localhost:{port}/{db_name}')


class EVSimulation:
    def __init__(self, scenario=None):
        self.network = None
        self.events = None
        self.agents_num = None
        self.agents = None
        self.scenario = scenario
        self.valid_agents = pd.read_csv(ROOT_dir + f'/dbs/output_summary/{self.scenario}/valid_agents.csv')
        self.valid_agents.loc[:, 'PId'] = self.valid_agents.loc[:, 'PId'].astype(str)
        self.potential_charging = pd.read_csv(f'dbs/output_summary/{self.scenario}/charging_opportunity.csv')
        self.potential_charging = self.potential_charging.sort_values(by=['PId', 'act_id'])
        self.potential_charging = self.potential_charging.loc[(self.potential_charging['mode'] == 'car') |
                                                              (self.potential_charging['act_id'] == 0), :]
        self.events_sim = None

    def load_network(self):
        self.network = gpd.read_file(ROOT_dir + f'/dbs/output_summary/{self.scenario}/volumes_slope.shp')
        self.network.loc[self.network['slope'] < -0.06, 'slope'] = -0.06
        self.network.loc[self.network['slope'] > 0.06, 'slope'] = 0.06
        self.network.loc[:, 'slope'] *= 100

    def load_events(self, df_event=None, valid_agents_filter=True):
        # Further filter out problematic agents who do not move for different activities
        invalid_agents_no_charging = self.potential_charging.loc[(self.potential_charging.trav_time_min < 1) &
                                                                 (self.potential_charging.act_id != 0), 'PId'].astype(
            str).unique()
        print(f'Filter out {len(invalid_agents_no_charging)} invalid agents with zero distances between activities.')

        # Prepare agents of car users
        df_event.loc[:, 'person'] = df_event.loc[:, 'person'].astype(str)
        agents_car = df_event.loc[:, 'person'].unique()
        if valid_agents_filter:
            print("Filtering agents.")
            agents_car = pd.DataFrame(agents_car, columns=['PId'])
            agents_car = agents_car.loc[agents_car['PId'].isin(self.valid_agents.loc[:, 'PId'].values), :]
            agents_car = agents_car.loc[~agents_car['PId'].isin(invalid_agents_no_charging), 'PId'].values
        self.agents_num = len(agents_car)
        print("Focusing on %s car users' daily plans." % self.agents_num)
        self.events = df_event.loc[df_event['person'].isin(agents_car) &
                                   df_event['type'].isin(['vehicle enters traffic',
                                                          'left link',
                                                          'vehicle leaves traffic']), :]
        return agents_car

    def sim_paras_preparation(self, agents_list=None, home_charger=False):
        # Get car information from valid agents
        self.agents = pd.DataFrame()
        self.agents.loc[:, 'person'] = agents_list
        self.agents.loc[:, 'agent_id'] = range(0, self.agents_num)
        self.agents = self.agents.merge(self.valid_agents.loc[:, ['PId', 'car']],
                                        left_on='person', right_on='PId', how='left')
        # Initial SOC
        if home_charger:
            maxValue = 0.9
            df_charger = pd.read_csv(ROOT_dir + f'/dbs/output_summary/home_charger_access.csv')
            df_charger.loc[:, 'PId'] = df_charger.loc[:, 'PId'].astype(str)
            self.agents = self.agents.merge(df_charger.loc[:, ['PId', 'home_charger']],
                                            left_on='person', right_on='PId', how='left')
            no_home_charger_size = self.agents_num - len(self.agents.loc[self.agents['home_charger'] == 1, :])
        else:
            maxValue = 1
            no_home_charger_size = self.agents_num
        minValue = 0.2
        skewness = -4
        soc_list = skewnorm.rvs(a=skewness, size=no_home_charger_size)
        soc_list = minmax_scale(soc_list, (minValue, maxValue))
        if home_charger:
            self.agents.loc[self.agents['home_charger'] == 0, 'soc'] = soc_list
            self.agents.loc[self.agents['home_charger'] == 1, 'soc'] = 1
        else:
            self.agents.loc[:, 'soc'] = soc_list

    def events_processing(self):
        def traj_processing(data):
            def find_speed2calculation(x):
                X = x.values
                return X[0] == X[1]

            data = data.drop_duplicates(subset=['time'], keep='first')
            data.loc[:, 'seq'] = range(0, len(data))
            data.loc[:, 'stage'] = data.loc[:, 'type'].apply(lambda x: stage_dict[x])
            data.loc[:, 'deltaT'] = data.loc[:, 'time'].diff()
            data.loc[:, 'energy_calculate'] = data.loc[:, 'stage'].rolling(window=2).apply(
                lambda x: find_speed2calculation(x))
            data.loc[:, 'energy_calculate'] = data.loc[:, 'energy_calculate'].fillna(0)
            return data

        print('Processing individual trajectories by adding labels for calculating energy etc.')
        tqdm.pandas()
        self.events = self.events.groupby('person').progress_apply(traj_processing).reset_index(drop=True)

        # Get road information
        print('Get road information.')
        self.events = self.events.merge(self.network.loc[:, ['length', 'link_id', 'length_seg', 'slope']],
                                        left_on='link', right_on='link_id', how='left')
        self.events = self.events.sort_values(by=['person', 'time', 'type'], ascending=[True, True, False])

        # Get person information
        print('Get person information.')
        self.events = self.events.merge(self.agents, on='person', how='left')
        print(f'Events with {len(self.events)} rows.')

    def ev_sim_discharging(self, home_charger=False, test=False):
        # Calculate speed and processing
        if test:
            partial_agents = self.events.loc[:, 'person'].unique()
            partial_agents = np.random.choice(partial_agents, size=200)
            self.events = self.events.loc[self.events['person'].isin(partial_agents), :]
        print('Calculating speed and processing.')
        self.events.loc[:, 'speed'] = self.events.loc[:, 'length'] / self.events.loc[:, 'deltaT']
        # Set speed higher than a threshold to threshold value
        tqdm.pandas()
        self.events.loc[:, 'speed'] = self.events.progress_apply(lambda row: car_dict[row['car']]['speed_max'] \
            if row['speed'] > car_dict[row['car']]['speed_max'] else row['speed'], axis=1)
        # Set speed lower than 1 m/s to 1 m/s
        self.events.loc[:, 'speed'] = self.events.loc[:, 'speed'].apply(lambda x: 1 if x < 1 else x)

        # Calculate energy consumption
        print('Calculating energy consumption- road segments.')
        tqdm.pandas()
        self.events.loc[:, 'energy'] = self.events.progress_apply(
            lambda row: car_dict[row['car']]['sp'](row['speed'], row['slope']) * (row['length_seg'] / 1000), axis=1)
        columns2keep = ['person', 'soc', 'car', 'agent_id', 'vehicle', 'time', 'type', 'link',
                        'seq', 'stage', 'energy_calculate', 'length', 'energy']
        if home_charger:
            columns2keep.append('home_charger')
        self.events = self.events.loc[:, columns2keep]

        # Get energy consumption by time
        print('Calculating energy consumption- total and merge.')
        energy = self.events.groupby(['person', 'time'])[['energy']].sum().reset_index()
        # Produce the energy-known set
        events_final = energy.merge(self.events.drop(columns=['energy']),
                                    on=['person', 'time'],
                                    how='inner').drop_duplicates(subset=['person', 'seq'])
        print('Refine energy consumption by checking the label for calculation.')
        tqdm.pandas()
        events_final.loc[:, 'energy'] = events_final.progress_apply(
            lambda row: row['energy'] if row['energy_calculate'] == 1 else 0, axis=1)
        tqdm.pandas()
        events_final.loc[:, 'length'] = events_final.progress_apply(
            lambda row: row['length'] if row['energy_calculate'] == 1 else 0, axis=1)
        events_final.loc[:, 'energy'] = -events_final['energy']
        self.events_sim = events_final

    def traj_energy(self, data=None, soc_threshold=None, min_parking_time=5, fast=150, intermediate=22):
        # soc_threshold is (threshold1, threshold2, threshold3)
        person = data.loc[:, 'person'].values[0]  # String
        battery = data.loc[:, 'car'].apply(lambda x: car_dict[x]['battery'])
        soc_initial = data.loc[:, 'soc'].values[0]
        data.loc[:, 'soc'] = soc_initial + np.cumsum(data.loc[:, 'energy'] / battery)
        data.loc[:, 'distance_driven'] = np.cumsum(data.loc[:, 'length']) / 1000
        data.loc[:, 'deltaT'] = data.loc[:, 'time'].diff()
        # Add activity purpose to where parking happens
        activities = self.potential_charging.loc[self.potential_charging.PId == int(person), :]
        activities = activities.act_purpose.values[1:-1]

        # Prepare for charging
        def getcharging(data):
            stages = data.values
            if ((stages[0], stages[1]) == (2, 0)) | ((stages[0], stages[1]) == (2, 2)):
                return 1
            else:
                return 0

        data.loc[:, 'parking'] = data['stage'].rolling(window=2).apply(getcharging)
        data.loc[data['parking'] == 1, 'purpose'] = activities

        # Get SOC for Type 3 - Event-actuated charging behaviour
        def charging_sim(data, cb=1, min_parking_time=min_parking_time, fast=fast, intermediate=intermediate):
            ev_charger = EVCharging(fast=fast, intermediate=intermediate)
            # type is an integer marking the type of charging behaviour
            # Prepare a place holder of SOC considering charging, with the one without charging to start with
            data.loc[:, f'soc_{cb}'] = data.loc[:, 'soc'].values
            data.loc[:, f'energy_{cb}'] = data.loc[:, 'energy'].values
            # Get potential charging opportunities
            events_charging = data.loc[
                (data['parking'] == 1) & (data['deltaT'] > min_parking_time * 60) & (data['purpose'] != 'home'),
                ['person', 'car', 'time', f'energy_{cb}', f'soc_{cb}', 'deltaT', 'purpose']]
            events_charging = events_charging.to_dict('records')
            num_charging = len(events_charging)

            # Find charging opportunities (Type 3 - parking > 5 min, soc < 0.9, not home)
            c = 0
            while c < num_charging:
                charging_event = events_charging[c]
                if cb == 2:
                    soc_end = data[f'soc_{cb}'].values[-1]
                    if (soc_end < soc_threshold[cb - 1]) & (0.01 <= charging_event[f'soc_{cb}']):
                        charge_flag = True
                    else:
                        charge_flag = False
                else:
                    if 0.01 <= charging_event[f'soc_{cb}'] < soc_threshold[cb - 1]:
                        charge_flag = True
                    else:
                        charge_flag = False
                if charge_flag:
                    charging_event = ev_charger.charging_module(charging_event,
                                                                soc_field=f'soc_{cb}',
                                                                energy_field=f'energy_{cb}')
                    # Add charged energy during parking
                    data.loc[(data['person'] == charging_event['person']) & (
                            data['time'] == charging_event['time']), f'energy_{cb}'] = charging_event[f'energy_{cb}']
                    data.loc[(data['person'] == charging_event['person']) & (
                            data['time'] == charging_event['time']), f'charger_{cb}'] = charging_event['C']
                    # Update soc after charging
                    data.loc[:, f'soc_{cb}'] = soc_initial + np.cumsum(data.loc[:, f'energy_{cb}'] / battery)
                c += 1
                events_charging = data.loc[
                    (data['parking'] == 1) & (data['deltaT'] > min_parking_time * 60) & (data['purpose'] != 'home'),
                    ['person', 'car', 'time', f'energy_{cb}', f'soc_{cb}', 'deltaT', 'purpose']]
                events_charging = events_charging.to_dict('records')
            return data

        for charging_behaviour_type in [1, 2, 3]:
            data = charging_sim(data, cb=charging_behaviour_type, min_parking_time=min_parking_time,
                                fast=fast, intermediate=intermediate)

        return data

    def ev_sim_charging(self, soc_threshold=None, home_charger=False, min_parking_time=5, fast=150, intermediate=22):
        # soc_threshold = 0.9 for Tyoe 3 Event-actuated charging (default)
        # soc_threshold = 0.2 for Type 1 Liquid-fuel strategy
        tqdm.pandas()
        print('Creating individual soc trajectories.')
        events_final = self.events_sim.groupby('person', as_index=False). \
            progress_apply(lambda x: self.traj_energy(data=x,
                                                      soc_threshold=soc_threshold,
                                                      min_parking_time=min_parking_time,
                                                      fast=fast,
                                                      intermediate=intermediate)). \
            reset_index(drop=True)
        columns2keep = ['person', 'purpose', 'car', 'agent_id', 'seq', 'stage', 'vehicle',
                        'time', 'type', 'link', 'distance_driven', 'soc', 'soc_1', 'soc_2', 'soc_3',
                        'energy_1', 'energy_2', 'energy_3', 'charger_1', 'charger_2', 'charger_3']
        if home_charger:
            columns2keep.append('home_charger')
        events_final = events_final.loc[:, columns2keep]
        return events_final

    def merge_results(self):
        # setting the path for joining multiple files
        files = os.path.join(ROOT_dir + f'/dbs/output_summary/{self.scenario}/ev_sim/', "ev_sim_*.csv")
        # list of merged files returned
        files = glob.glob(files)
        # joining files with concat and read_csv
        df = pd.concat(map(pd.read_csv, files), ignore_index=True)
        df.to_csv(ROOT_dir + f'/dbs/output_summary/{self.scenario}/ev_sim/ev_sim.csv', index=False)
        map(os.remove, files)


class EVCharging:
    def __init__(self, fast=None, intermediate=None):
        self.fast = fast
        self.intermediate = intermediate

    def charging_module(self, charging_event=None, soc_field=None, energy_field=None):
        # Initial soc and total parking time
        soc_start = charging_event[soc_field]
        parking_time = charging_event['deltaT']
        battery = car_dict[charging_event['car']]['battery']
        if (parking_time < 30 * 60) & (soc_start < 0.8):
            power = self.fast  # kW
            soc_max = 0.8
        else:
            power = self.intermediate  # kW
            soc_max = 1
        time0 = car_dict[charging_event['car']]['sp_dict'][power]['soc'](soc_start)
        time1 = car_dict[charging_event['car']]['sp_dict'][power]['soc'](soc_max)
        if parking_time + time0 < time1:
            soc_end = car_dict[charging_event['car']]['sp_dict'][power]['time'](parking_time + time0)
        else:
            soc_end = soc_max
        charging_event[energy_field] = (soc_end - soc_start) * battery
        charging_event[soc_field] = soc_end
        charging_event['C'] = power
        return charging_event


class EVSimulationMulti(EVSimulation):
    def __init__(self, scenario=None):
        super().__init__(scenario)
        self.soc_tracker = None

    def initialise_soc_tracker(self):
        self.soc_tracker = []
        df_soc_list = []
        for cb in [1, 2, 3]:
            df_soc = self.agents.loc[:, ['person', 'home_charger', 'soc']]
            df_soc.loc[:, 'resi_charger'] = 0
            df_soc.loc[:, 'charging_type'] = cb
            df_soc.loc[:, 'finish_day'] = 1
            df_soc.loc[:, 'day'] = 0
            df_soc_list.append(df_soc)
        df_soc = pd.concat(df_soc_list)
        self.soc_tracker.append(df_soc)
        # Map values of initial soc
        soc_init = df_soc.groupby('person')[['charging_type', 'soc']] \
            .apply(lambda x: x.set_index('charging_type').to_dict(orient='index')) \
            .to_dict()
        for cb in [1, 2, 3]:
            self.events_sim.loc[:, f'soc_0_{cb}'] = self.events_sim.loc[:, 'person'].apply(
                lambda x: soc_init[x][cb]['soc'])

    def update_soc_init(self, events_final=None, day=2, paraset=1):
        # prepare soc for next simulation day
        if events_final is None:
            df_soc = pd.read_sql(f'''SELECT person, charging_type, soc_end, finish_day 
                                     FROM sim_statistics.scenario_vg_car_individual
                                     WHERE paraset = {paraset};''',
                                 con=engine)
            df_soc = df_soc.rename(columns={'soc_end': 'soc'})
        else:
            df_soc = events_final.groupby('person').progress_apply(lambda data: ev_sim_soc_end(data)).reset_index()
        df_soc = df_soc.merge(self.agents.loc[:, ['person', 'home_charger']], on='person', how='left')
        # For failed cases or soc_end < 0.2 (without home charger), assign residential charger (on-street parking)
        df_soc.loc[:, 'resi_charger'] = df_soc.apply(lambda row: 1 if ((row['finish_day'] == 0) | (row['soc'] < 0.2))
                                                                      & (row['home_charger'] == 0) else 0,
                                                     axis=1)
        # Having access to home charger or residential charger => soc_initial is 1, otherwise, from yesterday
        df_soc.loc[:, 'soc'] = df_soc.apply(
            lambda row: 1 if (row['resi_charger'] == 1) | (row['home_charger'] == 1) else row['soc'], axis=1)
        df_soc.loc[:, 'day'] = day
        self.soc_tracker.append(df_soc)
        # Map values of new initial soc
        soc_init = df_soc.groupby('person')[['charging_type', 'soc']] \
            .apply(lambda x: x.set_index('charging_type').to_dict(orient='index')) \
            .to_dict()
        for cb in [1, 2, 3]:
            self.events_sim.loc[:, f'soc_0_{cb}'] = self.events_sim.loc[:, 'person'].apply(
                lambda x: soc_init[x][cb]['soc'])

    def traj_energy(self, data=None, soc_threshold=None, min_parking_time=5, fast=150, intermediate=22):
        # soc_threshold is (threshold1, threshold2, threshold3)
        person = data.loc[:, 'person'].values[0]  # String
        # print(person)
        battery = data.loc[:, 'car'].apply(lambda x: car_dict[x]['battery'])
        data.loc[:, 'distance_driven'] = np.cumsum(data.loc[:, 'length']) / 1000
        data.loc[:, 'deltaT'] = data.loc[:, 'time'].diff()
        # Add activity purpose to where parking happens
        activities = self.potential_charging.loc[self.potential_charging.PId == int(person), :]
        activities = activities.act_purpose.values[1:-1]

        # Prepare for charging
        def getcharging(data):
            stages = data.values
            if ((stages[0], stages[1]) == (2, 0)) | ((stages[0], stages[1]) == (2, 2)):
                return 1
            else:
                return 0

        data.loc[:, 'parking'] = data['stage'].rolling(window=2).apply(getcharging)
        data.loc[data['parking'] == 1, 'purpose'] = activities

        # Get SOC for Type 3 - Event-actuated charging behaviour
        def charging_sim(data, cb=1, min_parking_time=min_parking_time, fast=fast, intermediate=intermediate):
            ev_charger = EVCharging(fast=fast, intermediate=intermediate)
            # type is an integer marking the type of charging behaviour
            # Prepare a place holder of SOC considering charging, with the one without charging to start with
            soc_initial = data.loc[:, f'soc_0_{cb}'].values[0]
            data.loc[:, f'soc_0_{cb}'] = soc_initial + np.cumsum(data.loc[:, 'energy'] / battery)
            data.loc[:, f'soc_{cb}'] = data.loc[:, f'soc_0_{cb}'].values
            data.loc[:, f'energy_{cb}'] = data.loc[:, 'energy'].values
            # Get potential charging opportunities
            events_charging = data.loc[
                (data['parking'] == 1) & (data['deltaT'] > min_parking_time * 60) & (data['purpose'] != 'home'),
                ['person', 'car', 'time', f'energy_{cb}', f'soc_{cb}', 'deltaT', 'purpose']]
            events_charging = events_charging.to_dict('records')
            num_charging = len(events_charging)

            # Find charging opportunities (Type 3 - parking > 5 min, soc < 0.9, not home)
            c = 0
            while c < num_charging:
                charging_event = events_charging[c]
                if cb == 2:
                    soc_end = data[f'soc_{cb}'].values[-1]
                    if (soc_end < soc_threshold[cb - 1]) & (0.01 <= charging_event[f'soc_{cb}']):
                        charge_flag = True
                    else:
                        charge_flag = False
                else:
                    if 0.01 <= charging_event[f'soc_{cb}'] < soc_threshold[cb - 1]:
                        charge_flag = True
                    else:
                        charge_flag = False
                if charge_flag:
                    charging_event = ev_charger.charging_module(charging_event,
                                                                soc_field=f'soc_{cb}',
                                                                energy_field=f'energy_{cb}')
                    # Add charged energy during parking
                    data.loc[(data['person'] == charging_event['person']) & (
                            data['time'] == charging_event['time']), f'energy_{cb}'] = charging_event[f'energy_{cb}']
                    data.loc[(data['person'] == charging_event['person']) & (
                            data['time'] == charging_event['time']), f'charger_{cb}'] = charging_event['C']
                    # Update soc after charging
                    data.loc[:, f'soc_{cb}'] = soc_initial + np.cumsum(data.loc[:, f'energy_{cb}'] / battery)
                c += 1
                events_charging = data.loc[
                    (data['parking'] == 1) & (data['deltaT'] > min_parking_time * 60) & (data['purpose'] != 'home'),
                    ['person', 'car', 'time', f'energy_{cb}', f'soc_{cb}', 'deltaT', 'purpose']]
                events_charging = events_charging.to_dict('records')
            return data

        for charging_behaviour_type in [1, 2, 3]:
            data = charging_sim(data, cb=charging_behaviour_type, min_parking_time=min_parking_time,
                                fast=fast, intermediate=intermediate)

        return data

    def ev_sim_charging(self, soc_threshold=None, home_charger=False, min_parking_time=5, fast=150, intermediate=22):
        # Default: soc_threshold = (0.2, 0.5, 0.9)
        # soc_threshold = 0.9 for Tyoe 3 Event-actuated charging (default)
        # soc_threshold = 0.2 for Type 1 Liquid-fuel strategy
        tqdm.pandas()
        print('Creating individual soc trajectories.')
        events_final = self.events_sim.groupby('person', as_index=False). \
            progress_apply(lambda x: self.traj_energy(data=x,
                                                      soc_threshold=soc_threshold,
                                                      min_parking_time=min_parking_time,
                                                      fast=fast,
                                                      intermediate=intermediate)). \
            reset_index(drop=True)
        columns2keep = ['person', 'purpose', 'car', 'agent_id', 'seq', 'stage', 'vehicle',
                        'time', 'type', 'link', 'distance_driven',
                        'soc_0_1', 'soc_0_2', 'soc_0_3', 'soc_1', 'soc_2', 'soc_3',
                        'energy_1', 'energy_2', 'energy_3', 'charger_1', 'charger_2', 'charger_3']
        if home_charger:
            columns2keep.append('home_charger')
        events_final = events_final.loc[:, columns2keep]
        return events_final


def ev_sim_soc_end(data):
    df_example = data.copy()
    # Preprocessing the simulation trajectories
    df_example = df_example.sort_values(by=['seq']).reset_index(drop=True)
    soc_end = df_example.iloc[-1][['soc_1', 'soc_2', 'soc_3']].to_list()
    finish_day = [1 if x > 0 else 0 for x in soc_end]
    charging_type = [1, 2, 3]
    # Aggregate results
    stats_dict = dict(charging_type=charging_type,
                      soc=soc_end,
                      finish_day=finish_day
                      )
    df_stats = pd.DataFrame(stats_dict)
    return df_stats


def ev_sim_stats(data, paraset=None, df_para=None):
    power_inter = df_para.loc[df_para['paraset'] == paraset, 'power_intermediate'].values[0]
    power_fast = df_para.loc[df_para['paraset'] == paraset, 'power_fast'].values[0]
    df_example = data.copy()
    # Preprocessing the simulation trajectories
    df_example = df_example.sort_values(by=['seq']).reset_index(drop=True)
    df_example.loc[:, 'deltaT'] = df_example.loc[:, 'time'].diff()
    for i in range(1, 4):
        df_example.loc[:, f'deltaSoc_{i}'] = df_example.loc[:, f'soc_{i}'].diff()

    # Get basic information
    soc_init = df_example.loc[0, 'soc']
    car = df_example.loc[0, 'car']
    total_distance = df_example.loc[:, 'distance_driven'].max()
    home_charger = df_example.loc[0, 'home_charger']

    # Get parking events
    parking_data = df_example.loc[~df_example['purpose'].isnull(), :]
    parking_dur = parking_data.loc[:, 'deltaT'].sum() / 60  # min
    parking_freq = len(parking_data)

    # Prepare charging types data
    charging_type = [1, 2, 3, 0]
    charging_threshold = df_para.loc[df_para['paraset'] == paraset, ['type1', 'type2', 'type3']].values.tolist()[0] + [
        0]
    soc_end = df_example.iloc[-1][['soc_1', 'soc_2', 'soc_3', 'soc']].to_list()
    finish_day = [1 if x > 0 else 0 for x in soc_end]

    # Get charging stats
    charging_energy_fast = []
    charging_energy_inter = []
    charging_time_fast = []
    charging_time_inter = []
    for i in [1, 2, 3]:
        if parking_data.loc[:, f'energy_{i}'].sum() == 0:
            charging_energy_fast.append(0)
            charging_energy_inter.append(0)
            charging_time_fast.append(0)
            charging_time_inter.append(0)
        else:
            energy_fast = 0
            energy_inter = 0
            time_fast = 0
            time_inter = 0
            for _, row in parking_data.iterrows():
                if row[f'charger_{i}'] == power_inter:
                    soc_start = max(row[f'soc_{i}'] - row[f'deltaSoc_{i}'], 0.01)
                    soc_end_ = min(row[f'soc_{i}'], 1)
                    energy_inter += row[f'energy_{i}']
                    time0 = car_dict[car]['sp_dict'][power_inter]['soc'](soc_start)
                    time1 = car_dict[car]['sp_dict'][power_inter]['soc'](soc_end_)
                    time_inter = time_inter + time1 - time0
                if row[f'charger_{i}'] == power_fast:
                    soc_start = max(row[f'soc_{i}'] - row[f'deltaSoc_{i}'], 0.01)
                    soc_end_ = min(row[f'soc_{i}'], 0.8)
                    energy_fast += row[f'energy_{i}']
                    time0 = car_dict[car]['sp_dict'][power_fast]['soc'](soc_start)
                    time1 = car_dict[car]['sp_dict'][power_fast]['soc'](soc_end_)
                    time_fast = time_fast + time1 - time0
            charging_energy_fast.append(energy_fast)
            charging_energy_inter.append(energy_inter)
            charging_time_fast.append(time_fast / 60)
            charging_time_inter.append(time_inter / 60)
    charging_energy_fast.append(0)
    charging_energy_inter.append(0)
    charging_time_fast.append(0)
    charging_time_inter.append(0)

    # Aggregate results
    stats_dict = dict(charging_type=charging_type,
                      soc_end=soc_end,
                      finish_day=finish_day,
                      charging_threshold=charging_threshold,
                      charging_energy_fast=charging_energy_fast,
                      charging_time_fast=charging_time_fast,
                      charging_energy_inter=charging_energy_inter,
                      charging_time_inter=charging_time_inter
                      )
    df_stats = pd.DataFrame(stats_dict)
    df_stats.loc[:, 'soc_init'] = soc_init
    df_stats.loc[:, 'car'] = car
    df_stats.loc[:, 'total_distance'] = total_distance
    df_stats.loc[:, 'home_charger'] = home_charger
    df_stats.loc[:, 'parking_dur'] = parking_dur
    df_stats.loc[:, 'parking_freq'] = parking_freq
    df_stats.loc[:, 'paraset'] = paraset
    return df_stats

def ev_sim_multi_stats(data, paraset=None, df_para=None):
    power_inter = df_para.loc[df_para['paraset'] == paraset, 'power_intermediate'].values[0]
    power_fast = df_para.loc[df_para['paraset'] == paraset, 'power_fast'].values[0]
    df_example = data.copy()
    # Preprocessing the simulation trajectories
    df_example = df_example.sort_values(by=['seq']).reset_index(drop=True)
    df_example.loc[:, 'deltaT'] = df_example.loc[:, 'time'].diff()
    for i in range(1, 4):
        df_example.loc[:, f'deltaSoc_{i}'] = df_example.loc[:, f'soc_{i}'].diff()

    # Get basic information
    soc_init = [df_example.loc[0, 'soc_0_1'], df_example.loc[0, 'soc_0_2'], df_example.loc[0, 'soc_0_3']]
    car = df_example.loc[0, 'car']
    total_distance = df_example.loc[:, 'distance_driven'].max()
    home_charger = df_example.loc[0, 'home_charger']

    # Get parking events
    parking_data = df_example.loc[~df_example['purpose'].isnull(), :]
    parking_dur = parking_data.loc[:, 'deltaT'].sum() / 60  # min
    parking_freq = len(parking_data)

    # Prepare charging types data
    charging_type = [1, 2, 3]
    charging_threshold = df_para.loc[df_para['paraset'] == paraset, ['type1', 'type2', 'type3']].values.tolist()[0]
    soc_end = df_example.iloc[-1][['soc_1', 'soc_2', 'soc_3']].to_list()
    finish_day = [1 if x > 0 else 0 for x in soc_end]

    # Get charging stats
    charging_energy_fast = []
    charging_energy_inter = []
    charging_time_fast = []
    charging_time_inter = []
    for i in [1, 2, 3]:
        if parking_data.loc[:, f'energy_{i}'].sum() == 0:
            charging_energy_fast.append(0)
            charging_energy_inter.append(0)
            charging_time_fast.append(0)
            charging_time_inter.append(0)
        else:
            energy_fast = 0
            energy_inter = 0
            time_fast = 0
            time_inter = 0
            for _, row in parking_data.iterrows():
                if row[f'charger_{i}'] == power_inter:
                    soc_start = max(row[f'soc_{i}'] - row[f'deltaSoc_{i}'], 0.01)
                    soc_end_ = min(row[f'soc_{i}'], 1)
                    energy_inter += row[f'energy_{i}']
                    time0 = car_dict[car]['sp_dict'][power_inter]['soc'](soc_start)
                    time1 = car_dict[car]['sp_dict'][power_inter]['soc'](soc_end_)
                    time_inter = time_inter + time1 - time0
                if row[f'charger_{i}'] == power_fast:
                    soc_start = max(row[f'soc_{i}'] - row[f'deltaSoc_{i}'], 0.01)
                    soc_end_ = min(row[f'soc_{i}'], 0.8)
                    energy_fast += row[f'energy_{i}']
                    time0 = car_dict[car]['sp_dict'][power_fast]['soc'](soc_start)
                    time1 = car_dict[car]['sp_dict'][power_fast]['soc'](soc_end_)
                    time_fast = time_fast + time1 - time0
            charging_energy_fast.append(energy_fast)
            charging_energy_inter.append(energy_inter)
            charging_time_fast.append(time_fast / 60)
            charging_time_inter.append(time_inter / 60)

    # Aggregate results
    stats_dict = dict(charging_type=charging_type,
                      soc_init=soc_init,
                      soc_end=soc_end,
                      finish_day=finish_day,
                      charging_threshold=charging_threshold,
                      charging_energy_fast=charging_energy_fast,
                      charging_time_fast=charging_time_fast,
                      charging_energy_inter=charging_energy_inter,
                      charging_time_inter=charging_time_inter
                      )
    df_stats = pd.DataFrame(stats_dict)
    df_stats.loc[:, 'car'] = car
    df_stats.loc[:, 'total_distance'] = total_distance
    df_stats.loc[:, 'home_charger'] = home_charger
    df_stats.loc[:, 'parking_dur'] = parking_dur
    df_stats.loc[:, 'parking_freq'] = parking_freq
    df_stats.loc[:, 'paraset'] = paraset
    return df_stats
