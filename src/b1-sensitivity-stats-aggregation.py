import os
import sys
import subprocess
import pandas as pd
import geopandas as gpd
import json
import sqlalchemy
from tqdm import tqdm


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


user = dw.keys_manager['database']['user']
password = dw.keys_manager['database']['password']
port = dw.keys_manager['database']['port']
db_name = dw.keys_manager['database']['name']
engine = sqlalchemy.create_engine(f'postgresql://{user}:{password}@localhost:{port}/{db_name}')


class EVSimStatsAggregation:
    def __init__(self, scenario=None):
        self.scenario = scenario
        self.df_para = None
        self.df_agents = None
        self.df_indi_raw = None
        self.df_indi = None
        self.df_inf_gt = pd.read_excel(ROOT_dir + '/dbs/ev/charging_sites.xlsx')

    def load_agents_parameters(self):
        # Parameters
        file = ROOT_dir + f'/dbs/output_summary/{self.scenario}/ev_sim/parameters.txt'
        list_lines = []
        with open(file) as f:
            for jsonObj in f:
                line = json.loads(jsonObj)
                list_lines.append(line)
        self.df_para = pd.DataFrame(list_lines)

        # Load valid agents
        file = ROOT_dir + f'/dbs/output_summary/{self.scenario}/valid_agents.csv'
        self.df_agents = pd.read_csv(file)

    def save_raw_indi_rec(self):
        df = pd.read_sql(f'''SELECT * FROM sim_statistics.{self.scenario}_individual_5days;''', con=engine)
        self.df_indi_raw = df.loc[:, ['person', 'charging_type', 'home_charger', 'finish_day',
                                 'soc_init', 'soc_end', 'total_distance', 'parking_dur', 'paraset']]
        # Drop duplicated results: charging_type == 3 & paraset is in [2, 3]
        self.df_indi_raw = self.df_indi_raw.loc[~((self.df_indi_raw['charging_type'] == 3) &
                                                  (self.df_indi_raw['paraset'].isin([2, 3]))), :]
        self.df_indi_raw.loc[:, 'residential_charger'] = self.df_indi_raw.apply(
            lambda row: 1 if (row['soc_init'] == 1) & (row['home_charger'] == 0) else 0, axis=1)
        self.df_indi_raw.to_csv(ROOT_dir + f'/results/sensitivity/{self.scenario}_indi_raw_5days.csv', index=False)

    def save_indi_charging(self):
        df = pd.read_sql(f'''SELECT * FROM sim_statistics.{self.scenario}_individual_5days;''', con=engine)
        # Drop duplicated results: charging_type == 3 & paraset is in [2, 3]
        df = df.loc[~((df['charging_type'] == 3) & (df['paraset'].isin([2, 3]))), :]
        df.loc[:, 'charging_time'] = df.loc[:, 'charging_time_fast'] + df.loc[:, 'charging_time_inter']
        df.loc[:, 'charging_time_ratio'] = df.loc[:, 'charging_time'] / df.loc[:, 'parking_dur']
        # Consider home and residential charger access
        self.df_indi_raw.person = self.df_indi_raw.person.astype(str)
        df = df.merge(self.df_indi_raw.loc[:, ['person', 'charging_type', 'paraset', 'residential_charger']],
                      on=['person', 'charging_type', 'paraset'])
        self.df_indi = df.merge(self.df_para, on=['paraset'])
        tqdm.pandas()
        self.df_indi.loc[:, 'Charging_type'] = self.df_indi.progress_apply(
            lambda row: str(int(row['charging_type'])) + ' (' + str(row['type1']) + ')' if row['charging_type'] in (
            1, 2) else str(int(row['charging_type'])) + ' (0.9)', axis=1)
        self.df_indi = self.df_indi.loc[:, ['person', 'car', 'home_charger', 'residential_charger', 'soc_end',
                                            'Charging_type', 'power_intermediate', 'power_fast',
                                            'charging_time_fast', 'charging_time_inter', 'charging_time_ratio',
                                            'charging_energy_fast', 'charging_energy_inter']]
        # Overnight charging energy calculation
        battery_dict = {"B": 40, "C": 60, "D": 100}

        def overnight_energy(row):
            if (row['home_charger'] == 1) | (row['residential_charger'] == 1):
                return battery_dict[row['car']] * (1 - row['soc_end'])
            else:
                return 0

        self.df_indi.loc[:, 'charging_energy_overnight'] = self.df_indi.apply(lambda row: overnight_energy(row), axis=1)
        self.df_indi.to_csv(ROOT_dir + f'/results/sensitivity/{self.scenario}_stats_indi_5days.csv', index=False)

        # Further aggregation
        total = df.person.nunique()

        def aggregation(data):
            num_failure = len(data.loc[data.finish_day == 0, :])
            num_charged = len(data.loc[data.charging_time != 0, :])
            data_charged = data.loc[data.charging_time != 0, :]
            soc_end = data.soc_end.median()
            charging_time_inter = data_charged.charging_time_inter.sum()
            charging_time_fast = data_charged.charging_time_fast.sum()
            charging_time_ratio = data_charged.charging_time_ratio.median()
            energy_fast = data.loc[:, 'charging_energy_fast'].sum()
            energy_inter = data.loc[:, 'charging_energy_inter'].sum()
            return pd.Series(dict(failure_rate=num_failure / total * 100,
                                  charging_share=num_charged / total * 100,
                                  num_charged=num_charged,
                                  soc_end=soc_end,
                                  charging_time_inter=charging_time_inter / num_charged,
                                  charging_time_fast=charging_time_fast / num_charged,
                                  charging_time_ratio=charging_time_ratio,
                                  energy_fast=energy_fast / num_charged,
                                  energy_inter=energy_inter / num_charged,
                                  energy_total=energy_fast + energy_inter))

        df_agg = df.groupby(['paraset', 'charging_type', 'home_charger']).apply(aggregation).reset_index()
        df_agg = df_agg.merge(self.df_para, on=['paraset'])
        df_agg.to_csv(ROOT_dir + f'/results/sensitivity/{self.scenario}_stats_indi_agg_5days.csv', index=False)

    def save_charger(self):
        df = pd.read_sql(f'''SELECT * FROM sim_statistics.{self.scenario}_charger_5days;''', con=engine).rename(
            columns={'0': 'number'})
        df = df.merge(self.df_para, on='paraset')

        tqdm.pandas()
        df.loc[:, 'Charging_type'] = df.progress_apply(
            lambda row: str(int(row['charging_type'])) + ' (' + str(row['type1']) + ')' if row['charging_type'] in (
            1, 2) else str(int(row['charging_type'])) + ' (0.9)', axis=1)
        df.loc[:, 'Purpose'] = df.purpose.progress_apply(lambda x: 'Work' if x == 'work' else 'Other')
        df.charger = df.charger.astype(int)
        df = df.loc[:, ['deso', 'Charging_type', 'Purpose', 'power_fast', 'charger', 'number']]
        df = df.drop_duplicates(subset=['Charging_type', 'Purpose', 'charger', 'power_fast', 'deso'])
        df = df.groupby(['deso', 'Charging_type', 'Purpose', 'power_fast', 'charger'])['number'].sum().reset_index()
        df.to_csv(ROOT_dir + f'/results/sensitivity/{self.scenario}_stats_charger_5days.csv', index=False)

    def save_dynamics(self):
        df = pd.read_sql(f'''SELECT * FROM sim_statistics.{self.scenario}_charging_dynamics_5days;''', con=engine)
        df = df.merge(self.df_para, on='paraset')
        # Drop duplicated results: charging_type == 3 & paraset is in [2, 3]
        df = df.loc[~((df['charging_type'] == 3) & (df['paraset'].isin([2, 3]))), :]
        tqdm.pandas()
        df.loc[:, 'Charging_type'] = df.progress_apply(
            lambda row: str(int(row['charging_type'])) + ' (' + str(row['type1']) + ')' if row['charging_type'] in (
            1, 2) else str(int(row['charging_type'])) + ' (0.9)', axis=1)
        df.loc[:, 'Purpose'] = df.purpose.progress_apply(lambda x: 'Work' if x == 'work' else 'Other')
        df.charger = df.charger.astype(int)
        df = df.loc[:, ['minute', 'Charging_type', 'Purpose', 'power_fast', 'charger', 'cars']]
        df = df.groupby(['minute', 'Charging_type', 'Purpose', 'power_fast', 'charger'])['cars'].sum().reset_index()
        df.loc[:, 'power'] = df.loc[:, 'charger'] * df.loc[:, 'cars']
        df.to_csv(ROOT_dir + f'/results/sensitivity/{self.scenario}_stats_power_5days.csv', index=False)

    def save_inf_comp(self):
        self.df_inf_gt = self.df_inf_gt.loc[:, ['id', 'lat', 'lng', 'pw', 'n']]
        self.df_inf_gt.loc[:, 'fast'] = self.df_inf_gt.apply(lambda row: row['n'] if row['pw'] > 22 else 0, axis=1)
        gdf = dw.df2gdf_point(self.df_inf_gt, 'lng', 'lat', crs=4326, drop=True).to_crs(3006)
        df_sim = pd.read_csv(ROOT_dir + f"/results/sensitivity/{self.scenario}_stats_charger_5days.csv")

        # Add all intermediate and fast power together to compare with today's infrastructure
        df_sim = df_sim.groupby(['deso', 'Charging_type', 'power_fast'])['number'].sum().reset_index()
        zones = gpd.read_file(ROOT_dir + f'/dbs/DeSO/DeSO_2018_v2.shp')
        zones = zones.loc[zones['deso'].isin(df_sim['deso'].unique()), :]
        gdf = gpd.sjoin(gdf, zones.loc[:, ['deso', 'befolkning', 'geometry']])
        self.df_inf_gt = gdf.groupby('deso')['n'].sum().reset_index()
        df = pd.merge(df_sim, self.df_inf_gt, on='deso', how='outer')
        df = df.fillna(0)
        df.columns = ['deso', 'Charging_type', 'power_fast', 'sim', 'gt']

        # Sale up to total VG residents
        df.sim = df.sim * (1 / 0.35)  # The results are based on 35% VG car users
        df.to_csv(ROOT_dir + f'/results/sensitivity/{self.scenario}_inf_comp.csv', index=False)


if __name__ == '__main__':
    stats_aggregator = EVSimStatsAggregation(scenario='scenario_vg_car')
    print("Load agents and parameters...")
    stats_aggregator.load_agents_parameters()

    print("Save raw individual records summary...")
    stats_aggregator.save_raw_indi_rec()

    print("Save individual charging statistics...")
    stats_aggregator.save_indi_charging()

    print("Save infrastructure charging demand...")
    stats_aggregator.save_charger()

    print("Save infrastructure charging dynamics...")
    stats_aggregator.save_dynamics()

    print("Save infrastructure comparison...")
    stats_aggregator.save_inf_comp()
