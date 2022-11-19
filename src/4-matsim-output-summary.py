import os
import sys
import subprocess
from tqdm import tqdm
import pandas as pd
import matsim


def get_repo_root():
    """Get the root directory of the repo."""
    dir_in_repo = os.path.dirname(os.path.abspath('__file__'))
    return subprocess.check_output('git rev-parse --show-toplevel'.split(),
                                   cwd=dir_in_repo,
                                   universal_newlines=True).rstrip()


ROOT_dir = get_repo_root()
sys.path.append(ROOT_dir)
sys.path.insert(0, ROOT_dir + '/lib')

import lib.dataworkers as workers


class ScenarioSummary:
    """
    Read MATSim simulation output files and process.
    """
    def __init__(self, filepath_plans=None, filepath_pop=None):
        self.purpose_dict4input = {1: 'home', 4: 'work', 10: 'other', 6: 'school'}
        self.mode_dict4input = {'Car': 'car', 'CarPassenger': 'car', 'Bike': 'bike',
                                'Walking': 'walk', 'PublicTransport': 'pt'}
        self.df_plan = pd.read_pickle(filepath_plans)
        self.df_pop = pd.read_pickle(filepath_pop)
        self.df_input = None
        self.df_output = None

    def input_processer(self):
        selected_vars = ['PId', 'act_id', 'act_purpose', 'act_start', 'act_end', 'mode',
                          'Deso', 'POINT_X_sweref99', 'POINT_Y_sweref99']
        # Keep only car users
        agents_car_users = self.df_plan.loc[self.df_plan['mode'].isin(['Car', 'CarPassenger']), 'PId'].unique()
        df = self.df_plan.loc[self.df_plan['PId'].isin(agents_car_users), selected_vars]

        # Recode mode and activity purpose
        df.loc[:, 'mode'] = df.loc[:, 'mode'].map(self.mode_dict4input)
        df.loc[:, 'act_purpose'] = df.loc[:, 'act_purpose'].map(self.purpose_dict4input)
        df = df.rename(columns={'POINT_X_sweref99': 'POINT_X', 'POINT_Y_sweref99': 'POINT_Y'})

        tqdm.pandas(desc='Add activity time')
        df.loc[:, 'act_time'] = df.progress_apply(
            lambda row: (row['act_end'] - row['act_start']) * 60 if row['act_end'] > row['act_start'] else \
                (24 - row['act_start'] + row['act_end']) * 60, axis=1)

        # Add dep_time (departure time) to input
        def dep_time_add(data):
            data.loc[:, 'dep_time'] = [0] + list(data.loc[:, 'act_end'].values[:-1])
            return data

        tqdm.pandas(desc='Add departure time')
        self.df_input = df.groupby('PId').progress_apply(dep_time_add).reset_index(drop=True)

        self.df_input.loc[:, 'trav_time'] = 0 # Only for output
        self.df_input.loc[:, 'distance'] = 0 # Only for output
        self.df_input.loc[:, 'speed'] = 0  # Only for output
        self.df_input.loc[:, 'src'] = 'input'
        self.df_input.loc[:, 'score'] = 0
        print(self.df_input.head())

    def output_processer(self, output_file=None, selectedPlansOnly=True):
        # Load output plan
        plans = matsim.plan_reader(output_file, selectedPlansOnly=selectedPlansOnly)
        # Aggregate all individuals' plans
        self.df_output = workers.plans_summary(
            pd.concat([workers.personplan2df(person, plan, experienced=True)
                       for person, plan in
                       tqdm(plans, desc='Processing individual plan')]))

    def merge_input_output(self):
        df_trips = pd.concat([self.df_output, self.df_input])
        df_trips = df_trips.sort_values(by=['PId', 'act_id', 'src'])
        df_trips.loc[:, 'PId'] = df_trips.loc[:, 'PId'].astype(str)
        return df_trips


if __name__ == '__main__':
    run_id = ''
    scenario = 'scenario_vg_car'
    filepath_pop = ROOT_dir + '/dbs/agents/syn_pop_vgr.pkl'
    filepath_plans = ROOT_dir + '/dbs/agents/df_act_plan_vgr.pkl'
    filepath_output = ROOT_dir + f"/dbs/{scenario}/output/{run_id}output_experienced_plans.xml.gz"

    # Load data and input processing
    print("Loading input...")
    ss = ScenarioSummary(filepath_pop=filepath_pop, filepath_plans=filepath_plans)
    ss.input_processer()

    # Load output and merge
    print("Loading output...")
    ss.output_processer(output_file=filepath_output, selectedPlansOnly=True)
    df_trips = ss.merge_input_output()
    df_trips.to_csv(ROOT_dir + f"/dbs/output_summary/{scenario}/{run_id}trips.csv", index=False)
    print("Share of negative activity time trips: %.2f%%" % ((len(df_trips.loc[df_trips['act_time'] <= 0, :]) /
                                                              len(df_trips)) * 100))

    # Input-output stats - simple
    df_stats_simple = pd.pivot_table(df_trips, index=['act_purpose'], columns=['src', 'mode'],
                                     values=['act_time', 'trav_time_min'], aggfunc=['min', 'median', 'max'])
    df_stats_simple.to_csv(ROOT_dir + f"/dbs/output_summary/{scenario}/{run_id}stats_simple.csv")

    # Input-output stats
    df_stats = workers.trips2stats(df_trips)
    df_stats.to_csv(ROOT_dir + f"/dbs/output_summary/{scenario}/{run_id}stats.csv", index=False)
