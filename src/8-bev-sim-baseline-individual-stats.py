import os
import sys
import subprocess
import pandas as pd
import sqlalchemy
from tqdm import tqdm
import json
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


class EVSimStats:
    def __init__(self, scenario=None):
        self.scenario = scenario
        self.df_para = None
        self.df_valid_agents = None

    def load_parameters(self):
        file = ROOT_dir + f'/dbs/output_summary/{self.scenario}/ev_sim/parameters.txt'
        list_lines = []
        with open(file) as f:
            for jsonObj in f:
                line = json.loads(jsonObj)
                list_lines.append(line)
        self.df_para = pd.DataFrame(list_lines)

    def load_agents(self):
        file = ROOT_dir + f'/dbs/output_summary/{self.scenario}/valid_agents.csv'
        self.df_valid_agents = pd.read_csv(file)

    def stats(self, paraset=None):
        user = dw.keys_manager['database']['user']
        password = dw.keys_manager['database']['password']
        port = dw.keys_manager['database']['port']
        db_name = dw.keys_manager['database']['name']
        engine = sqlalchemy.create_engine(f'postgresql://{user}:{password}@localhost:{port}/{db_name}')
        vars = ['person', 'purpose', 'car', 'seq', 'time', 'distance_driven', 'soc', 'home_charger'] + \
               ['soc_' + str(x) for x in (1, 2, 3)] +\
               ['energy_' + str(x) for x in (1, 2, 3)] +\
               ['charger_' + str(x) for x in (1, 2, 3)]
        vars_sql = ','.join(vars)
        start = time.time()
        df = pd.read_sql(f'''SELECT {vars_sql} FROM scenario_vg_car_ev_sim_paraset{paraset};''',
                         con=engine)
        epoch = (time.time() - start)/60
        print(f'Data loaded in {epoch} minutes.')
        print(f'Calculating individual statistics for parameter set {paraset}.')
        tqdm.pandas()
        df_stats = df.groupby('person').progress_apply(lambda data: ev.ev_sim_stats(data,
                                                                                    paraset=paraset,
                                                                                    df_para=self.df_para)).reset_index()
        df_stats.to_sql(name=self.scenario + '_individual',
                        schema='sim_statistics',
                        con=engine,
                        index=False,
                        if_exists='append',
                        method='multi', chunksize=10000)


if __name__ == '__main__':
    stats_aggregator = EVSimStats(scenario='scenario_vg_car')
    stats_aggregator.load_agents()
    stats_aggregator.load_parameters()
    for paraset in range(0, 4):
        print(f'Processing parameter set {paraset}...')
        stats_aggregator.stats(paraset=paraset)
