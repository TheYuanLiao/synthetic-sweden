import os
import sys
import subprocess
import pandas as pd
import sqlalchemy
import pprint
import json


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


class EVBatchSim:
    def __init__(self, scenario=None, min_parking_time=5, intermediate=22):
        user = dw.keys_manager['database']['user']
        password = dw.keys_manager['database']['password']
        port = dw.keys_manager['database']['port']
        db_name = dw.keys_manager['database']['name']
        engine = sqlalchemy.create_engine(f'postgresql://{user}:{password}@localhost:{port}/{db_name}')
        self.scenario = scenario
        self.engine = engine
        self.min_parking_time = min_parking_time
        self.intermediate = intermediate
        self.simulation = ev.EVSimulation(scenario=scenario)
        self.simulation.load_network()
        print("Network loaded.")

    def load_events_and_discharging(self, batch=None):
        print(f'Processing batch {batch}...')
        df_event = pd.read_csv(ROOT_dir + f'/dbs/output_summary/{scenario}/ev_sim/events_batch{batch}.csv.gz',
                               compression='gzip')
        agents_car = self.simulation.load_events(df_event=df_event, valid_agents_filter=True)
        print(f"Events batch {batch} loaded.")
        self.simulation.sim_paras_preparation(agents_list=agents_car, home_charger=True)
        print("Agents parameters loaded.")
        self.simulation.events_processing()
        print(f"Events batch {batch} processed.")
        self.simulation.ev_sim_discharging(home_charger=True, test=False)
        print(f"Events batch {batch} discharging simulated.")

    def simulate_charging(self, paraset=None, soc_threshold=None, fast=None):
        events_final = self.simulation.ev_sim_charging(soc_threshold=soc_threshold,
                                                       fast=fast,
                                                       min_parking_time=self.min_parking_time,
                                                       home_charger=True,
                                                       intermediate=self.intermediate)
        print("Charging simulated.")
        # Dump to database
        events_final.to_sql(scenario + f'_ev_sim_paraset{paraset}', self.engine, index=False,
                            if_exists='append', method='multi', chunksize=10000)
        print(f"EV simulated saved.")

    def simulate_charging_multiple(self, paraset_list=None):
        for para in paraset_list:
            self.simulate_charging(paraset=para[0], soc_threshold=para[1], fast=para[2])


if __name__ == '__main__':
    scenario = 'scenario_vg_car'
    para_path = ROOT_dir + f'/dbs/output_summary/{scenario}/ev_sim/parameters.txt'
    min_parking_time = 5 # min
    intermediate = 22 # kW
    sim = EVBatchSim(scenario=scenario, min_parking_time=5, intermediate=22)

    parasets = [0, 1, 2, 3]
    soc_thresholds = [(0.2, 0.2, 0.9), (0.3, 0.3, 0.9)]
    fast_powers = [150, 50]
    paraset_list = [(x, y, z) for x, (y, z) in zip(parasets,
                                                   [(soc_thr, fast) for soc_thr in soc_thresholds
                                                    for fast in fast_powers])]
    # Store parameters used for simulation
    for para in paraset_list:
        dic = {'paraset': para[0],
               'type1': para[1][0],
               'type2': para[1][1],
               'type3': para[1][2],
               'power_fast': para[2],
               'min_parking_time': min_parking_time,
               'power_intermediate': intermediate}
        pprint.pprint(dic)
        with open(para_path, 'a') as outfile:
            json.dump(dic, outfile)
            outfile.write('\n')

    # Start simulation
    batch_num = 20
    for batch in range(0, batch_num):
        sim.load_events_and_discharging(batch=batch)
        sim.simulate_charging_multiple(paraset_list=paraset_list)
