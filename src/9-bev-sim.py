import os
import sys
import subprocess
import pandas as pd
import sqlalchemy


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
        self.simulation = ev.EVSimulationMulti(scenario=scenario)
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

    def simulate_charging(self, paraset=None, soc_threshold=None, fast=None, write=False):
        events_final = self.simulation.ev_sim_charging(soc_threshold=soc_threshold,
                                                       fast=fast,
                                                       min_parking_time=self.min_parking_time,
                                                       home_charger=True,
                                                       intermediate=self.intermediate)
        print("Charging simulated.")
        if write:
            # Dump to database
            events_final.to_sql(scenario + f'_ev_sim_paraset{paraset}_5days', self.engine, index=False,
                                if_exists='append', method='multi', chunksize=10000)
            # events_final.to_csv(f'dbs/output_summary/{scenario}/ev_sim/ev_sim_{batch}.csv', index=False)
            print(f"EV simulated saved.")
        return events_final

    def track_soc(self, paraset=None):
        valid_agents = sim.simulation.soc_tracker[1].loc[:, 'person'].unique()
        df_soc = pd.concat([x.loc[x.person.isin(valid_agents), :] for x in sim.simulation.soc_tracker])
        df_soc = df_soc.sort_values(by=['person', 'day', 'charging_type'])
        df_soc = df_soc.drop(columns=['level_1'])
        df_soc.to_sql(scenario + f'_ev_sim_paraset{paraset}_5days_soc', self.engine, index=False,
                      if_exists='append', method='multi', chunksize=10000)


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
    for paraset in [0, 1, 2, 3]:
        print(f"Working on paraset = {paraset}")
        para = paraset_list[paraset]
        # Start simulation
        batch_num = 20
        for batch in range(0, batch_num):
            sim.load_events_and_discharging(batch=batch)
            # Baseline day
            sim.simulation.initialise_soc_tracker()
            events_final = sim.simulate_charging(paraset=para[0], soc_threshold=para[1], fast=para[2], write=False)
            # Day 1-4
            for day in [1, 2, 3, 4]:
                sim.simulation.update_soc_init(events_final=events_final, day=day, paraset=paraset)
                events_final = sim.simulate_charging(paraset=para[0], soc_threshold=para[1], fast=para[2], write=False)
            # Day 5
            sim.simulation.update_soc_init(events_final=events_final, day=5, paraset=paraset)
            events_final = sim.simulate_charging(paraset=para[0], soc_threshold=para[1], fast=para[2], write=True)
            # Log soc traj over multiple days (informing resi_charger demand)
            sim.track_soc(paraset=paraset)
