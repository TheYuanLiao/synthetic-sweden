import os
import sys
import subprocess
import pandas as pd


def get_repo_root():
    """Get the root directory of the repo."""
    dir_in_repo = os.path.dirname(os.path.abspath('__file__'))
    return subprocess.check_output('git rev-parse --show-toplevel'.split(),
                                   cwd=dir_in_repo,
                                   universal_newlines=True).rstrip()


ROOT_dir = get_repo_root()
sys.path.append(ROOT_dir)
sys.path.insert(0, ROOT_dir + '/lib')


class EVChargingSim:
    def __init__(self):
        self.charging_rate = {0.5: 1.75, 0.75: 1.25, 1: 0.75}
        self.power_cap = 120 # kW

    def time2target(self, soc_delta=None, battery_size=None, power=None):
        return battery_size * soc_delta / power * 3600 # seconds

    def charging_time(self, soc_end=None, power=None, battery_size=None):
        boundaries = list(self.charging_rate.keys())
        C0, C1, C2 = tuple(self.charging_rate.values())
        P0, P1, P2 = min(self.power_cap, power * C0), min(self.power_cap, power * C1), min(self.power_cap, power * C2)
        if soc_end <= boundaries[0]:
            t = self.time2target(soc_delta=soc_end, battery_size=battery_size, power=P0)
        elif boundaries[0] < soc_end <= boundaries[1]:
            t = self.time2target(soc_delta=boundaries[0], battery_size=battery_size, power=P0)
            t += self.time2target(soc_delta=soc_end - boundaries[0], battery_size=battery_size, power=P1)
        else:
            t = self.time2target(soc_delta=boundaries[0], battery_size=battery_size, power=P0)
            t += self.time2target(soc_delta=boundaries[1] - boundaries[0], battery_size=battery_size, power=P1)
            t += self.time2target(soc_delta=soc_end - boundaries[1], battery_size=battery_size, power=P2)
        return t


if __name__ == '__main__':
    ev_charging = EVChargingSim()
    power_list = [22, 50, 150]
    battery_size_list = [40, 60, 100]
    soc_end_sim = [x/100 for x in list(range(1, 101))]
    df = pd.DataFrame([(x, y, z) for x in power_list for y in battery_size_list for z in soc_end_sim],
                      columns=['power', 'battery_size', 'soc_end'])
    df = df.loc[~((df['power'].isin([50, 150])) & (df['soc_end'] > 0.8)), :]
    df.loc[:, 'time'] = df.apply(lambda row: ev_charging.charging_time(soc_end=row['soc_end'],
                                                                       power=row['power'],
                                                                       battery_size=row['battery_size']), axis=1)
    df.to_csv(ROOT_dir + '/dbs/ev/charging_dynamics.csv', index=False)