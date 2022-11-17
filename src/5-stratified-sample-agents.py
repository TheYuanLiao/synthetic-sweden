import os
import sys
import subprocess
from tqdm import tqdm
import pandas as pd
import random


def get_repo_root():
    """Get the root directory of the repo."""
    dir_in_repo = os.path.dirname(os.path.abspath('__file__'))
    return subprocess.check_output('git rev-parse --show-toplevel'.split(),
                                   cwd=dir_in_repo,
                                   universal_newlines=True).rstrip()


ROOT_dir = get_repo_root()
sys.path.append(ROOT_dir)
sys.path.insert(0, ROOT_dir + '/lib')


if __name__ == '__main__':
    scenario = 'scenario_vg_car'
    file_path2output = ROOT_dir + f"/dbs/output_summary/{scenario}/"
    df = pd.read_csv(file_path2output + "trips.csv")
    df_output = df.loc[df['src'] == 'output', :]
    df_car_users = df.loc[df['src'] == 'input', ['PId', 'Deso']].drop_duplicates(subset=['PId'])
    df_deso = df_car_users.groupby('Deso').count().reset_index().rename(columns={'Deso': 'deso', 'PId': 'befolkning'})

    # Remove agents with infeasible activity plans
    invalid_agents = df_output.loc[df_output['act_time'] <= 0, 'PId'].unique()
    valid_agents = df_output.loc[~df_output['PId'].isin(invalid_agents), 'PId'].unique()
    df_valid_agents = df.loc[(df['src'] == 'input') & (df['PId'].isin(valid_agents)), ['PId', 'Deso']].drop_duplicates(
        subset=['PId'])

    # Count agent number per deso zone and compare with the true population size
    deso_share = df_valid_agents.groupby('Deso').count().reset_index().rename(
        columns={'Deso': 'deso', 'PId': 'pop_sim'})
    deso_share = deso_share.merge(df_deso, on='deso', how='left')
    deso_share.loc[:, 'share'] = deso_share.loc[:, 'pop_sim'] / deso_share.loc[:, 'befolkning']

    # Remove border deso zones with very small number of agents
    deso_share = deso_share.loc[deso_share['pop_sim'] > 2, :]

    # Create the number of agents to draw from simulated valid agents to
    # make sure they are representative of true population in each deso zone
    print("The resample share is bounded by the minimum share of deso populations: %.2f" % deso_share.share.min())
    share2keep = deso_share.share.min()
    deso_share.loc[:, 'resampled'] = round(deso_share.loc[:, 'befolkning'] * share2keep).astype(int)
    deso_share_dict = dict(zip(deso_share.deso, deso_share.resampled))
    random_seed = 3


    def sample_pop_deso(data):
        deso_code = data['Deso'].values[0]
        return data.sample(n=deso_share_dict[deso_code], random_state=random_seed)


    tqdm.pandas('Resample simulated agents to represent 35% population of each DeSO zone')
    df_valid_agents_resampled = df_valid_agents.loc[df_valid_agents['Deso'].isin(deso_share['deso']), :].groupby(
        'Deso').progress_apply(sample_pop_deso).reset_index(drop=True)
    print(f'Number of car agents left: %s'%len(df_valid_agents_resampled))

    # Load population statistics
    df_pop = pd.read_pickle(ROOT_dir + f'/dbs/agents/syn_pop_vgr.pkl')
    selected_vars = ['PId', 'income_class', 'HId', 'marital']
    df_pop = df_pop.loc[:, selected_vars]

    # Within each household, decide the car user's income level
    def income4car(data):
        if 'child' not in data.marital.values:
            data.loc[:, 'income4car'] = data['income_class']
        else:
            incomes = data.loc[data.marital != 'child', 'income_class'].values
            data.loc[data.marital != 'child', 'income4car'] = data.loc[data.marital != 'child', 'income_class']
            L = len(data.loc[data.marital == 'child', 'income4car'])
            data.loc[data.marital == 'child', 'income4car'] = [incomes[random.randint(0, len(incomes) - 1)] for l in
                                                               range(0, L)]
        return data


    tqdm.pandas()
    df_pop = df_pop.groupby('HId').progress_apply(income4car).reset_index(drop=True)

    # Enrichment with income information
    df_valid_agents_resampled = df_valid_agents_resampled.merge(df_pop.loc[:, ['PId', 'income4car']],
                                                                left_on='PId', right_on='PId', how='left')
    df_valid_agents_resampled.loc[:, 'income4car'] = df_valid_agents_resampled.loc[:, 'income4car'].astype(int)

    # Read car fleet assignment according to income
    df_valid_agents_carfleet = pd.read_csv(f'dbs/output_summary/{scenario}/valid_agents_car_fleet.csv')
    df_valid_agents_resampled.loc[:, 'car'] = ''
    for var in range(0, 5):
        # Car fleet
        composition = df_valid_agents_carfleet.loc[df_valid_agents_carfleet.income4car == var, :].values
        car_types = ['B'] * composition[0, 1] + ['C'] * composition[0, 2] + ['D'] * composition[0, 3]
        random.Random(random_seed).shuffle(car_types)
        df_valid_agents_resampled.loc[df_valid_agents_resampled.loc[:, 'income4car'] == var, 'car'] = car_types

    # Save valid agents with assigned car type for further BEV simulation
    df_valid_agents_resampled.to_csv(f'dbs/output_summary/{scenario}/valid_agents.csv', index=False)
