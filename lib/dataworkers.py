import os
import subprocess
from geopandas import GeoDataFrame
from shapely.geometry import Point
from math import radians, cos, sin, asin, sqrt
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import mapping
import matsim
import random
import gzip
import scipy.interpolate
from tqdm import tqdm
import yaml
import sqlalchemy


def get_repo_root():
    """Get the root directory of the repo."""
    dir_in_repo = os.path.dirname(os.path.abspath('__file__'))
    return subprocess.check_output('git rev-parse --show-toplevel'.split(),
                                   cwd=dir_in_repo,
                                   universal_newlines=True).rstrip()


ROOT_dir = get_repo_root()
purpose_dict = {'h': 'home', 'o': 'other', 'w': 'work', 's': 'school'}
mode_dict = {'Car': 'car', 'CarPassenger': 'car', 'Bike': 'bike',
             'Walking': 'walk', 'PublicTransport': 'pt'}
activity_purpose_dict = {1: 'h', 4: 'w', 10: 'o', 6: 's'}
with open(os.path.join(ROOT_dir, 'dbs', 'keys.yaml')) as f:
    keys_manager = yaml.load(f, Loader=yaml.FullLoader)


def df2gdf_point(df, x_field, y_field, crs=4326, drop=True):
    """
    Convert two columns of GPS coordinates into POINT geo dataframe
    :param drop: boolean, if true, x and y columns will be dropped
    :param df: dataframe, containing X and Y
    :param x_field: string, col name of X
    :param y_field: string, col name of Y
    :param crs: int, epsg code
    :return: a geo dataframe with geometry of POINT
    """
    geometry = [Point(xy) for xy in zip(df[x_field], df[y_field])]
    if drop:
        gdf = GeoDataFrame(df.drop(columns=[x_field, y_field]), geometry=geometry)
    else:
        gdf = GeoDataFrame(df, crs=crs, geometry=geometry)
    gdf.set_crs(epsg=crs, inplace=True)
    return gdf


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees) in km
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    return c * r


def personplan2df(person, plan, output=True, experienced=False):
    """
    Convert a person's activity plan from MATSim format to a dataframe.
    :param person: matsim.plan_reader() object
    :param plan: matsim.plan_reader() object
    :param output: whether read in output file
    :param experienced: whether read in experienced plans
    :return: a re-organised individual activity plan executed by matsim (dataframe)
    """
    pid = person.attrib['id']

    activities = filter(lambda x: x.tag == 'activity', plan)
    types = [activity.attrib['type'] for activity in activities]
    activities = filter(lambda x: x.tag == 'activity', plan)
    end_times = []
    for activity in activities:
        try:
            end_times.append(activity.attrib['end_time'])
        except:
            end_times.append('23:59:59')
    if experienced:
        activities = filter(lambda x: x.tag == 'activity', plan)
        xs = [0]
        count = 0
        for activity in activities:
            if count != 0:
                xs.append(activity.attrib['x'])
            count += 1
        activities = filter(lambda x: x.tag == 'activity', plan)
        ys = [0]
        count = 0
        for activity in activities:
            if count != 0:
                ys.append(activity.attrib['y'])
            count += 1
    else:
        activities = filter(lambda x: x.tag == 'activity', plan)
        xs = [activity.attrib['x'] for activity in activities]
        activities = filter(lambda x: x.tag == 'activity', plan)
        ys = [activity.attrib['y'] for activity in activities]

    legs = filter(lambda x: x.tag == 'leg', plan)
    modes = [leg.attrib['mode'] for leg in legs]
    modes = [''] + modes

    df = pd.DataFrame()
    df.loc[:, 'act_purpose'] = types
    df.loc[:, 'PId'] = pid
    df.loc[:, 'act_end'] = end_times
    df.loc[:, 'act_id'] = range(0, len(types))
    df.loc[:, 'mode'] = modes
    df.loc[:, 'POINT_X'] = xs
    df.loc[:, 'POINT_Y'] = ys

    if output:
        legs = filter(lambda x: x.tag == 'leg', plan)
        trav_times = [leg.attrib['trav_time'] for leg in legs]
        trav_times = ["00:00:00"] + trav_times

        legs = filter(lambda x: x.tag == 'leg', plan)
        dep_times = [leg.attrib['dep_time'] for leg in legs]
        dep_times = ["00:00:00"] + dep_times

        legs = filter(lambda x: x.tag == 'leg', plan)
        distances = [0]
        for leg in legs:
            routes = filter(lambda x: x.tag == 'route', leg)
            distances += [float(route.attrib['distance']) / 1000 for route in routes]

        df.loc[:, 'dep_time'] = dep_times
        df.loc[:, 'trav_time'] = trav_times
        df.loc[:, 'distance'] = distances
        df.loc[:, 'score'] = float(plan.attrib['score'])
    return df


def plans_summary(df):
    """
    :param df: dataframe, containing multiple agents' plans
    :return: dataframe, containing multiple agents' plans with more information
    """
    # calculate travel time
    df.loc[:, 'trav_time_min'] = df.trav_time.apply(lambda x: pd.Timedelta("0 days " + x))
    df.loc[:, 'trav_time_min'] = df.loc[:, 'trav_time_min'].apply(lambda x: x.seconds / 60)
    df.loc[:, 'speed'] = df.loc[:, 'distance'] / (df.loc[:, 'trav_time_min'] / 60)  # km/h

    # act_end - dep_time + trav_time = act_time for 1:
    df.loc[:, 'act_end_t'] = df.act_end.apply(
        lambda x: pd.Timedelta("0 days " + x) if x.split(':')[0] != '24' else pd.Timedelta(
            "1 days " + ':'.join(['00'] + x.split(':')[1:])))
    df.loc[:, 'dep_time_t'] = df.dep_time.apply(
        lambda x: pd.Timedelta("0 days " + x) if x.split(':')[0] != '24' else pd.Timedelta(
            "1 days " + ':'.join(['00'] + x.split(':')[1:])))
    df.loc[:, 'trav_time_t'] = df.trav_time.apply(lambda x: pd.Timedelta("0 days " + x))
    df.loc[:, 'act_time'] = df.apply(
        lambda row: (row['act_end_t'].seconds - row['dep_time_t'].seconds - row['trav_time_t'].seconds) / 60 if row[
                                                                                                                    'act_id'] != 0 else
        row['act_end_t'].seconds / 60, axis=1)
    # df.loc[:, 'act_time'] = df.loc[:, 'act_time'].apply(lambda x: x if x <= 1440 else x - 1440)
    df.drop(columns=['act_end_t', 'dep_time_t', 'trav_time_t'], inplace=True)

    # Convert act_end into the input format
    df.loc[:, 'act_end'] = df.act_end.apply(
        lambda x: pd.Timedelta("0 days " + x) if x.split(':')[0] != '24' else pd.Timedelta(
            "1 days " + ':'.join(['00'] + x.split(':')[1:])))
    df.loc[:, 'act_end'] = df.act_end.apply(lambda x: x.seconds / 3600)

    # Convert act_end into the input format
    df.loc[:, 'dep_time'] = df.dep_time.apply(
        lambda x: pd.Timedelta("0 days " + x) if x.split(':')[0] != '24' else pd.Timedelta(
            "1 days " + ':'.join(['00'] + x.split(':')[1:])))
    df.loc[:, 'dep_time'] = df.dep_time.apply(lambda x: x.seconds / 3600)  # hour
    df.loc[:, 'src'] = 'output'
    return df


def trips2stats(df_trips):
    """
    :param df_trips: dataframe, containing agents' activity plans
    :return: dataframe, describing agents' activity plans with indicators e.g., distance, utility score.
    """
    # Plan sequence
    def plan2stats(data):
        acts = data['act_purpose'].values
        acts_length = len(acts)
        modes = data['mode'].values[1:]
        seq = []
        for i in range(acts_length):
            seq.append(acts[i])
            if i != acts_length - 1:
                seq.append(modes[i])
        return pd.Series({'plan_seq': '-'.join(seq), 'travel_time': np.nansum(data['trav_time_min']),
                          'activity_time': np.nansum(data['act_time']),
                          'distance': np.nansum(data['distance']), 'speed': np.mean(data['speed']),
                          'score': data['score'].values[0]})

    df_stats = df_trips.groupby(['PId', 'src']).apply(plan2stats).reset_index()
    df_stats.loc[:, 'PId'] = df_stats.loc[:, 'PId'].astype(str)
    df_stats.sort_values(by=['PId', 'src'], inplace=True)
    return df_stats


def shp2poly(filename=None, targetfile=None):
    """
    Convert a given shapefile to a .poly file for the use by osmosis.
    :param filename: str, file path of the shape file
    :param targetfile: str, target .poly file
    """
    gdf_city_boundary = gpd.GeoDataFrame.from_file(filename).to_crs(4326)
    g = [i for i in gdf_city_boundary.geometry]
    all_coords = mapping(g[0])["coordinates"][0]
    F = open(targetfile, "w")
    F.write("polygon\n")
    F.write("1\n")
    for point in all_coords:
        F.write("\t" + str(point[0]) + "\t" + str(point[1]) + "\n")
    F.write("END\n")
    F.write("END\n")
    F.close()


def load_eff_lookup_table(lookup_table=None):
    """
    Create a look-up table of BEV energy efficiency given speed and slope.
    :param lookup_table: str, file name of the look-up table
    :return: a interpolated look-up table
    """
    filepath = f'dbs/ev/IDEAS_EnergyConsumptionModels/CSV/{lookup_table}'
    ev_discharge = pd.read_csv(filepath, index_col=0).unstack().reset_index()
    ev_discharge.columns = ['speed (m/s)', 'slope (%)', 'efficiency (kWh/km)']
    sp = scipy.interpolate.LinearNDInterpolator(ev_discharge.loc[:, ['speed (m/s)', 'slope (%)']].values,
                                                ev_discharge.loc[:, ['efficiency (kWh/km)']].values)
    return sp


def load_charging_table(battery_size=None):
    """
    Create a look-up table dictionary for charging time.
    :param battery_size: int, battery size
    :return: dict, a lookup table of power, soc, charging time
    """
    filepath = f'dbs/ev/charging_dynamics.csv'
    ev_charge = pd.read_csv(filepath)
    ev_charge = ev_charge.loc[ev_charge['battery_size'] == battery_size, :]
    sp_dict = dict()
    for power, data in ev_charge.groupby('power'):
        sp_dict[power] = dict()
        sp_dict[power]['time'] = scipy.interpolate.interp1d(data.loc[:, ['time']].values.reshape(-1),
                                        data.loc[:, ['soc_end']].values.reshape(-1))
        sp_dict[power]['soc'] = scipy.interpolate.interp1d(data.loc[:, ['soc_end']].values.reshape(-1),
                                        data.loc[:, ['time']].values.reshape(-1))
    return sp_dict


def matsim_events2database(scenario=None, test=False):
    """
    Write MATSim output events to the database.
    :param scenario: str, scenario name
    :param test: boolean, whether to do a test
    """
    user = keys_manager['database']['user']
    password = keys_manager['database']['password']
    port = keys_manager['database']['port']
    db_name = keys_manager['database']['name']
    engine = sqlalchemy.create_engine(f'postgresql://{user}:{password}@localhost:{port}/{db_name}')
    file_path2output = ROOT_dir + f'/dbs/{scenario}/output/'
    selected_types = ['actend', 'actstart', 'vehicle enters traffic',
                      'left link', 'vehicle leaves traffic']
    selected_vars = ['person', 'vehicle', 'time', 'type', 'link']
    events = matsim.event_reader(file_path2output + 'output_events.xml.gz', types=','.join(selected_types))
    events_list = []
    count = 0
    count2write = 0
    for event in tqdm(events, desc='Streaming events'):
        to_delete = set(event.keys()).difference(selected_vars)
        for d in to_delete:
            del event[d]
        events_list.append(event)
        count += 1
        count2write += 1
        if count2write > 2000000:
            df2write = pd.DataFrame.from_records(events_list)
            df2write.loc[:, 'person'] = df2write.loc[:, 'person'].fillna(df2write.loc[:, 'vehicle'])
            df2write.to_sql(scenario, engine, index=False, if_exists='append', method='multi', chunksize=10000)
            count2write = 0
            del df2write
            events_list = []
        if test & (count > 2000000):
            break
    if count2write > 0:
        df2write = pd.DataFrame.from_records(events_list)
        df2write.loc[:, 'person'] = df2write.loc[:, 'person'].fillna(df2write.loc[:, 'vehicle'])
        df2write.to_sql(scenario, engine, index=False, if_exists='append', method='multi', chunksize=10000)


def eventsdb2batches(scenario=None, batch_num=20):
    """
    Divide the database-stored events into batches for the BEV simulation.
    :param scenario: str, scenario name
    :param batch_num: int, number of batches
    """
    user = keys_manager['database']['user']
    password = keys_manager['database']['password']
    port = keys_manager['database']['port']
    db_name = keys_manager['database']['name']
    engine = sqlalchemy.create_engine(f'postgresql://{user}:{password}@localhost:{port}/{db_name}')
    # Connect to PostgreSQL server
    dbConnection = engine.connect()
    # Read data from PostgreSQL database table and load into a DataFrame instance
    df_event = pd.read_sql(f'''select * from {scenario} order by person, time, type DESC;''', dbConnection)

    print('Creating batches of events...')
    agents_car = df_event.loc[:, 'person'].unique()
    num_agents = len(agents_car)
    batch_num = batch_num
    batch_size = num_agents // batch_num
    if num_agents % batch_num == 0:
        batch_seq = list(range(0, batch_num)) * batch_size
    else:
        batch_seq = list(range(0, batch_num)) * batch_size + list(range(0, num_agents % batch_num))
    random.Random(4).shuffle(batch_seq)
    agents_car_batch_dict = {person: bt for person, bt in zip(agents_car, batch_seq)}
    df_event.loc[:, 'batch'] = df_event.loc[:, 'person'].map(agents_car_batch_dict)
    print(f'Number of car agents: {num_agents}.')

    batch_id = 0
    for _, data in tqdm(df_event.groupby('batch'), desc='Saving batches'):
        data.to_csv(ROOT_dir + f'/dbs/output_summary/{scenario}/ev_sim/events_batch{batch_id}.csv.gz',
                    index=False, compression="gzip")
        batch_id += 1
