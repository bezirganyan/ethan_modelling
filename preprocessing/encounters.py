import argparse
from datetime import date, timedelta
from os import path, listdir, makedirs

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Memory, memory


class EncounterProcessor:
    def __init__(self, encounters_dir: str,
                 district_graph_file: str = 'district_graph.gpickle',
                 data_output_dir: str = 'output',
                 start_day: int = None,
                 end_day: int = None,
                 verbose: bool = False,
                 mode: str = 'graph_learning'):
        """
        Parameters
        ----------
        encounters_dir : str
            path to the folders containing the encounters data (output of the simulation)
        district_graph_file : srt, default = district_graph.gpickle
            the name of the district graph file in the encounters folder
        data_output_dir: str, default = output
            Directory where preprocessing output must be saved
        start_day : int, optional
            the day from which the data needs to be loaded. If not provided the smallest day in directory will be taken
        end_day : int, optional
            the day until which the data needs to be loaded.  If not provided the biggest day in directory will be taken
        verbose : bool = False
        """
        self.mode = mode
        self.data_output_dir = data_output_dir
        self.end_day = end_day
        self.start_day = start_day
        self.district_graph_file = district_graph_file
        self.encounters_dir = encounters_dir
        self.unique_fac_types = None
        self.verbose = verbose
        self.mobility_graph = None
        self.data = None
        self.nodes = None

        self.modes_set = {'graph_learning', 'tabular_visit'}
        if self.mode not in self.modes_set:
            raise ValueError(f'mode must be in: {self.modes_set}')

    def prepare(self):
        if not path.exists(self.data_output_dir):
            makedirs(self.data_output_dir)
        if self.verbose:
            print(f'Mode chosen: {self.mode}')
        if self.mode == 'graph_learning':
            self.prepare_data_for_graph_learning()
        elif self.mode == 'tabular_visit':
            self.prepare_data_for_tabular_visit_learning()
        else:
            raise ValueError(f'mode must be in: {self.modes_set}')

    def prepare_data_for_graph_learning(self):
        self.data = self.read_day_partitioned_data(self.encounters_dir, 'encounters')
        self.nodes = self.read_day_partitioned_data(self.encounters_dir, 'nodes')
        self.clean_encounters_data([*range(14), 15])
        self.add_inverse_data()
        self.add_district_infection_numbers()
        self.compute_mobility_graph('mobility_graph.gpickle')
        self.create_mobility_dataset()

    def prepare_data_for_tabular_visit_learning(self):
        self.data = self.read_day_partitioned_data(self.encounters_dir, 'encounters')
        self.nodes = self.read_day_partitioned_data(self.encounters_dir, 'nodes')
        self.clean_encounters_data([*range(14), 15])
        self.add_crowdedness()
        self.add_inverse_data()
        self.add_district_infection_numbers()
        self.add_node_status()
        self.add_total_cases_in_past_day()
        self.data = self.data[self.data['status'] == 0]
        agg_list = ('duration', 'lead_to_infection', 'status_node2', 'crowdedness_fac', 'intensity', 'dist_inf')
        group_by_list = ('node_id1', 'gen_fac', 'dm', 'facility_id', 'day', 'starting_time', 'cases')
        self.data = self.data.groupby([*group_by_list])[[*agg_list]].agg(
            duration=pd.NamedAgg(column='duration', aggfunc='sum'),
            inf=pd.NamedAgg(column='lead_to_infection', aggfunc='max'),
            avg_intensity=pd.NamedAgg(column='intensity', aggfunc=np.mean),
            num_inf_in_fac=pd.NamedAgg(column='status_node2', aggfunc='sum'),
            crowdedness_fac=pd.NamedAgg(column='crowdedness_fac', aggfunc=np.mean),
            dist_inf=pd.NamedAgg(column='dist_inf_x', aggfunc=np.mean))
        self.data.to_csv(path.join(self.data_output_dir, 'visits.csv'))

    def read_day_partitioned_data(self, data_dir: str, prefix: str) -> pd.DataFrame:
        """
        Read data from a directory, where data is partitioned based on days. The formats of the files
        in the directory has to be: `[prefix][day].csv`, where prefix is an argument provided
        Parameters
        ----------
        data_dir : str
            path to the directory where the data is stored
        prefix : str
            the prefix of each file. The file format after appending the day will be `[prefix][day].csv`, where
            `[prefix]` is the prefix provided, and `[day]` is the simulation day in range [`start_day`, `end_day`)

        Returns
        -------
        pandas.DataFrame
            returns loaded and concatenated data in a single pandas dataframe
        """
        if self.verbose:
            print(f'- Reading {prefix} data . . .')

        if not self.start_day or not self.end_day:
            files = listdir(data_dir)
            files = [f for f in files if f.startswith(prefix) and f.endswith('.csv')]
            try:
                files = [int(f[len(prefix):-len('.csv')]) for f in files]
            except ValueError:
                raise ValueError('Files are in wrong format or the prefix is incorrect. Please refer to documentation')
        self.start_day = self.start_day if self.start_day else min(files)
        self.end_day = self.end_day if self.end_day else max(files)

        data = pd.read_csv(path.join(data_dir, f'{prefix}0.csv'))
        data['dm'] = 0
        for i in range(self.start_day, self.end_day):
            new_data = pd.read_csv(path.join(data_dir, f'{prefix}{i}.csv'))
            new_data['dm'] = i
            data = pd.concat([data, new_data], ignore_index=True)
        data.columns = [col.strip() for col in data.columns]
        return data

    def clean_encounters_data(self, int_columns: list) -> None:
        """
        Basic cleaning of the encounters data
        Parameters
        ----------
        int_columns : list
            list of column indexes to be converted to int
        Returns
        -------
        None
        """
        if self.verbose:
            print(f'- Cleaning data . . .')

        for c in int_columns:
            self.data.iloc[:, c] = self.data.iloc[:, c].astype(int)

        fac_dict = {'00': 'home', '01': 'edu', '02': 'work', '10': 'gym', '11': 'shopping', '12': 'restaurant',
                    '13': 'cinema', '20': 'bus', '21': 'car', '22': 'train', '30': 'etc'}

        self.data['gen_fac'] = self.data['generator'].astype(int).astype(str) + self.data['facility_type'].astype(
            int).astype(str)
        self.data['gen_fac'] = self.data['gen_fac'].apply(lambda x: fac_dict[x] if x[0] != '3' else 'etc')
        self.unique_fac_types = self.data['gen_fac'].unique()
        self.data['fac_id'] = self.data['gen_fac'] + self.data['facility_id'].astype(str)

    def add_crowdedness(self) -> None:
        """
        Add facility crowdedness information to data
        Returns
        -------
        None
        """
        if self.verbose:
            print(f'- Adding crowdedness . . .')

        rev_dict = {d: i for i, d in enumerate(self.data['fac_id'].unique())}
        crowdedness = np.zeros((self.data['fac_id'].nunique(), 24, 7))
        for i, fac in tqdm(enumerate(self.data['fac_id'].unique()), total=len(self.data['fac_id'].unique())):
            cr_fac = self.data[(self.data['fac_id'] == fac)]
            for hour in range(48):
                crd = cr_fac[(cr_fac['starting_time'] <= hour) & (
                        cr_fac['starting_time'] + (cr_fac['duration'] / 60) * 2 >= hour)].groupby(['dm', 'day'])[
                    'node_id1'].count().reset_index().groupby('day')['node_id1'].mean()
                crowdedness[i, hour // 2, crd.index.astype(int).values] += crd.values

        res = []
        for i in tqdm(range(0, self.data.shape[0], 100)):
            res.extend(list(self.data.iloc[i:i + 100].apply(lambda x: crowdedness[
                rev_dict[x['gen_fac'] + str(x['facility_id'])], int(x['starting_time'] // 2), int(x['day'])],
                                                            axis=1).values))
        res = 1
        self.data['crowdedness_fac'] = res

    def add_inverse_data(self) -> None:
        if self.verbose:
            print(f'- Adding inverse data . . .')

        """
        Inverse the data based on node_id, and add inverse linkes. i.e if there exists encounter between nodes 0, 1,
        add same encounter as 1, 0 as well
        Returns
        -------
        None
        """
        data_c = self.data.copy()
        data_c['node_id1_c'] = data_c['node_id2']
        data_c['node_id2'] = data_c['node_id1']
        data_c['node_id1'] = data_c['node_id1_c']
        data_c = data_c.drop(['node_id1_c'], axis=1)
        self.data = pd.concat([self.data, data_c], axis=0)

    def add_node_status(self) -> None:
        """
        Add columns which indicate the status of each node.
        Returns
        -------
        None
        """
        if self.verbose:
            print(f'- Adding node status . . .')

        nodes_past_inf = self.nodes[['id', 'status', 'dm']].copy()
        nodes_past_inf.loc[:, 'dm'] = nodes_past_inf.loc[:, 'dm'] + 1
        self.data = self.data.merge(nodes_past_inf, 'left', left_on=['node_id2', 'dm'], right_on=['id', 'dm'])
        self.data.loc[:, 'status'] = self.data.loc[:, 'status'].fillna(0).astype(int)
        self.data = self.data.rename({'status': 'status_node2'}, axis=1)
        self.data = self.data.merge(nodes_past_inf, 'left', left_on=['node_id1', 'dm'], right_on=['id', 'dm'])
        self.data.loc[:, 'status'] = self.data['status'].fillna(0).astype(int)
        self.data = self.data.drop(['id_x', 'id_y'], axis=1)

    def add_total_cases_in_past_day(self):
        if self.verbose:
            print(f'- Adding total cases in prev day . . .')

        cases = self.nodes[self.nodes['status'] == 1].groupby('dm')['status'].sum()
        cases = list(cases.values)
        cases.insert(0, cases[0])
        cases = cases[:-1]
        res = []
        for i in tqdm(range(0, self.data.shape[0], 100)):
            res.extend(list(self.data.iloc[i:i + 100].apply(lambda x: cases[x['dm']], axis=1)))
        res = 1
        self.data['cases'] = res

    def add_district_infection_numbers(self):
        if self.verbose:
            print(f'- Adding district infection numbers . . .')

        district_inf = self.data.groupby(['dm', 'district'])['lead_to_infection'].sum().reset_index()
        district_inf.columns = ['dm', 'district', 'dist_inf']
        district_inf_t = district_inf.copy()
        district_inf.loc[:, 'dm'] = district_inf['dm'] + 1
        self.data = self.data.merge(district_inf, 'left', ['dm', 'district'])
        self.data = self.data.merge(district_inf_t, 'left', ['dm', 'district'])

    def compute_mobility_graph(self, mobility_graph_file: str = None) -> None:
        if self.verbose:
            print(f'- Computing the  mobility graph . . .')

        '''
        Compute agent mobility information for the district graph
        Parameters
        ----------
        mobility_graph_file : str, optional
            file where mobility graph needs to be saved. If not provided graph will not be saved.
        Returns
        -------
        None
        '''
        self.mobility_graph = nx.read_gpickle(path.join(self.encounters_dir, self.district_graph_file))
        for u in tqdm(self.data['node_id1'].unique()):
            ds = self.data[self.data['node_id1'] == u]
            for d in range(self.data.dm.max() + 1):
                links = ds[ds['dm'] == d].sort_values(by='starting_time')['district'].values
                for a, b in zip(links[:-1], links[1:]):
                    if not self.mobility_graph.has_edge(a, b):
                        self.mobility_graph.add_edge(a, b)
                    if d not in self.mobility_graph.edges[(a, b)]:
                        self.mobility_graph.edges[(a, b)][d] = 0
                    self.mobility_graph.edges[(a, b)][d] += 1
        if mobility_graph_file:
            nx.write_gpickle(self.mobility_graph, path.join(self.data_output_dir, mobility_graph_file))
        if self.verbose:
            print('[+] Mobility graph computed!')

    def create_mobility_dataset(self, startdate: date = date(2020, 1, 1)):
        """
        Create Pandas dataset compatible with MPNN_LSTM model from mobility graph
        Parameters
        ----------
        startdate : datetime.date, default = 2020, 1, 1

        Returns
        -------
        None
        """
        if self.verbose:
            print(f'- Creating mobility dataset . . .')

        if self.mobility_graph is None:
            raise ValueError('Mobility graph cannot be None, run compute_mobility_graph function to generate the graph')

        graphs_path = path.join(self.data_output_dir, 'graphs')
        if not path.exists(graphs_path):
            makedirs(graphs_path)
        labels = self.data.groupby(['district', 'dm'])['dist_inf_y'].mean().reset_index()
        labels = labels.pivot(index='district', columns='dm', values='dist_inf_y').reset_index()
        labels.columns = ['name'] + [(startdate + timedelta(d)).strftime("%Y-%m-%d") for d in
                                     range(self.start_day, self.end_day)]

        labels.to_csv(path.join(self.data_output_dir, 'simulation_labels.csv'), index=False)
        for d in range(self.start_day, self.end_day):
            weights = []
            w = nx.get_edge_attributes(self.mobility_graph, d)
            for e in w:
                weights.append((e[0], e[1], w[e]))
            df = pd.DataFrame(weights)
            df.to_csv(path.join(graphs_path, f'SI_{startdate.strftime("%Y-%m-%d")}.csv'), header=False, index=False)
            startdate += timedelta(1)


if __name__ == '__main__':
    memory = Memory("cachedir")
    parser = argparse.ArgumentParser()
    parser.add_argument('--encounters_dir', type=str, required=True,
                        help='Path to the folders containing the encounters data (output of the simulation)')
    parser.add_argument('--district_graph_file', type=str, default='district_graph.gpickle',
                        help='The name of the district graph file in the encounters folder')
    parser.add_argument('--data_output_dir', type=str, default='output',
                        help='Directory where preprocessing output must be saved')
    parser.add_argument('--start_day', type=str, default=None,
                        help='The day from which the data needs to be loaded. If not provided the smallest day in directory will be taken')
    parser.add_argument('--end_day', type=str, default=None,
                        help='The day until which the data needs to be loaded. If not provided the biggest day in directory will be taken')
    parser.add_argument('--verbose', dest='verbose', action='store_true',
                        help='Verbosity')
    parser.add_argument('--mode', type=str, default='graph_learning',
                        help='The day until which the data needs to be loaded.  If not provided the biggest day in directory will be taken')
    args = parser.parse_args()

    ep = EncounterProcessor(**vars(args))
    ep.prepare()
