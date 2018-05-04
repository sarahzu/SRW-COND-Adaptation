"""
A Factory to handle and cluster all networkx-based graph activities. Contains methods for creating, pruning, 
plotting and information-gathering.
"""
import itertools
import operator
from multiprocessing import Pool
import community
import numpy as np
import ast
import networkx as nx
import pandas as pd


class GraphFactory:
    """
    GraphFactory-Class contains all relevant class-variables and methods. Represents a nx.Graph()-object
    """
    def __init__(self, name_of_graph):
        """
        Initializes graph and gives name to graph
        
        :param str name_of_graph: Name of graph needs to be specified here
        
        """
        print('Start Creating Graph "' + name_of_graph + '"')
        self.graph = nx.Graph()
        self.name_of_graph = name_of_graph
        self.pruning_information = self._PruningInfo()
        self.community_partitions = dict()
        self.dataframe = None
        self.subgraphs = []

    def add_node(self, node, label, leaning):
        """
        Method to add node and its label to the graph-object. Handles increasing of 'weight' attribute of each node
        automatically each time a node gets added multiple times
        
        :param int node: Node in the form of a twitter user-id or a hashtag
        
        :param str label: Label of node, either screen_name of a user or the name of the hashtag
        
        """
        node = int(node)
        label = str(label)
        if not self.graph.has_node(node):
            self.graph.add_node(node, weight = 1, label = label, leaning = leaning)
        else:
            self.graph.node[node]['weight'] += 1
            self.graph.node[node]['label'] = label
            self.graph.node[node]['leaning'] = leaning


    def add_edge(self, node_1, label_1, leaning_1 ,node_2, label_2, leaning_2):
        """
        Method to add edge to the graph-object. Handles increasing of 'weight' attribute of each edge and the related 
        nodes automatically each time an edge gets added multiple times.
        
        :param int node_1: First node or edge start node
        
        :param str label_1: Label of first node
        
        :param int node_2: Second node or edge end node
        
        :param str label_2: Label of second node
        
        """
        node_1 = int(node_1)
        node_2 = int(node_2)
        label_1 = str(label_1)
        label_2 = str(label_2)

        if not self.graph.has_edge(node_1, node_2):
            if not self.graph.has_node(node_1):
                self.graph.add_node(node_1, weight = 1, label = label_1, leaning = leaning_1)
            if not self.graph.has_node(node_2):
                self.graph.add_node(node_2, weight = 1, label = label_2, leaning = leaning_2)
            self.graph.add_edge(node_1, node_2, weight =  1)
        else:
            self.graph[node_1][node_2]['weight'] += 1

    def write_graphml(self, output_path):
        """
        Writes the graph-object to file in the form of a Graphml-File
        
        :param str output_path: Path to destination of Graphml-File
        
        """
        print("Writing Graph to " + str(output_path))
        nx.write_graphml(self.graph, output_path)

    def load_graph(self, input_path):
        """
        Loads a graph-object from a Graphml-File
        
        :param str input_path: path to Graphml-File
        
        """
        print("Loading Graph from " + str(input_path))
        with open(input_path) as inp:
            self.graph = nx.read_graphml(inp)

    def print_info(self):
        """
        Prints a short overview of the graph including the amount of nodes, edges, the average degree and the average
        weight
        """
        list_of_degrees = [val for (node, val) in self.graph.degree()]
        print('--------------------------------------')
        print('Number of nodes: ' + str(self.graph.number_of_nodes()))
        print('Number of edges: ' + str(self.graph.number_of_edges()))
        print('Average degree: ' + str(
            round(float(sum(list_of_degrees)) / float(self.graph.number_of_nodes()), 2)))

        avg_sum = 0
        for _, data in self.graph.nodes(data=True):
            avg_sum += data['weight']

        print('Average weight: ' + str(round(float(avg_sum) / float(self.graph.number_of_nodes()), 2)))
        print('--------------------------------------')

    def get_threshold(self, total_percentage_of_nodes_remaining, blur, name_of_measure):
        """
        Calculates the threshold for a given name_of_measure such that, when specified measure is pruned according to
        that threshold, the percentage of nodes remaining in the graph is equal to the parameter total_percentage_of_nodes_remaining.
        Calculated threshold is only right, when pruning option remove_isolated is applied separately.

        :param int total_percentage_of_nodes_remaining: Percentage of nodes to keep in the graph after pruning

        :param int blur: How many percentage-points can the threshold be wrong

        :param str name_of_measure: On what node measure should the algorithm be applied. So far only 'weight_thresh' or 'degree_thresh' are possible

        :return int:
        """
        total_node_count = self.graph.number_of_nodes()
        if total_node_count == 0:
            return 0
        if name_of_measure == 'weight_thresh':
            temp_threshold = 2
            continue_eval = True
            nodes_to_remove = []

            error_count_minus = 0
            error_count_plus = 0
            while continue_eval:
                for node, data in self.graph.nodes(data=True):
                    if data['weight'] < temp_threshold:
                        nodes_to_remove.append(node)
                temp_remaining = 100 - len(nodes_to_remove) / total_node_count * 100
                if total_percentage_of_nodes_remaining + blur > temp_remaining > total_percentage_of_nodes_remaining - blur:
                    continue_eval = False
                elif total_percentage_of_nodes_remaining - blur < temp_remaining:
                    temp_threshold = temp_threshold + 1
                    error_count_plus += 1
                elif temp_remaining < total_percentage_of_nodes_remaining - blur:
                    temp_threshold = temp_threshold - 1
                    error_count_minus += 1
                if error_count_plus > 2 and error_count_minus > 2:
                    error_count_plus = 0
                    error_count_minus = 0
                    continue_eval = False
                nodes_to_remove = []
            return temp_threshold
        if name_of_measure == 'degree_thresh':
            temp_threshold = 2
            continue_eval = True
            nodes_to_remove = []

            error_count_minus = 0
            error_count_plus = 0
            while continue_eval:
                for node, data in self.graph.nodes(data=True):
                    if 'Degree' not in data:
                        print('Graph has no attribute "degree" - please use "add_stats"-function')
                        exit(1)
                    if data['Degree'] < temp_threshold:
                        nodes_to_remove.append(node)
                temp_remaining = 100 - len(nodes_to_remove) / total_node_count * 100
                if total_percentage_of_nodes_remaining + blur > temp_remaining > total_percentage_of_nodes_remaining - blur:
                    continue_eval = False
                elif total_percentage_of_nodes_remaining - blur < temp_remaining:
                    temp_threshold = temp_threshold + 1
                    error_count_plus += 1
                elif temp_remaining < total_percentage_of_nodes_remaining - blur:
                    temp_threshold = temp_threshold - 1
                    error_count_minus += 1
                if error_count_plus > 2 and error_count_minus > 2:
                    error_count_plus = 0
                    error_count_minus = 0
                    continue_eval = False
                nodes_to_remove = []
            return temp_threshold

    def prune_graph(self, edge_weight, degree_thresh, weight_thresh, remove_isolates):
        """
        Prunes the graph-object on the specified measurements.

        :param int edge_weight: minimum weight an edge can have to remain in the graph

        :param int degree_thresh: minimum degree an node can have to remain in the graph

        :param int weight_thresh: minimum weight an node can have to remain in the graph

        :param bool remove_isolates: should isolated nodes be removed?

        """
        print("Pruning Graph")

        if edge_weight != 0:
            self.pruning_information.edge_weight = edge_weight
        if degree_thresh != 0:
            self.pruning_information.node_degree = degree_thresh
        if weight_thresh != 0:
            self.pruning_information.node_weight = weight_thresh
        if remove_isolates:
            self.pruning_information.isolated = remove_isolates

        edges_to_remove = []
        for edge in self.graph.edges(data=True):
            if edge[0] == edge[1]:
                edges_to_remove.append(edge)
            if edge[2]['weight'] < edge_weight:
                edges_to_remove.append(edge)

        print(".......... Removing self refering edges(" + str(len(edges_to_remove))+')')
        self.graph.remove_edges_from(edges_to_remove)

        nodes_to_remove = []
        for node, data in self.graph.nodes(data=True):
            if data['weight'] < weight_thresh:
                nodes_to_remove.append(node)
            elif data['Degree'] < degree_thresh:
                nodes_to_remove.append(node)

        print(".......... Removing nodes according to degree and weight(" + str(len(nodes_to_remove)) + ')')
        self.graph.remove_nodes_from(nodes_to_remove)

        if remove_isolates:
            isolated_nodes_to_remove = list(nx.isolates(self.graph))
            print(".......... Removing isolated nodes(" + str(len(isolated_nodes_to_remove)) + ')')
            self.graph.remove_nodes_from(isolated_nodes_to_remove)

        # components = sorted(nx.connected_component_subgraphs(self.graph),key=len,reverse=True)
        # self.graph = self.graph.subgraph(components[0:10],inplace=True)

        print("Pruning Done")

    class _PruningInfo():
        """
        Class to pack all the Information on the applied Pruning that has been done on current graph-object.
        """
        def __init__(self):
            """
            Sets all pruning-info to 0.
            """
            self.node_weight = 0
            self.node_degree = 0
            self.edge_weight = 0
            self.isolated = False

        def to_string(self):
            """
            Returns all the info contained in _PruningInfo in one string.
            """
            output = 'Minimum Edge Weight = ' + str(self.edge_weight) + '\n'
            output += 'Minimum Degree of Nodes = ' + str(self.node_degree) + '\n'
            output += 'Minimum Weight of Nodes = ' + str(self.node_weight) + '\n'
            output += 'Isolated Nodes were removed = ' + str(self.isolated)
            return output

    def add_stats(self, degree, closeness, betweenness):
        """
        Calculates measures and statistics such as degree, closeness and betweenness to the graph-object and to its
        nodes.

        :param bool degree: If set to true, the degree of each node is added as an attribute

        :param bool closeness: If set to true, the closeness of each node is added as an attribute

        :param bool betweenness: If set to true, the betweeness of each node is added as an attribute. ATTENTION: Takes a long time

        """
        print("Calculating Stats")
        if degree:
            print(".......... Degree")
            q={}
            for (k,v) in self.graph.degree():
                q[k]=v
            deg = list(self.graph.degree())
            nx.set_node_attributes(self.graph, q,'Degree')
        if closeness:
            print(".......... Closeness")
            close = nx.closeness_centrality(self.graph)
            nx.set_node_attributes(self.graph, 'Closeness_centrality', close)
        if betweenness:
            print(".......... Betweeness")
            betwe = calculate_betweenness(self.graph)
            nx.set_node_attributes(self.graph, 'Betweenness_centrality', betwe)
        print("Calculating Degree")

    def _get_communities(self):
        """
        Helper method to calculate communities with the help of the 'community'-package
        """
        self.community_partitions = community.best_partition(self.graph)


    def save_as_subgraph(self, subgraph, name):
        sub_nodes = list(subgraph.nodes())
        sub_dataframe = self.dataframe[self.dataframe['u_id'].isin(sub_nodes)]
        sub_dataframe = sub_dataframe.set_index('u_id',drop=False)
        sub_graph = GraphFactory(name)
        sub_graph.dataframe = sub_dataframe
        self.subgraphs.append(sub_graph)


    def create_graphs_from_dataframe(self, on_key, path_to_csv):
        """
        Method to quickly generate graphs on QT, RT or hashtags. If the method write_out_community_user_info should be
        applied, the graph needs to be generated with this method or else there will be no underlying dataframe
        to gather information from.


        :param str on_key: On which key should the graph be generated. Possible are retweeted('RT'), quoted('QT') or hashtags('hashtags')

        :param str path_to_csv: Path to Csv-File containing twitter-information

        """
        assert on_key == 'RT' or on_key == 'QT' or on_key == 'hashtag' or on_key == 'QT_REP', print('Graph not known')

        print('Loading file')
        dataframe = pd.read_csv(path_to_csv, low_memory=False)

        if on_key == 'QT_REP':
            print('Creating Graph using Quotes and Replies')
            self.dataframe = dataframe[(dataframe.QT == 1) | (dataframe.is_reply == 1)]
            self.dataframe = self.dataframe.set_index('u_id',drop=False)
            for row in self.dataframe.itertuples():
                leaning_2 = row.leaning
                if row.QT == 1:
                    temp = self.dataframe[self.dataframe['u_id'] == row.q_uid]
                    try:
                        leaning_1 = temp.at[int(row.q_uid), 'leaning']
                        if isinstance(leaning_1,np.ndarray):
                            leaning_1 = max(set(leaning_1.tolist()), key=leaning_1.tolist().count)
                    except:
                        leaning_1 = 'Unknown'
                    self.add_node(row.q_uid, row.q_screen_name, leaning_1)
                    self.add_edge(row.q_uid, row.q_screen_name, leaning_1, row.u_id, row.screen_name, leaning_2)
                else:
                    temp = self.dataframe[self.dataframe['u_id'] == row.in_reply_to_user_id_str]
                    try:
                        leaning_1 = temp.at[int(row.in_reply_to_user_id_str),'leaning']
                        if isinstance(leaning_1,np.ndarray):
                            leaning_1 = max(set(leaning_1.tolist()), key=leaning_1.tolist().count)
                    except:
                        leaning_1 = 'Unknown'
                    self.add_node(row.in_reply_to_user_id_str, row.in_reply_to_screen_name, leaning_1)
                    self.add_edge(row.in_reply_to_user_id_str, row.in_reply_to_screen_name, leaning_1,row.u_id, row.screen_name,leaning_2)
            self.add_stats(degree=True, betweenness=False, closeness=False)
            self.print_info()


    def create_labeled_subgraph(self):
        """
        Creates a subgraph containing only the nodes that have a value different from "Unknown" under leaning
        :return: SubGraph
        """
        labeled_nodes = []
        for (p, d) in self.graph.nodes(data=True):
            if (d['leaning'] == 'R') or (d['leaning'] == 'D'):
                labeled_nodes.append(p)
        return nx.subgraph(self.graph, labeled_nodes)

    def create_unlabeled_subgraph(self):
        """
        Creates a subgraph containing only the nodes that have a value equals to "Unknown" under leaning
        :return: SubGraph
        """
        unlabeled_nodes = []
        for (p, d) in self.graph.nodes(data=True):
            if (d['leaning'] == 'Unknown'):
                unlabeled_nodes.append(p)
        return nx.subgraph(self.graph, unlabeled_nodes)

    def create_feature_vector(self, graph = None):
        """
        Generates a vector that sepcifies for each edge:
            - common hashtags for both nodes
            - common usermentions for both nodes
            - the edge weight
        :param graph: If not specified self.graph is taken
        :return: Feature vector for every edge
        """
        if graph == None:
            used_graph = self.graph
        else:
            used_graph = graph
        feature_vector = {}
        for edge in used_graph.edges(data=True):
            node_1, node_2 = edge[0],edge[1]
            try:
                node_1_hashtags = self.dataframe.at[node_1,'hashtags']
                node_2_hashtags = self.dataframe.at[node_2,'hashtags']
                node_1_mentions = self.dataframe.at[node_1, 'usermentions']
                node_2_mentions = self.dataframe.at[node_2, 'usermentions']

                if isinstance(node_1_hashtags,np.ndarray):
                    hashtags_1 = node_1_hashtags.tolist()
                    hashtags_1 = [ast.literal_eval(i) for i in hashtags_1]
                    hashtags_1 = [item for sublist in hashtags_1 for item in sublist]
                else:
                    hashtags_1 = []

                if isinstance(node_2_hashtags,np.ndarray):
                    hashtags_2 = node_2_hashtags.tolist()
                    hashtags_2 = [ast.literal_eval(i) for i in hashtags_2]
                    hashtags_2 = [item for sublist in hashtags_2 for item in sublist]
                else:
                    hashtags_2 = []

                if isinstance(node_1_mentions,np.ndarray):
                    user_mentions_1 = node_1_mentions.tolist()
                    user_mentions_1 = [ast.literal_eval(i) for i in user_mentions_1]
                    user_mentions_1 = [item for sublist in user_mentions_1 for item in sublist]
                else:
                    user_mentions_1 = []

                if isinstance(node_2_mentions,np.ndarray):
                    user_mentions_2 = node_2_mentions.tolist()
                    user_mentions_2 = [ast.literal_eval(i) for i in user_mentions_2]
                    user_mentions_2 = [item for sublist in user_mentions_2 for item in sublist]
                else:
                    user_mentions_2 = []

                feature_1 = len(set(hashtags_1).intersection(set(hashtags_2)))
                feature_2 = len(set(user_mentions_1).intersection(set(user_mentions_2)))
                feature_vector[(edge[0],edge[1])] = [feature_1,feature_2,edge[2]['weight']]
            except:
                feature_vector[(edge[0], edge[1])] = [0,0,0]
        return feature_vector

    def extend_labeled_graph(self,number_of_comm = 10, number_of_top_comm = 5):
        """
        Extends self.graph through taking the top communities and adding unlabeled nodes to the original graph
        :param number_of_comm: Amount of communities to use
        :param number_of_top_comm: Amount of communities to use for extension
        :return: The extended labeled graph and the original seed node graph
        """
        la = community.best_partition(self.graph)
        nx.set_node_attributes(self.graph, la, 'community')
        nodes = self.graph.nodes(data=True)

        a = list(set(list(la.values())))
        temp = {}
        for comm in a:
            temp[comm] = [k for k, v in la.items() if v == comm]

        s = sorted(temp, key=lambda k: len(temp[k]), reverse=True)[:number_of_comm]
        comm_size = {}
        for key in s:
            if key in temp:
                comm_size[key] = temp[key]

        dict_leaning_amount = {}
        for comm, ids in comm_size.items():
            count_r = 0
            for node in ids:
                if self.graph.node[node]['leaning'] == 'R':
                    count_r += 1
            dict_leaning_amount[comm] = count_r
        sort_lean = sorted(dict_leaning_amount.items(), key=operator.itemgetter(1), reverse=True)
        top_x = [k for k, v in sort_lean][0:number_of_top_comm]

        extendible_nodes = []
        for comm in top_x:
            nodes = temp[comm]
            for node in nodes:
                if self.graph.node[node]['leaning'] == 'Unknown':
                    extendible_nodes.append(node)

        original_graph = self.create_labeled_subgraph()
        extendible_nodes.extend(list(self.create_labeled_subgraph().nodes()))
        extendible_node_Set = set(extendible_nodes)

        extended_graph = nx.subgraph(self.graph, list(extendible_node_Set))
        return original_graph,extended_graph


"""
The following methods __btwn_pool, __partitions and calculate_betweenness where taken from
https://blog.dominodatalab.com/social-network-analysis-with-networkx/ (21.4.17 - 13:24)

Author: Manojit Nandi, 14.7.15
Modified: Lenz Baumann 21.4.17
"""

def _btwn_pool(G_tuple):
    """
    Wrapper for the method nx.betweenness_centrality_source.
    :param G_tuple: nodes of graph G 
    :return dict:  nodes with their betweenness centrality
    """
    return nx.betweenness_centrality_source(*G_tuple)


def _partitions(nodes, n):
    """
    Splits the nodes up into n different partitions
    :param nodes: 
    :param int n: 
    :return tuple: the partitions
    """
    nodes_iter = iter(nodes)
    while True:
        partition = tuple(itertools.islice(nodes_iter, n))
        if not partition:
            return
        yield partition


def calculate_betweenness(graph):
    """
    Calculates the betweeness of all nodes of the graph in parallel.
    ATTENTION: May take a long time
    
    :param nx.Graph graph: 
    
    :return list of int: Returns the betweenness for each node
     
    """
    p = Pool(processes=None)
    part_generator = 4 * len(p._pool)
    node_partitions = list(_partitions(graph.nodes(), int(len(graph) / part_generator)))
    num_partitions = len(node_partitions)

    bet_map = p.map(_btwn_pool,
                    zip([graph] * num_partitions,
                        [True] * num_partitions,
                        [None] * num_partitions,
                        node_partitions))

    bt_c = bet_map[0]
    for bt in bet_map[1:]:
        for n in bt:
            bt_c[n] += bt[n]
    p.close()
    return bt_c


