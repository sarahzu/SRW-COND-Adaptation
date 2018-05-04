import gradient_ascent
import derivatives_of_the_random_walk
import numpy as np
from GraphFactory import GraphFactory
import operator
import networkx as nx

# Lets put it all together

# load the pruned graph containing only labeled data without isolated notes from
# complete_graph.graphml
labeled_graph = GraphFactory('labeled_graph_trump')

csv_path = './data_2/clean_data/election_debate_leaning_2016.csv'
graph_path = './data_2/clean_data/election_debate_leaning_2016.graphml'
ext_graph_path = './data_2/clean_data/extended_graph.graphml'

labeled_graph.create_graphs_from_dataframe('QT_REP', csv_path)
labeled_graph.prune_graph(1, 2, 2, True)
labeled_graph.print_info()
original_graph, extended_graph = labeled_graph.extend_labeled_graph()
original_nodes = np.asarray(list(original_graph.nodes()))
extended_nodes = np.asarray(list(extended_graph.nodes()))

# generating dict in the form of {node_1:[neigbors],node_2:[neighbors]}
# print('Calculate Neighbors...\n')
# neighbor_dict = {}
# for node in original_graph.nodes():
#     neighbor_dict[int(node)] = list(original_graph.neighbors(node))
#
# print('Calculate Feature Vector...\n')
# feature_vector = labeled_graph.create_feature_vector(extended_graph)

# initial_omega = np.array([1.0, 1.0, 1.0])
# print('\n\n')
# print('Beginning SRW-COND-Algorithm\n')
# V = original_nodes
# V_L_ext = extended_nodes
# v = V[10]
# u = V[4]
# Xe = feature_vector
# algo2_object = gradient_ascent.Algorithm2(1.0, u, v, 4, 10, initial_omega, neighbor_dict, Xe, V_L_ext, V)
# omega = algo2_object.gradient_ascent()
# algo3_object = derivatives_of_the_random_walk.Algorithm3(omega, Xe,
#                                                          neighbor_dict)
# Q = algo2_object.generate_full_transition_probability_matrix_Q(V_L_ext, omega)
# print("Q: \n", Q)
# pT, d_pT = algo3_object.derivatives_of_the_random_walk(V_L_ext, Q)
# print("Page Rank: \n", pT, "\nDerivative Page Rank: \n", d_pT, "\nOmega: \n", omega)

pT = nx.pagerank(extended_graph, weight='weight')

node_pt = dict(zip(extended_nodes,pT))
sorted_node_pt = sorted(node_pt.items(), key=operator.itemgetter(1), reverse=True)
infered_nodes = [k for (k,v) in sorted_node_pt][0:100]

for node in extended_graph.nodes():
    if node in infered_nodes:
        extended_graph.node[node]['leaning'] = 'NewR'

nx.write_graphml(extended_graph,'./data_2/clean_data/result_pT.graphml')