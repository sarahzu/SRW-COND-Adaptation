import gradient_ascent
import derivatives_of_the_random_walk
import numpy as np
from GraphFactory import GraphFactory
import operator
import networkx as nx

"""
SRW-COND algorithm
"""

# Creating Graph
labeled_graph = GraphFactory('labeled_graph_trump')

# Specifying paths to datasets and graph-files
csv_path = './data_2/clean_data/election_debate_leaning_2016.csv'
graph_path = './data_2/clean_data/election_debate_leaning_2016.graphml'
ext_graph_path = './data_2/clean_data/extended_graph.graphml'

# Create QT and REPLY - Graph
labeled_graph.create_graphs_from_dataframe('QT_REP', csv_path)

# Prune Graph and print information before and after pruning
labeled_graph.print_info()
labeled_graph.prune_graph(1, 2, 2, True)
labeled_graph.print_info()

# Extend original graph
original_graph, extended_graph = labeled_graph.extend_labeled_graph()

# Calculate neighbors for each node
neighbor_dict = {}
for node in original_graph.nodes():
    neighbor_dict[int(node)] = list(original_graph.neighbors(node))

# Calculate the feature vector for each edge
Xe = labeled_graph.create_feature_vector(extended_graph)

# Define the initial Omega
initial_omega = np.array([1.0, 1.0, 1.0])

# Convert graphs to numpy nodesets
V = np.asarray(list(original_graph.nodes()))
V_L_ext = np.asarray(list(extended_graph.nodes()))

# Calculate optimal omega
algo2_object = gradient_ascent.Algorithm2(1.0, V[10], V[4], 4, 10, initial_omega, neighbor_dict, Xe, V_L_ext, V)
omega = algo2_object.gradient_ascent()

# Calculate Transition Matrix
Q = algo2_object.generate_full_transition_probability_matrix_Q(V_L_ext, omega)

# Calculate Page Rank and Derivate Page Rank
algo3_object = derivatives_of_the_random_walk.Algorithm3(omega, Xe,neighbor_dict)
pT, d_pT = algo3_object.derivatives_of_the_random_walk(V_L_ext, Q)

# Find top n (n=500) nodes with highest page rank
node_pt = dict(zip(V_L_ext,pT))
sorted_node_pt = sorted(node_pt.items(), key=operator.itemgetter(1), reverse=True)
infered_nodes = [k for (k,v) in sorted_node_pt][0:500]

# Infer attribute
for node in extended_graph.nodes():
    if node in infered_nodes:
        extended_graph.node[node]['leaning'] = 'NewR'

# Save resulting graph to file
nx.write_graphml(extended_graph,'./data_2/clean_data/result_pT.graphml')