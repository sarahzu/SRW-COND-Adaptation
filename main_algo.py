import networkx as nx
import pandas as pd
import gradient_ascent
import derivatives_of_the_random_walk
import numpy as np
from GraphFactory import GraphFactory

# Lets put it all together

# load the pruned graph containing only labeled data without isolated notes from
# complete_graph.graphml
labeled_graph = GraphFactory('labeled_graph_trump')

csv_path = './data_2/clean_data/election_debate_leaning_2016.csv'
graph_path = './data_2/clean_data/election_debate_leaning_2016.graphml'
ext_graph_path = './data_2/clean_data/extended_graph.graphml'


labeled_graph.create_graphs_from_dataframe('QT_REP',csv_path)
labeled_graph.prune_graph(1,2,2,True)
labeled_graph.print_info()
labeled_graph.extend_labeled_graph()
original_graph = labeled_graph.subgraphs[0].nodes()
extended_graph = labeled_graph.subgraphs[1].nodes()


# generating dict in the form of {node_1:[neigbors],node_2:[neighbors]}
neighbor_dict = {}
for node in original_graph.nodes():
    neighbor_dict[int(node)] = list(original_graph.neighbors(node))

feature_vector = labeled_graph.create_feature_vector()

original_nodes = list(original_graph.nodes())
extended_nodes = list(extended_graph.nodes())


def generate_page_rank_score_plus_derivative_and_optimal_omega():
    # initial_omega = np.empty([1])
    initial_omega = np.array([1,1,1])
    print("initial omega:", initial_omega)
    V = original_nodes
    V_L_ext = extended_nodes
    v = V[10]
    u = V[4]
    Xe = feature_vector

    algo2_object = gradient_ascent.Algorithm2(1.0, u, v, 4, 10, initial_omega,
                                              neighbor_dict, Xe, V_L_ext, V)
    # Q = algo2_object.generate_full_transition_probability_matrix_Q(V, initial_omega)
    # print(Q)
    omega = algo2_object.gradient_ascent()
    algo3_object = derivatives_of_the_random_walk.Algorithm3(omega, Xe,
                                                              neighbor_dict)
    Q = algo2_object.generate_full_transition_probability_matrix_Q(V, omega)
    print("Q:", Q)
    pT, d_pT = algo3_object.derivatives_of_the_random_walk(V, Q)
    print("Page Rank:", pT, "\nDerivative Page Rank:", d_pT, "\nomega:", omega)

    print("v:", v)
    print("u:", u)
    print("V:", V)
    print("V L ext:", V_L_ext)
    print("Xe:", Xe)
    print("neighbors:", neighbor_dict)


if __name__ == '__main__':
    generate_page_rank_score_plus_derivative_and_optimal_omega()

# Voila
# 1. Neighbors of each node
# neighbor_dict
# 2. Feature Vector
# feature_vector
# 3. extended und originales set von Nodes
# original_nodes,extended_nodes
