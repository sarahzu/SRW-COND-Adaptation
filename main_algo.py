import networkx as nx
import pandas as pd
import gradient_ascent
import derivatives_of_the_random_walk
import numpy as np

# Lets put it all together

# load the pruned graph containing only labeled data without isolated notes from
# complete_graph.graphml
with open('./data/original_graph.graphml', 'r') as input_graph:
    original_graph = nx.read_graphml(input_graph)


# generating dict in the form of {node_1:[neigbors],node_2:[neighbors]}
neighbor_dict = {}
for node in original_graph.nodes():
    neighbor_dict[int(node)] = list(original_graph.neighbors(node))


# generate the feature vector for each edge as a dict in the form of
# {edge_1:[feature_vector],edge_2:[feature_vetor]}
with open('./data/follower_info_cleaned.csv', 'r') as input_info:
    attr_info = pd.read_csv(input_info)
attr_info = attr_info[['id', 'language']]
attr_info = attr_info.set_index('id', drop=False)
feature_vector = {}

for edge in original_graph.edges():
    n_1, n_2 = int(edge[0]), int(edge[1])
    l_1, l_2 = attr_info.at[n_1, 'language'], attr_info.at[n_2, 'language']
    if l_1 == l_2:
        vector = [2]
    else:
        vector = [1]
    feature_vector[edge] = vector

# generate extended label_set -- start graph is only 'Democrats' the total graph
# contains both 'Democrats','Republicans'
# and 'Unknown' nodes
with open('./data/extended_graph.graphml') as extended:
    extended_graph = nx.read_graphml(extended)

original_nodes = list(original_graph.nodes())
extended_nodes = list(extended_graph.nodes())


def generate_page_rank_score_plus_derivative_and_optimal_omega():
    # initial_omega = np.empty([1])
    initial_omega = np.array([1])
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