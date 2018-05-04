import networkx as nx
import math
import operator
import community


def create_labeled_subgraph(graph):
    """
    Creates a subgraph containing only the nodes that have a value different from "Unknown" under leaning
    :return: SubGraph
    """
    labeled_nodes = []
    for (p, d) in graph.nodes(data=True):
        if (d['leaning'] == 'R') or (d['leaning'] == 'D'):
            labeled_nodes.append(p)
    return nx.subgraph(graph, labeled_nodes)


def create_unlabeled_subgraph(graph):
    """
    Creates a subgraph containing only the nodes that have a value equals to "Unknown" under leaning
    :return: SubGraph
    """
    unlabeled_nodes = []
    for (p, d) in graph.nodes(data=True):
        if (d['leaning'] == 'Unknown'):
            unlabeled_nodes.append(p)
    return nx.subgraph(graph, unlabeled_nodes)


cond_on_iteration = {}


def stop_criterion_reached(iteration_no):
    """
    Defines the stop criterion for the Conductance loop
    :param iteration_no:
    :return: Boolean value whether to stop or to continue
    """
    C_t = cond_on_iteration[iteration_no]
    C_0 = cond_on_iteration[0]
    C_2_f = cond_on_iteration[math.floor(iteration_no / 2)]
    if iteration_no == 0:
        left_below = 1
        right_below = 1
    elif iteration_no == 1:
        right_below = 1
        left_below = iteration_no - math.floor(iteration_no / 2)
    else:
        left_below = iteration_no - math.floor(iteration_no / 2)
        right_below = math.floor(iteration_no / 2)

    left = 100 * (C_t - C_2_f) / left_below
    right = (C_2_f - C_0) / right_below
    return left < right


def extend_labeled_data(input_path, output_path):
    """
    Extends the graph under input_path based on the Conductance metric and saves the extended graph under output_path
    :param input_path: Location of original graph
    :param output_path: Location of output extended graph
    """
    total_graph = nx.read_graphml(input_path)
    A_pos = create_labeled_subgraph(total_graph)
    sub_a_pos = list(A_pos.nodes(data=True))
    B_neg = create_unlabeled_subgraph(total_graph)
    sub_b_neg = list(B_neg.nodes(data=True))
    i = 0
    if i == 0:
        cond_on_iteration[i] = nx.conductance(total_graph, A_pos)
    while not stop_criterion_reached(i):
        i += 1
        argmax = {}
        nodes_to_iter = set()
        for node in sub_a_pos:
            if node[1]['leaning'] == 'R':
                neighbors = nx.neighbors(total_graph, node[0])
                strong_edge_neighbors = [n_2 for (n_1, n_2, w) in total_graph.edges(node[0], data=True) if
                                         w['weight'] > 1]
                clean_neighbors = []
                for neighbor in neighbors:
                    if neighbor in [b[0] for b in sub_b_neg]:
                        if neighbor in strong_edge_neighbors:
                            clean_neighbors.append(neighbor)
                nodes_to_iter.update(clean_neighbors)

        counter = 0
        for node in nodes_to_iter:
            counter += 1
            c = math.floor(counter / len(nodes_to_iter) * 100)
            temp = [x[0] for x in sub_a_pos]
            temp.append(node)
            argmax[node] = nx.conductance(total_graph, temp)
            print(str(c) + '%')
        b = max(argmax.items(), key=operator.itemgetter(1))[0]
        cond_on_iteration[i] = argmax[b]
        b_data = [(x, y) for (x, y) in sub_b_neg if x == b][0]
        sub_a_pos.append(b_data)
        sub_b_neg.remove(b_data)
        print('Finished with ' + str(len(sub_a_pos)) + ' and ' + str(len(sub_b_neg)) + 'nodes')
    extended_graph = nx.subgraph(total_graph, [x for (x, y) in sub_a_pos])
    nx.write_graphml(extended_graph, output_path)


def extend_labeled_graph(graph):
    """
    Alterative approach based on the louvain algorithm for community detection.
    :param graph: Takes a graph as an input
    :return: Two graphs, original and extended
    """
    la = community.best_partition(graph)
    nx.set_node_attributes(graph, la, 'community')
    nodes = graph.nodes(data=True)
    # nx.write_graphml(graph,'./data_2/clean_data/comm_graph.graphml')

    a = list(set(list(la.values())))
    temp = {}
    for comm in a:
        temp[comm] = [k for k, v in la.items() if v == comm]

    s = sorted(temp, key=lambda k: len(temp[k]), reverse=True)[:10]
    comm_size = {}
    for key in s:
        if key in temp:
            comm_size[key] = temp[key]

    dict_leaning_amount = {}
    for comm, ids in comm_size.items():
        count_r = 0
        for node in ids:
            if graph.node[node]['leaning'] == 'R':
                count_r += 1
        dict_leaning_amount[comm] = count_r
    sort_lean = sorted(dict_leaning_amount.items(), key=operator.itemgetter(1), reverse=True)
    top_3 = [k for k, v in sort_lean][0:3]

    extendible_nodes = []
    for comm in top_3:
        nodes = temp[comm]
        for node in nodes:
            if graph.node[node]['leaning'] == 'Unknown':
                extendible_nodes.append(node)

    original_graph = create_labeled_subgraph(graph)
    extendible_nodes.extend(list(create_labeled_subgraph(graph).nodes()))
    extendible_node_Set = set(extendible_nodes)

    extended_graph = nx.subgraph(graph, list(extendible_node_Set))
    return original_graph, extended_graph
