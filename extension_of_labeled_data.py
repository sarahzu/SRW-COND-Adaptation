import networkx as nx
import math
import operator


def import_graphml():
    with open('./data/graph.graphml', 'r') as inp:
        return nx.read_graphml(inp)


def write_graphml(graph, output_path):
    print("Writing Graph to " + str(output_path))
    nx.write_graphml(graph, output_path)


def create_labeled_subgraph(graph):
    labeled_nodes = []
    for (p, d) in graph.nodes(data=True):
        if (d['leaning'] == 'DEMOCRAT'):
            labeled_nodes.append(p)
    return nx.subgraph(graph, labeled_nodes)


def create_unlabeled_subgraph(graph):
    unlabeled_nodes = []
    for (p, d) in graph.nodes(data=True):
        if (d['leaning'] == 'Unknown') or (d['leaning'] == 'REPUBLICAN'):
            unlabeled_nodes.append(p)
    return nx.subgraph(graph, unlabeled_nodes)



cond_on_iteration = {}


def stop_criterion_reached(iteration_no):
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

    left = (C_t - C_2_f) / left_below
    right = 0.5 * (C_2_f - C_0) / right_below
    print('Difference between '+str(right)+'-'+str(left)+'=' + str(right-left))
    return left < right


def extend_labeled_data():
    total_graph = nx.read_graphml('./data/complete_graph.graphml')
    all_nodes = total_graph.nodes()
    A_pos = create_labeled_subgraph(total_graph)
    sub_a_pos = list(A_pos.nodes())
    nx.write_graphml(A_pos,'./data/original_graph.graphml')
    B_neg = create_unlabeled_subgraph(total_graph)
    sub_b_neg = list(B_neg.nodes())
    i = 0
    if i == 0:
        cond_on_iteration[i] = nx.conductance(total_graph,A_pos)
    while not stop_criterion_reached(i):
        i += 1
        argmax = {}
        for node in all_nodes:
            if node not in sub_a_pos:
                sub_a_pos.append(node)
                sub_b_neg.remove(node)
                argmax[node] = nx.conductance(total_graph, sub_a_pos)
                sub_b_neg.append(node)
                sub_a_pos.remove(node)
        b = max(argmax.items(), key=operator.itemgetter(1))[0]
        t = min(argmax.items(), key=operator.itemgetter(1))[0]
        cond_on_iteration[i] = argmax[b]
        sub_a_pos.append(b)
        sub_b_neg.remove(b)
        print('Finished with ' + str(len(sub_a_pos))+ ' and ' + str(len(sub_b_neg)) +  'nodes')
    extended_graph = nx.subgraph(total_graph,sub_a_pos)
    return extended_graph


ext_graph = extend_labeled_data()
nx.write_graphml(ext_graph,'./data/extended_graph.graphml')
