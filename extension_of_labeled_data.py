import networkx as nx
import math
import operator


def import_graphml():
    with open('./data/test_graph.graphml', 'r') as inp:
        return nx.read_graphml(inp)


def write_graphml(graph, output_path):
    print("Writing Graph to " + str(output_path))
    nx.write_graphml(graph, output_path)


def create_labeled_subgraph(graph):
    labeled_nodes = []
    for (p, d) in graph.nodes(data=True):
        if (d['leaning'] == 'DEMOCRAT') or (d['leaning'] == 'REPUBLICAN'):
            labeled_nodes.append(p)
    return nx.subgraph(graph, labeled_nodes)


def create_unlabeled_subgraph(graph):
    unlabeled_nodes = []
    for (p, d) in graph.nodes(data=True):
        if d['leaning'] == 'Unknown':
            unlabeled_nodes.append(p)
    return nx.subgraph(graph, unlabeled_nodes)


cond_on_iteration = {}


def calculate_condactance(iteration, A_pos, B_neg, graph):
    E_ab = amount_of_edges_between(graph, A_pos, B_neg)
    E_a = E_ab + len(A_pos.nodes())
    E_b = E_ab + len(B_neg.nodes())
    return E_ab / min(E_a, E_b)


def amount_of_edges_between(graph, A_pos, B_neg):
    count = 0
    edges_total = list(graph.edges())
    nodes_a = list(A_pos.nodes())
    nodes_b = list(B_neg.nodes())
    for edge in edges_total:
        if edge[0] in nodes_a:
            if edge[1] in nodes_b:
                count += 1
        elif edge[0] in nodes_b:
            if edge[1] in nodes_a:
                count += 1

    # q = len([a for a in list(A_pos.edges()) if a not in list(graph.edges())])
    # t = len([b for b in list(B_neg.edges()) if b not in list(graph.edges)])
    # a = len(list(graph.edges()))-(q+t)
    # t = len(list(graph.edges()).remove(list(A_pos.edges())).remove(list(B_neg.edges())))
    return count


def stop_criterion_reached(iteration_no):
    below = iteration_no - math.floor(iteration_no / 2)
    if below == 0:
        below = 1
    left = ((cond_on_iteration[iteration_no] - cond_on_iteration[math.floor(iteration_no / 2)]) / below)

    right_top = 0.5 * (cond_on_iteration[math.floor(iteration_no / 2)] - cond_on_iteration[0])
    right_bottom = math.floor(iteration_no / 2)
    right = right_top if right_bottom == 0 else right_top / right_bottom
    print("right:" + str(right) + " - left: " + str(left))
    return left < right


def extend_labeled_data(total_graph):
    all_nodes = total_graph.nodes()
    A_pos = create_labeled_subgraph(total_graph)
    sub_a_pos = list(A_pos.nodes())
    B_neg = create_unlabeled_subgraph(total_graph)
    sub_b_neg = list(B_neg.nodes())

    temp_count = 0

    i = 0
    if i == 0:
        cond_on_iteration[i] = calculate_condactance(0, A_pos, B_neg, total_graph)
    while not stop_criterion_reached(i):
        i += 1
        argmax = {}
        for node in all_nodes:
            temp_count += 1
            print(temp_count / len(all_nodes) * 100)
            if node not in sub_a_pos:
                sub_a_pos.append(node)
                A_pos_temp = total_graph.subgraph(sub_a_pos)
                sub_a_pos.remove(node)
                sub_b_neg.remove(node)
                B_neg_temp = total_graph.subgraph(sub_b_neg)
                sub_b_neg.append(node)
                argmax[node] = calculate_condactance(i, A_pos_temp, B_neg_temp, total_graph)
        b = max(argmax.items(), key=operator.itemgetter(1))[0]
        cond_on_iteration[i] = argmax[b]
        print("finished----" + str(cond_on_iteration[i]))
        sub_a_pos.append(b)
        sub_b_neg.remove(b)


extend_labeled_data(import_graphml())
