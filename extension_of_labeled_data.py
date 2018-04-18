import networkx as nx
import math

cond_on_iteration = {}

def calculate_condactance(iteration, A_pos, B_neg, graph):
    E_ab = amount_of_edges_between(graph, A_pos, B_neg)
    E_a = E_ab + len(A_pos.nodes())
    E_b = E_ab + len(B_neg.nodes())
    cond_on_iteration[iteration] = E_ab/min(E_a,E_b)

def amount_of_edges_between(graph, A_pos, B_neg):
    return 1


def stop_criterion_reached(iteration_no):
    left = (cond_on_iteration[iteration_no] - cond_on_iteration[math.floor(iteration_no/2)])/iteration_no-math.floor(iteration_no/2)
    right_top = 0.5 * (cond_on_iteration[math.floor(iteration_no/2)] - cond_on_iteration[0])
    right_bottom = math.floor(iteration_no/2)
    right = right_top if right_bottom == 0 else right_top/right_bottom
    return left < right


def extend_labeled_data(total_graph):
    total_graph = nx.Graph()
    A_pos = total_graph.nodes()[1]['positive'] = True
    B_neg = total_graph.nodes() - A_pos

    i = 0
    while not stop_criterion_reached(i):
        all_nodes = total_graph.nodes()
        argmax = {}
        for node in all_nodes:
            if node not in sub_a_pos:
                A_pos.append(node)
                sub_a_pos = total_graph.subgraph(A_pos)
                B_neg.pop(node)
                sub_b_neg = total_graph.subgraph(B_neg)
                calculate_condactance(i,sub_a_pos,sub_b_neg)
                argmax[node] = cond_on_iteration[i]
        b = max(argmax.items(), key=lambda key: argmax[key])

        A_pos.append(b)
        B_neg.pop(b)




