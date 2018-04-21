#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import numpy as np
from numpy import diff


def get_transition_prob_matrix_Q(self, u, v, omega, E):
    """
    generat the transition probability matrix Q for the edge (u, v). The
    matrix is 0 if the edge (u, v) is not in the set of nodes E.

    :param E:       list of edges
    :param v:       node v
    :param u:       node u
    :param omega:   feature weight vector omega
    :return:        transition probability matrix Q_uv
    """

    a_uv = self.f_omega(np.dot(omega.T, self.get_Xe(u, v)))
    if u in E and v in E:
        Q_uv = a_uv / self.get_a_ui(self.get_neighbors_of_node(u), u,
                                    omega)
    else:
        Q_uv = 0

    return Q_uv


class Algorithm2:
    # TODO generate u, v, initial omega and E (set of Edges)
    v = [0, 0, 0]
    u = [0, 0, 0]
    initial_omega = 0
    E = [[0, 0, 0], [0, 0, 0]]
    Q = get_transition_prob_matrix_Q(u, v, initial_omega, E)

    @staticmethod
    def f_omega(x):
        """
        Sigmund Function

        :param x: parameter of the Sigmund Function
        :return: f_omega(x)
        """
        return 1.0 / (1.0 + math.exp(x * (-1)))

    def get_pT(self, alpha, initial_pT):
        """

        :param alpha:       restart probability
        :param Q:           transition probability matrix of node in the graph
        :param initial_pT:  initial vector of visiting probabilities of all
                            nodes
        :return:
        """
        pT = (1 - alpha) * np.dot(initial_pT, self.Q) + alpha * 1
        return pT

    # TODO: generate neighbors
    def get_neighbors_of_node(self, u):
        return []

    def get_a_ui(self, neighbors, u, omega):
        a_u = 0
        for i in neighbors:
            a_u += self.f_omega(np.dot(omega.T, self.get_Xe(u, i)))
        return a_u

    # TODO: get Xe(u, v)
    def get_Xe(self, u, v):
        # empty value
        return np.array([])[:, np.newaxis]

    def get_j_omega(self, V_L_ext, V):
        """
        Convergence criteria form Algorithm 2.

        :param V_L_ext: extended set of nodes
        :param V:       seed nodes
        :return:        J(omega)
        """
        j_omega = self.get_sum_pu_in_V_L_ext(V_L_ext) / \
                  self.get_sum_pv_in_V(V)
        return j_omega

    def get_derivative_of_j_omega(self, V_L_ext, V, omega):
        """
        generate the derivative of j(omega) with the quotient rule

        :param V_L_ext: extended set of nodes
        :param V:       seed nodes
        :param omega:   feature weight vector
        :return:        j'(omega) derived after omega
        """

        derivative_j_omega = \
            ((self.get_sum_derivative_pu_in_V_L_ext(V_L_ext, omega)
              * self.get_sum_pv_in_V(V)) - (
             self.get_sum_pu_in_V_L_ext(V_L_ext)
             * self.get_sum_derivative_pv_in_V(V, omega))) / pow(
                self.get_sum_pv_in_V(V), 2)

        return derivative_j_omega

    def get_sum_pu_in_V_L_ext(self, V_L_ext):
        """
        return the sum of all probabilities of all nodes in the extended set of
        nodes

        :param V_L_ext: extended set of nodes
        :return:        probability sum
        """
        summed_pu = 0
        for u in V_L_ext:
            summed_pu += self.get_pu(u)
        return summed_pu

    def get_sum_derivative_pu_in_V_L_ext(self, V_L_ext, omega):
        """
        return the sum of all derivative of the probabilities of all nodes in
        the extended set of nodes

        :param V_L_ext: extended set of nodes
        :param omega:   probability sum
        :return:
        """
        summed_derivative_pu = 0
        for u in V_L_ext:
            d_omega = omega
            pu = self.get_pu(u)
            d_pu = diff(pu) / d_omega
            summed_derivative_pu += d_pu
        return summed_derivative_pu

    def get_sum_pv_in_V(self, V):
        """
        return the sum of all probabilities of all nodes in the seed node set

        :param V:   seed nodes
        :return:    probability sum
        """
        summed_pv = 0
        for v in V:
            summed_pv += self.get_pv(v)
        return summed_pv

    def get_sum_derivative_pv_in_V(self, V, omega):
        """
        return the sum of all derivative of the probabilities of all nodes in
        the seed node set
        :param V:       seed nodes
        :param omega:   probability sum
        :return:
        """
        summed_derivative_pv = 0
        for v in V:
            d_omega = omega
            pv = self.get_pv(v)
            d_pv = diff(pv) / d_omega
            summed_derivative_pv += d_pv
        return summed_derivative_pv

    def get_pu(self, u):
        """
        generate visiting probability of node u

        :param u:       node u
        :return:        visiting probability of node u
        """

        u_index = self.Q.index(u)
        pu = 0
        for j in range(0, len(self.Q[u_index])):
            pu += self.p[j] * self.Q[j][u_index]
        return pu

    def get_derivative_pu(self, u, omega):
        u_index = self.Q.index(u)
        d_omega = omega
        sum_d_pj_times_Q_ju = 0
        sum_pj_times_d_Q_ju = 0

        for j in range(0, len(self.Q[u_index])):
            p_j = self.p[j]
            d_pj = diff(p_j) / d_omega
            Q_ju = self.Q[j][u]
            sum_d_pj_times_Q_ju += d_pj * Q_ju

        for j in range(0, len(self.Q[u_index])):
            p_j = self.p[j]
            Q_ju = self.Q[j][u]
            d_Q_ju = diff(Q_ju) / d_omega
            sum_pj_times_d_Q_ju += p_j * d_Q_ju

        return sum_d_pj_times_Q_ju + sum_pj_times_d_Q_ju

    def get_pv(self, v):
        """
        generate visiting probability of node v

        :param v:       node v
        :return:        visiting probability of node v
        """
        v_index = self.Q.index(v)
        pv = 0
        for j in range(0, len(self.Q[v_index])):
            pv += self.p[j] * self.Q[j][v_index]
        return pv

    def gradient_ascent(self, learning_rate):
        """
        generate the optimal feature weight vector (omega) according to
        Algorithm 2

        :param learning_rate:   given learning rate
        :return:                optimal omega
        """
        omega = np.array([])[:, np.newaxis]
        for k in range(0, len(self.initial_omega)):
            omega[k] = 0

        prev_j_omega = -1
        threshold = 0.001

        V_L_ext = self.get_V_L_ext()
        V = self.get_V()

        while self.get_j_omega(V_L_ext, V) \
                != prev_j_omega - threshold:
            # TODO compute pT and derivative of p with Algorithm 3
            for k in range(0, len(omega)):
                if k > 0:
                    omega[k] = omega[k - 1] \
                               + (learning_rate * \
                                 self.get_derivative_of_j_omega(
                                     V_L_ext, V))

    # TODO: generate V_L_ext
    def get_V_L_ext(self):
        return []

    # TODO: generate V
    def get_V(self):
        return []


if __name__ == '__main__':
    optimal_weight_vector = Algorithm2.gradient_ascent(0, 0, 3)
