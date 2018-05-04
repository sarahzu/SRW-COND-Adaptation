#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import numpy as np
from numpy import diff
import derivatives_of_the_random_walk


class Algorithm2:
    iteration_max = 5

    def __init__(self, learning_rate, initial_omega, neighbors, Xe, V_L_ext, V):
        self.learning_rate = learning_rate
        self.initial_omega = initial_omega
        self.neighbors = neighbors
        self.Xe = Xe
        self.V_L_ext = V_L_ext
        self.V = V

    def generate_full_transition_probability_matrix_Q(self, V, omega):
        """
        create the full transition probability matrix for all nodes in the
        given list

        :param V:       given list of nodes
        :param omega:   given omega for the computation
        :return:        full transition probability matrix
        """
        Q = np.full((len(V), len(V)), 1)
        Q = Q.astype(float)
        for u, index_u in zip(V, range(0, len(V))):
            for v, index_v in zip(V, range(0, len(V))):
                Q_uv = self.get_transition_prob_matrix_Q(u, v, omega)
                Q[index_u][index_v] = Q_uv

        return Q

    def get_neighbors_of_node(self, u):
        """
        Return the list of neighbors for a given node u

        :param u:   given node
        :return:    list of all neighbors
        """
        neighbors = self.neighbors.get(int(u))
        if not neighbors:
            neighbors = []
        return neighbors

    def get_Xe(self, u, v):
        """
        get the feature Vector of two nodes (Xe_uv). The full feature vector Xe
        e.g. looks like this:
        {('161148986', '56777838'): [1,1,1], ...}

        :param u:   node u
        :param v:   node v
        :return:    feature vector list of node u and v
        """
        Xe_uv = self.Xe.get((int(u), int(v)))
        if not Xe_uv:
            Xe_uv = np.array([0, 0, 0])
        return Xe_uv

    def get_V_L_ext(self):
        """
        get the extended seed set

        :return: extended seed set
        """
        return self.V_L_ext

    def get_V(self):
        """
        get the original seed set

        :return: original seed set
        """
        return self.V

    def get_a_ui(self, neighbors, u, omega):
        """
        generate weight (a) of node u and all its neighbors

        :param neighbors:   list of neighbors of node u
        :param u:           given node
        :param omega:       feature weigh vector
        :return:            weight of node u and its neighbors
        """
        a_u = 0
        for i in neighbors:
            i = int(i)
            a_u += self.f_omega(np.dot(omega.T, self.get_Xe(u, i)))
        return a_u

    @staticmethod
    def f_omega(x):
        """
        Sigmund Function

        :param x:   parameter of the Sigmund Function
        :return:    f_omega(x)
        """
        if math.isnan(x):
            f_omega = 0
        else:
            if (1.0 + np.exp(x * (-1))) != 0:
                f_omega = 1.0 / (1.0 + np.exp(-x))
            else:
                f_omega = 0
        return f_omega

    def d_f_omega(self, x):
        """
        Derivative Sigmund Function

        :param x: function parameter
        :return:  derivative sigmund function result
        """
        if math.isnan(x):
            d_f_omega = 0
        else:
            if (1.0 + np.exp(x * (-1))) != 0:
                d_f_omega = (1.0 / (1.0 + np.exp(-x))) \
                            - np.power((1.0 / (1.0 + np.exp(-x))), 2)
            else:
                d_f_omega = 0
        return d_f_omega

    def get_transition_prob_matrix_Q(self, u, v, omega):  # , E):
        """
        generate the transition probability matrix Q for the edge (u, v).

        :param v:       node v
        :param u:       node u
        :param omega:   feature weight vector omega
        :return:        transition probability matrix Q_uv
        """
        if isinstance(u, int):
            u = '0'
        if isinstance(v, int):
            v = '0'
        # generate weight of u and v
        a_uv = self.f_omega(np.dot(omega.T, self.get_Xe(u, v)))

        # generate weight of u and al its neighbors
        a_ui = self.get_a_ui(self.get_neighbors_of_node(u), u, omega)

        if a_ui != 0:
            Q_uv = a_uv / a_ui
        else:
            # TODO
            Q_uv = a_uv

        #       if not Q_uv or math.isnan(Q_uv):
        #       Q_uv = 0

        return Q_uv

    def get_j_omega(self, V_L_ext, V, omega, Q, p):
        """
        Convergence criteria form Algorithm 2.

        :param V_L_ext: extended set of nodes
        :param V:       seed nodes
        :param omega:   feature weight vector
        :param Q:       transition probability matrix
        :param p:       page rank score vector
        :return:        J(omega)
        """
        sum_pv_in_V = self.get_sum_pv_in_V(V, omega, Q, p)
        sum_pu_in_V_L_ext = self.get_sum_pu_in_V_L_ext(V_L_ext, omega, Q, p)
        if sum_pv_in_V != 0:
            j_omega = sum_pu_in_V_L_ext / sum_pv_in_V
        else:
            j_omega = 0.0
        return j_omega

    def get_derivative_of_j_omega(self, V_L_ext, V, omega, Q, p, d_p):
        """
        generate the derivative of j(omega) with the quotient rule

        :param V_L_ext: extended set of nodes
        :param V:       seed nodes
        :param omega:   feature weight vector
        :param Q:       transition probability matrix
        :param p:       page rank score vector
        :param d_p:     derivative page rank score vector
        :return:        j'(omega) derived after omega
        """
        sum_derivative_pu_in_V_L_ext = self.get_sum_derivative_pv_in_V(V_L_ext, omega, Q, d_p)
        sum_pv_in_V = self.get_sum_pv_in_V(V, omega, Q, p)
        sum_pu_in_V_L_ext = self.get_sum_pv_in_V(V_L_ext, omega, Q, p)
        sum_derivative_pv_in_V = self.get_sum_derivative_pv_in_V(V, omega, Q, d_p)

        derivative_j_omega = \
            ((sum_derivative_pu_in_V_L_ext * sum_pv_in_V) - (sum_pu_in_V_L_ext
                                                             * sum_derivative_pv_in_V)) / np.power(sum_pv_in_V, 2)

        return derivative_j_omega

    def get_sum_pu_in_V_L_ext(self, V_L_ext, omega, Q, p):
        """
        return the sum of all probabilities of all nodes in the extended set of
        nodes

        :param V_L_ext: extended set of nodes
        :param omega:   feature weight vector
        :param Q:       transition probability matrix
        :param p:       page rank score vector
        :return:        probability sum
        """
        summed_pu = 0.0
        for u, index_u in zip(V_L_ext, range(0, len(V_L_ext))):
            summed_pu += self.get_pu(index_u, V_L_ext, omega, Q, p)
        return summed_pu

    def get_sum_derivative_pu_in_V_L_ext(self, V_L_ext, omega, Q):
        """
        return the sum of all derivative of the probabilities of all nodes in
        the extended set of nodes

        :param V_L_ext: extended set of nodes
        :param omega:   feature weigh vector
        :param Q:       transition probability matrix
        :return:        sum of all derivatives of all nodes in the extended seed set
        """
        summed_derivative_pu = 0
        for u_index in range(0, len(V_L_ext)):
            d_omega = omega
            pu = self.get_pu(u_index, V_L_ext, omega, Q)
            if d_omega != 0:
                d_pu = diff(pu) / d_omega
            else:
                d_pu = pu
            summed_derivative_pu += d_pu
        return summed_derivative_pu

    def get_sum_pv_in_V(self, V, omega, Q, p):
        """
        return the sum of all probabilities of all nodes in the seed node set

        :param V:       seed nodes
        :param omega:   feature weight vector
        :param Q:       transition probability matrix
        :param p:       page rank score vector
        :return:        probability sum
        """
        summed_pv = 0.0
        for index_v in range(0, len(V)):
            # print("v in V:", v)
            summed_pv += self.get_pu(index_v, V, omega, Q, p)
        return summed_pv

    def get_sum_derivative_pv_in_V(self, V, omega, Q, d_p):
        """
        return the sum of all derivative of the probabilities of all nodes in
        the seed node set

        :param V:       seed nodes
        :param omega:   feature weight vector
        :param Q:       transition probability matrix
        :param d_p:     derivative page rank score vector
        :return:        probability sum
        """
        summed_derivative_pv = 0
        for v_index in range(0, len(V)):
            d_pu = self.get_pu(v_index, V, omega, Q, d_p)
            summed_derivative_pv += d_pu[0]
        return summed_derivative_pv

    def get_pu(self, u_index, V, omega, Q, p):
        """
        generate visiting probability of node v

        :param u_index:     index of node u
        :param V:           list of all nodes
        :param omega:       feature weight vector
        :param Q            transition probability matrix
        :param p:           page rank score vector
        :return:            visiting probability of node v
        """
        pu = 0

        for index_j in range(0, len(V)):
            try:
                Q_ju = Q[index_j][int(u_index)]
            except IndexError:
                Q_ju = 0
            try:
                p_j = p[index_j]
            except IndexError:
                p_j = 0
            pu += p_j * Q_ju
        return pu

    def gradient_ascent(self):
        """
        generate the optimal feature weight vector (omega)

        :return:    optimal feature weight vector

        """
        omega = self.initial_omega

        prev_j_omega = -1

        V_L_ext = self.get_V_L_ext()
        V = self.get_V()
        counter = 0

        Q = self.generate_full_transition_probability_matrix_Q(V_L_ext, omega)
        p = np.zeros([len(V_L_ext)])
        j_omega = self.get_j_omega(V_L_ext, V, omega, Q, p)

        while j_omega != prev_j_omega:
            prev_j_omega = j_omega
            if counter >= self.iteration_max:
                break
            counter += 1

            Q = self.generate_full_transition_probability_matrix_Q(V_L_ext, omega)

            # compute pT and derivative of pT with Algorithm 3
            p, d_p = derivatives_of_the_random_walk.Algorithm3(
                omega, self.Xe, self.neighbors).derivatives_of_the_random_walk(
                self.V_L_ext, Q)

            for k in range(0, len(omega)):
                if k > 0:
                    omega[k] = omega[k - 1] \
                               + (self.learning_rate
                                  * self.get_derivative_of_j_omega(

                                V_L_ext, V, omega, Q, p, d_p))
                else:
                    omega[k] = self.learning_rate \
                               * self.get_derivative_of_j_omega(
                        V_L_ext, V, omega, Q, p, d_p)

            Q = self.generate_full_transition_probability_matrix_Q(V, omega)
            j_omega = self.get_j_omega(V_L_ext, V, omega, Q, p)

        return omega


if __name__ == '__main__':
    pass
