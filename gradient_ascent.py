#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import numpy as np
from numpy import diff
import derivatives_of_the_random_walk


class Algorithm2:

    interation_max = 100

    def __init__(self, learning_rate, u, v,
                 initial_omega, neighbors, Xe, V_L_ext, V):
        self.learning_rate = learning_rate
        self.u = u
        self.v = v
        self.initial_omega = initial_omega
        self.neighbors = neighbors
        self.Xe = Xe
        self.V_L_ext = V_L_ext
        self.V = V

    # TODO generate u, v, initial omega and E (set of Edges)
    # v = [0, 0, 0]
    # u = [0, 0, 0]
    # initial_omega = np.array([])[:, np.newaxis]
    # E = [[0, 0, 0], [0, 0, 0]]

    # TODO: generate neighbors
    def get_neighbors_of_node(self, u):
        neighbors = self.neighbors.get(u)
        if not neighbors:
            neighbors = []
        return neighbors

    # TODO: get Xe(u, v)
    def get_Xe(self, u, v):
        # Xe = {('161148986', '56777838'): [1], ...}
        if self.Xe.get((u, v)):
            Xe_uv = np.array([1])
        else:
            Xe_uv = np.array([0])
        return Xe_uv

    # TODO: generate V_L_ext
    def get_V_L_ext(self):
        return self.V_L_ext

    # TODO: generate V
    def get_V(self):
        return self.V

    def get_a_ui(self, neighbors, u, omega):
        a_u = 0
        for i in neighbors:
            a_u += self.f_omega(np.dot(omega.T, self.get_Xe(u, i)))
        return a_u

    @staticmethod
    def f_omega(x):
        """
        Sigmund Function

        :param x: parameter of the Sigmund Function
        :return: f_omega(x)
        """
        if math.isnan(x):
            f_omega = 0
        else:
            if (1.0 + np.exp(x * (-1))) != 0:
                f_omega = 1.0 / (1.0 + np.exp(x * (-1)))
            else:
                f_omega = 0
        return f_omega

    def get_transition_prob_matrix_Q(self, u, v, omega):  # , E):
        """
        generate the transition probability matrix Q for the edge (u, v).

        :param v:       node v
        :param u:       node u
        :param omega:   feature weight vector omega
        :return:        transition probability matrix Q_uv
        """
        # a_uv = self.f_omega(np.dot(omega.T, self.get_Xe(u, v)[0]))
        if isinstance(u, int):
            u = '0'
        # print("u & v:", type(u), type(v))
        # print("omega and Xe:", omega, self.get_Xe(u, v))
        # print("dot:", np.dot(omega, self.get_Xe(u, v)))

        a_uv = self.f_omega(np.dot(omega, self.get_Xe(u, v)))
        # if u in E and v in E:
        a_ui = self.get_a_ui(self.get_neighbors_of_node(u), u,
                                    omega)
        if a_ui != 0:
            Q_uv = a_uv / a_ui
        else:
            Q_uv = 0
        # else:
        #    Q_uv = 0

        if not Q_uv or math.isnan(Q_uv):
            Q_uv = np.empty([1])

        return Q_uv

    def get_pT(self, alpha, initial_pT):
        """

        :param alpha:       restart probability
        :param Q:           transition probability matrix of node in the graph
        :param initial_pT:  initial vector of visiting probabilities of all
                            nodes
        :return:
        """
        Q = self.get_transition_prob_matrix_Q(
            self.u, self.v, self.initial_omega) #, self.E)
        pT = (1 - alpha) * np.dot(initial_pT, Q) + alpha * 1
        return pT

    def get_j_omega(self, V_L_ext, V):
        """
        Convergence criteria form Algorithm 2.

        :param V_L_ext: extended set of nodes
        :param V:       seed nodes
        :return:        J(omega)
        """
        if self.get_sum_pv_in_V(V) != 0:
            j_omega = self.get_sum_pu_in_V_L_ext(V_L_ext) / \
                  self.get_sum_pv_in_V(V)
        else:
            j_omega = 0
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
        Q = self.get_transition_prob_matrix_Q(self.u, self.v, self.initial_omega) #, self.E)
        # FIXME: index of np array
        try:
            u_index = list(Q).index(u)
        except ValueError:
            # if Q is empty
            u_index = -1

        if u_index == -1:
            # return 0 probability if Q is empty
            return 0
        pu = 0
        for j in range(0, len(Q[u_index])):
            pu += self.p[j] * Q[j][u_index]
        return pu

    def get_pv(self, v):
        """
        generate visiting probability of node v

        :param v:       node v
        :return:        visiting probability of node v
        """
        Q = self.get_transition_prob_matrix_Q(self.u, self.v,
                                         self.initial_omega) #, self.E)
        try:
            v_index = list(Q).index(v)
        except ValueError:
            # if Q is empty
            v_index = -1

        if v_index == -1:
            # return 0 probability if Q is empty
            return 0
        pv = 0
        for j in range(0, len(Q[v_index])):
            pv += self.p[j] * Q[j][v_index]
        return pv

    def gradient_ascent(self):
        """
        generate the optimal feature weight vector (omega), the page rank
        score and the derivative page rank score according to Algorithm 2

        :return:    omega:      optimal omega
                    pT:         page rank score
                    d_pT:       derivative page rank score
        """
        omega = np.empty([1])
        for k in range(0, len(self.initial_omega)):
            omega[k] = 0

        prev_j_omega = -1

        V_L_ext = self.get_V_L_ext()
        V = self.get_V()
        counter = 0

        while self.get_j_omega(V_L_ext, V) != prev_j_omega:
            if counter >= self.interation_max:
                break
            counter += 1
            Q = self.get_transition_prob_matrix_Q(self.u, self.v, omega)
            # compute pT and derivative of pT with Algorithm 3
            pT, d_pT = derivatives_of_the_random_walk.\
                Algorithm3(omega).derivatives_of_the_random_walk(
                self.V, self.u, self.v, Q)
            for k in range(0, len(omega)):
                if k > 0:
                    omega[k] = omega[k - 1] \
                               + (self.learning_rate
                                  * self.get_derivative_of_j_omega(
                        V_L_ext, V, omega))

        return omega, pT, d_pT

if __name__ == '__main__':
    optimal_weight_vector = Algorithm2.gradient_ascent()
