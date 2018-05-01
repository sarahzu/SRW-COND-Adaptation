#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import numpy as np
from numpy import diff
import derivatives_of_the_random_walk


class Algorithm2:

    iteration_max = 2

    def __init__(self, learning_rate, u, v, index_u, index_v,
                 initial_omega, neighbors, Xe, V_L_ext, V):
        self.learning_rate = learning_rate
        self.u = u
        self.v = v
        self.index_u = index_u
        self.index_v = index_v
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
        V_without_first_element = V[1:]
        Q = np.empty([len(V), len(V)])
        for u, index_u in zip(V, range(0, len(V))):
            for v, index_v in zip(V_without_first_element, range(0, len(V_without_first_element))):
                Q_uv = self.get_transition_prob_matrix_Q(u, v, omega)
                Q[index_u][index_v] = Q_uv
                # print(Q_uv)

        return Q

    def get_neighbors_of_node(self, u):
        neighbors = self.neighbors.get(int(u))
        if not neighbors:
            neighbors = []
        return neighbors

    def get_Xe(self, u, v):
        # Xe = {('161148986', '56777838'): [1,1,1], ...}

        # if self.Xe.get((u, v)):
        #     Xe_uv = self.Xe.get((u, v))
        # else:
        #     Xe_uv = np.array([0, 0, 0])
        Xe_uv = self.Xe.get((int(u), int(v)))
        if not Xe_uv:
            Xe_uv = np.array([0, 0, 0])
        return Xe_uv

    def get_V_L_ext(self):
        return self.V_L_ext

    def get_V(self):
        return self.V

    def get_a_ui(self, neighbors, u, omega):
        a_u = 0
        for i in neighbors:
            i = int(i)
            a_u += self.f_omega(np.dot(omega.T, self.get_Xe(u, i)))
            # if a_u != 0:
                # print("omega:", omega, "\nXe:", self.get_Xe(u, i), "\na_u:", a_u)
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
        if isinstance(v, int):
            v = '0'
        # print("u & v:", u, v)
        # print("omega and Xe:", omega, self.get_Xe(u, v))
        # print("dot:", np.dot(omega, self.get_Xe(u, v)))
        # print("a_omega:", omega.T)
        # print("a_Xe_uv", self.get_Xe(u, v))
        a_uv = self.f_omega(np.dot(omega.T, self.get_Xe(u, v)))
        # print("a_uv:", a_uv)
        # if u in E and v in E:
        a_ui = self.get_a_ui(self.get_neighbors_of_node(u), u, omega)

        if a_ui != 0:
            Q_uv = a_uv / a_ui
        else:
            Q_uv = 0

        if not Q_uv or math.isnan(Q_uv):
            Q_uv = 0

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

    def get_j_omega(self, V_L_ext, V, omega, Q):
        """
        Convergence criteria form Algorithm 2.

        :param V_L_ext: extended set of nodes
        :param V:       seed nodes
        :return:        J(omega)
        """
        sum_pv_in_V = self.get_sum_pv_in_V(V, omega, Q)
        sum_pu_in_V_L_ext = self.get_sum_pu_in_V_L_ext(V_L_ext, omega, Q)
        if sum_pv_in_V != 0:
            j_omega = sum_pu_in_V_L_ext / sum_pv_in_V
        else:
            j_omega = 0
        return j_omega

    def get_derivative_of_j_omega(self, V_L_ext, V, omega, Q):
        """
        generate the derivative of j(omega) with the quotient rule

        :param V_L_ext: extended set of nodes
        :param V:       seed nodes
        :param omega:   feature weight vector
        :return:        j'(omega) derived after omega
        """
        sum_derivative_pu_in_V_L_ext = self.get_sum_derivative_pv_in_V(V_L_ext, omega, Q)
        sum_pv_in_V = self.get_sum_pv_in_V(V, omega, Q)
        sum_pu_in_V_L_ext = self.get_sum_pv_in_V(V_L_ext, omega, Q)
        sum_derivative_pv_in_V = self.get_sum_derivative_pv_in_V(V, omega, Q)

        derivative_j_omega = \
            ((sum_derivative_pu_in_V_L_ext * sum_pv_in_V) - (sum_pu_in_V_L_ext
             * sum_derivative_pv_in_V)) / pow(sum_pv_in_V, 2)

        return derivative_j_omega

    def get_sum_pu_in_V_L_ext(self, V_L_ext, omega, Q):
        """
        return the sum of all probabilities of all nodes in the extended set of
        nodes

        :param V_L_ext: extended set of nodes
        :return:        probability sum
        """
        summed_pu = 0
        for u, index_u in zip(V_L_ext, range(0, len(V_L_ext))):
            summed_pu += self.get_pu(index_u, V_L_ext, omega, Q)
        return summed_pu

    def get_sum_derivative_pu_in_V_L_ext(self, V_L_ext, omega, Q):
        """
        return the sum of all derivative of the probabilities of all nodes in
        the extended set of nodes

        :param V_L_ext: extended set of nodes
        :param omega:   probability sum
        :return:
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

    def get_sum_pv_in_V(self, V, omega, Q):
        """
        return the sum of all probabilities of all nodes in the seed node set

        :param V:   seed nodes
        :return:    probability sum
        """
        summed_pv = 0
        for index_v in range(0, len(V)):
            # print("v in V:", v)
            summed_pv += self.get_pu(index_v, V, omega, Q)
        return summed_pv

    def get_sum_derivative_pv_in_V(self, V, omega, Q):
        """
        return the sum of all derivative of the probabilities of all nodes in
        the seed node set
        :param V:       seed nodes
        :param omega:   probability sum
        :return:
        """
        summed_derivative_pv = 0
        for v_index in range(0, len(V)):
            d_omega = omega[0] #np.gradient(omega)
            print(d_omega)
            pv = self.get_pu(v_index, V, omega, Q)
            if d_omega != 0:
                d_pv = diff(pv) / d_omega
            else:
                d_pv = pv
            summed_derivative_pv += d_pv
        return summed_derivative_pv

    # def get_pu(self, u):
    #     """
    #     generate visiting probability of node u
    #
    #     :param u:       node u
    #     :return:        visiting probability of node u
    #     """
    #     Q = self.get_transition_prob_matrix_Q(self.u, self.v, self.initial_omega) #, self.E)
    #     # FIXME: index of np array
    #     try:
    #         print("list Q:", list(Q))
    #         u_index = list(Q).index(u)
    #     except ValueError:
    #         # if Q is empty
    #         u_index = -1
    #
    #     if u_index == -1:
    #         # return 0 probability if Q is empty
    #         return 0
    #     pu = 0
    #     for j in range(0, len(Q[u_index])):
    #         pu += self.p[j] * Q[j][u_index]
    #     return pu

    def get_pu(self, u_index, V, omega, Q):
        """
        generate visiting probability of node v

        :param u:       node v
        :return:        visiting probability of node v
        """
        # Q = self.get_transition_prob_matrix_Q(self.u, self.v,
        #                                 self.initial_omega) #, self.E)
        # try:
        #     print("QQ:", Q)
        #     v_index = list(Q).index(v)
        # except ValueError:
        #     # if Q is empty
        #     return 0

        # print("u:", u)

        pu = 0
        # Q = self.generate_full_transition_probability_matrix_Q(V, omega)
        # for j in range(0, len(V)):
        # print("QQ:", Q)
        for index_j in range(0, len(V)-1):
            # pv += self.p[j] * self.get_transition_prob_matrix_Q(j, v, omega)
            try:
                Q_ju = Q[index_j][int(u_index)]
            except IndexError:
                Q_ju = 0
            pu += Q_ju
            # self.get_transition_prob_matrix_Q(j, u, omega)
        # print("pu:", pu)
        return pu

    def gradient_ascent(self):
        """
        generate the optimal feature weight vector (omega), the page rank
        score and the derivative page rank score according to Algorithm 2

        :return:    omega:      optimal omega
                    pT:         page rank score
                    d_pT:       derivative page rank score
        """
        omega = np.zeros([len(self.initial_omega)])
        #for k in range(0, len(omega)):
        #    omega[k] = 0

        prev_j_omega = -1

        V_L_ext = self.get_V_L_ext()
        V = self.get_V()
        counter = 0

        Q = self.generate_full_transition_probability_matrix_Q(V, omega)

        while self.get_j_omega(V_L_ext, V, omega, Q) != prev_j_omega:
            if counter >= self.iteration_max:
                break
            counter += 1
            # print("u:", type(self.u), "v:", type(self.v))
            # Q = self.get_transition_prob_matrix_Q(self.u, self.v, omega)
            # compute pT and derivative of pT with Algorithm 3
            # pT, d_pT = derivatives_of_the_random_walk.\
            #     Algorithm3(omega, self.Xe, self.neighbors).\
            #     derivatives_of_the_random_walk(self.V, self.u, Q)
            for k in range(0, len(omega)):
                if k > 0:
                    omega[k] = omega[k - 1] \
                               + (self.learning_rate
                                  * self.get_derivative_of_j_omega(
                        V_L_ext, V, omega, Q))
                    print(omega)
                else:
                    omega[k] = self.learning_rate \
                               * self.get_derivative_of_j_omega(
                        V_L_ext, V, omega, Q)

            Q = self.generate_full_transition_probability_matrix_Q(V, omega)

        return omega

if __name__ == '__main__':
    optimal_weight_vector = Algorithm2.gradient_ascent()
