#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from numpy import diff
import math


class Algorithm3:

    iteration_max = 5

    def __init__(self, omega, Xe, neighbors):
        self.omega = omega
        self.Xe = Xe
        self.neighbors = neighbors

    def derivatives_of_the_random_walk(self, V, Q):
        """
        Compute the page rank scores in form of vector p and its
        derivatives in form of vector d_p.

        :param V:           seed set
        :param Q:           transition probability matrix
        :return:    p:      page rank score vector
                    d_p:    derivative page rank score vector
        """
        # page rank
        p = np.zeros([len(V)])
        # derivative page rank
        d_p = np.zeros([len(V), len(self.omega)])

        if len(V) != 0:
            for u_index in range(0, len(V)):
                p[u_index] = 1 / len(V)
        else:
            for u_index in range(0, len(V)):
                p[u_index] = 0

        for u_index, k in zip(range(0, len(V)), range(0, len(self.omega))):
            d_p[u_index][k] = 0

        prev_p = np.empty([len(V)])
        prev_d_p = np.empty([len(V), len(self.omega)])
        first = True
        counter = 0

        # p
        while not np.array_equal(p, prev_p) or first:
            if counter >= self.iteration_max:
                break
            counter += 1
            first = False
            prev_p = p
            for u_index in range(0, len(V)):
                if prev_p is not None:
                    p[u_index] = self.get_pj_and_Qju_sum(prev_p, Q, u_index)
                else:
                    p[u_index] = 0

        # derivative d_p
        for k in range(0, len(self.omega)):
            first = True
            counter = 0
            while not np.array_equal(d_p, prev_d_p) or first:
                if counter >= self.iteration_max:
                    break
                counter += 1
                prev_d_p = d_p
                for u_index in range(0, len(V)):
                    d_p[u_index] = self.get_derivative_pu(V[u_index], u_index,
                                                          self.omega, p, Q)
                first = False
        return p, d_p

    def get_derivative_pu(self, u, u_index, omega, p, Q):
        """
        get the derivative of the visiting probability of node u.

        :param u:           given node
        :param u_index:     index of given node
        :param omega:       feature weight vector
        :param p:           page rank score vector
        :param Q:           transition probability matrix
        :return:
        """

        sum_d_pj_times_Q_ju = 0
        sum_pj_times_d_Q_ju = 0

        for j in range(0, len(p)):
            p_j = p[j]
            Q_ju = Q[j][u_index]
            sum_d_pj_times_Q_ju += p_j * Q_ju

        for j in range(0, len(p)):
            p_j = p[j]
            d_Q_ju = self.get_d_Q_ju(j, u, self.neighbors, omega)
            sum_pj_times_d_Q_ju += p_j * d_Q_ju

        return sum_d_pj_times_Q_ju + sum_pj_times_d_Q_ju

    def get_d_Q_ju(self, j, u, neighbors, omega):
        """
        Compute the derivative of the transition matrix at position (j, u)
        (Q_ju)

        :param j:           given node j
        :param u:           given node u
        :param neighbors:   list of neighbors of node u
        :param omega:       feature weight vector
        :return:            derivative of Qju
        """
        d_Q_ju = 0
        sum_f_omega_of_feature_vector_of_neighbors = 0
        for i in neighbors:
            sum_f_omega_of_feature_vector_of_neighbors += self.f_omega(np.dot(omega.T, self.get_Xe(u, i)))

        f_omega_of_omega_and_Xe_ju = \
            self.f_omega(np.dot(omega.T, self.get_Xe(j, u)))

        d_f_omega_of_omega_and_Xe_ju = self.d_f_omega(np.dot(omega.T, self.get_Xe(j, u)))

        first_formula_part = \
            (d_f_omega_of_omega_and_Xe_ju
             * sum_f_omega_of_feature_vector_of_neighbors) / \
            np.power(sum_f_omega_of_feature_vector_of_neighbors, 2)

        sum_d_f_omega_of_feature_vector_of_neighbors = 0
        for i in neighbors:
            sum_d_f_omega_of_feature_vector_of_neighbors += self.d_f_omega(np.dot(omega.T, self.get_Xe(j, i)))

        second_formula_part = f_omega_of_omega_and_Xe_ju \
                              * sum_d_f_omega_of_feature_vector_of_neighbors / \
                              np.power(sum_d_f_omega_of_feature_vector_of_neighbors,2)

        d_Q_ju = first_formula_part - second_formula_part
        return d_Q_ju

    def get_Xe(self, u, v):
        """
        get the feature vector of edge (u, v).

        :param u:   given node u
        :param v:   given node v
        :return:    feature vector for edge (u, v)
        """
        Xe_uv = self.Xe.get((int(u), int(v)))
        if not Xe_uv:
            Xe_uv = np.array([0, 0, 0])
        return Xe_uv

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
                            - np.power((1.0/(1.0 + np.exp(-x))), 2)
            else:
                d_f_omega = 0
        return d_f_omega

    @staticmethod
    def get_pj_and_Qju_sum(p, Q, index_u):
        """
        sum all elements in page rank score vector (p) and
        transition probability matrix for entry u (Qu) and return the result

        :param Q:       transition matrix
        :param p:       page rank score
        :param index_u: index of node u
        :return:        sum of all elements p and Qu
        """
        elements_sum = 0
        for j_index in range(0, len(p)):
            elements_sum += p[j_index] + Q[j_index][index_u]
        return elements_sum


if __name__ == '__main__':
    pass
