#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import gradient_ascent


class Algorithm3:

    def __init__(self, omega):
        self.omega = omega

    def derivatives_of_the_random_walk(self, V, u, v, E):
        """
        Compute the page rank scores in form of vector p and its
        derivatives in form of vector d_p.

        :param V:           seed set
        :param u:           node u
        :param v:           node v
        :param E:           set of edges
        :return:    p:      page rank score vector
                    d_p:    derivative page rank score vector
        """
        # page rank
        p = np.array([])[:, np.newaxis]
        # derivative page rank
        d_p = np.array([])[:, np.newaxis]

        for u in range(0, len(V)):
            p[u] = 1 / len(V)

        for u, k in zip(range(0, len(V)), range(0, len(self.omega))):
            d_p[u][k] = 0

        Q = gradient_ascent.get_transition_prob_matrix_Q(
            u, v, self.omega, E)

        prev_p = np.array([])[:, np.newaxis]
        prev_d_p = np.array([])[:, np.newaxis]
        threshold = 0.001

        while p != prev_p - threshold:
            for u in range(0, len(V)):
                if prev_p:
                    p[u] = self.get_pj_and_Qju_sum(prev_p, Q, u)

        for k in range(0, len(self.omega)):
            while d_p != prev_d_p - threshold:
                for u in range(0, len(V)):
                    d_p[u] = gradient_ascent.Algorithm2.get_derivative_pu(
                        V[u], self.omega)

        return p, d_p

    @staticmethod
    def get_pj_and_Qju_sum(p, Q, u):
        """
        sum all elements in p and and Qu and return result

        :param u:   index node u
        :param Q:   transition matrix
        :param p:   page rank score
        :return:    sum of all elements p and Qu
        """
        elements_sum = 0
        for j in range(0, len(p)):
            elements_sum += p[j] + Q[j][u]
        return elements_sum


if __name__ == '__main__':
    pass
