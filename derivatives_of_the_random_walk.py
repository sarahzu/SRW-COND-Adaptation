#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import gradient_ascent
from numpy import diff


class Algorithm3:

    interation_max = 100

    def __init__(self, omega):
        self.omega = omega

    def derivatives_of_the_random_walk(self, V, u, v, Q):
        """
        Compute the page rank scores in form of vector p and its
        derivatives in form of vector d_p.

        :param V:           seed set
        :param u:           node u
        :param v:           node v
        :return:    p:      page rank score vector
                    d_p:    derivative page rank score vector
        """
        # page rank
        p = np.empty([len(V)])
        # derivative page rank
        d_p = np.empty([len(V), len(self.omega)])

        if len(V) != 0:
            for u in range(0, len(V)):
                p[u] = 1 / len(V)
        else:
            p[u] = 0

        for u, k in zip(range(0, len(V)), range(0, len(self.omega))):
            d_p[u][k] = 0

        # Q = gradient_ascent.Algorithm2.get_transition_prob_matrix_Q(
        #     gradient_ascent.Algorithm2, u, v, self.omega)

        prev_p = np.empty([len(V)])
        prev_d_p = np.empty([len(V), len(self.omega)])
        first = True
        counter = 0
        # print("p & prev p:", p, prev_p)
        while not np.array_equal(p, prev_p) and not first:
            if counter >= self.interation_max:
                break
            counter += 1
            first = False
            prev_p = p
            for u in range(0, len(V)):
                if prev_p:
                    p[u] = self.get_pj_and_Qju_sum(prev_p, Q, u)
                else:
                    p[u] = 0
        for k in range(0, len(self.omega)):
            first = True
            counter = 0
            while not np.array_equal(d_p, prev_d_p) and not first:
                if counter >= self.interation_max:
                    break
                counter += 1
                prev_d_p = d_p
                for u in range(0, len(V)):
                    d_p[u] = self.get_derivative_pu(V[u], self.omega, p, Q)
        return p, d_p

    @staticmethod
    def get_derivative_pu(u, omega, p, Q):
        try:
            u_index = list(Q).index(u)
        except ValueError:
            return 0

        d_omega = omega
        sum_d_pj_times_Q_ju = 0
        sum_pj_times_d_Q_ju = 0

        for j in range(0, len(Q[u_index])):
            p_j = p[j]
            d_pj = diff(p_j) / d_omega
            Q_ju = Q[j][u]
            sum_d_pj_times_Q_ju += d_pj * Q_ju

        for j in range(0, len(Q[u_index])):
            p_j = p[j]
            Q_ju = Q[j][u]
            d_Q_ju = diff(Q_ju) / d_omega
            sum_pj_times_d_Q_ju += p_j * d_Q_ju

        return sum_d_pj_times_Q_ju + sum_pj_times_d_Q_ju

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
