import networkx as nx
import numpy as np
import unittest

from polarization import (
    Agent, calculate_weight,
    raw_opinion_update_vec, opinion_update_vec, polarization
)


class TestBasicCalculations(unittest.TestCase):

    def setUp(self):

        self.a1_2 = Agent()
        self.a2_2 = Agent()
        self.a3_2 = Agent()
        self.a4_2 = Agent()

        self.a1_2.opinions = np.array([-1.0, 0.5])
        self.a2_2.opinions = np.array([-.5, .2])
        self.a3_2.opinions = np.array([-.7, .6])
        self.a4_2.opinions = np.array([-.7, .4])

        self.a1_3 = Agent()
        self.a2_3 = Agent()
        self.a3_3 = Agent()
        self.a4_3 = Agent()

        self.a1_3.opinions = np.array([-1.0, 0.5, -.7])
        self.a2_3.opinions = np.array([-.5, .2, .8])

    def test_calculate_weight(self):
        '''
        Weights correctly calcuated; F&M (2011) Equation 1
        '''

        num = 0.5 + 0.3
        expected = 1 - (num/2.0)
        assert calculate_weight(self.a1_2, self.a2_2) == expected

        num = 0.5 + 0.3 + 1.5
        expected = 1 - (num/3.0)
        assert calculate_weight(self.a1_3, self.a2_3) == expected

    def test_raw_state_update(self):
        '''
        Correct "raw" update to opinion, F&M (2011) Equation 2
        '''

        num_neighbors_fac = 1.0 / (2.0 * 3)
        w_12 = calculate_weight(self.a1_2, self.a2_2)
        w_13 = calculate_weight(self.a1_2, self.a3_2)
        w_14 = calculate_weight(self.a1_2, self.a4_2)

        self.a1_2.weights[self.a2_2] = w_12
        self.a1_2.weights[self.a3_2] = w_13
        self.a1_2.weights[self.a4_2] = w_14

        S = (w_12*np.array([.5, -.3])) + \
            (w_13*np.array([.3, .1])) + \
            (w_14*np.array([.3, -.1]))

        expected = num_neighbors_fac * S
        calculated = raw_opinion_update_vec(
            self.a1_2, [self.a2_2, self.a3_2, self.a4_2]
        )

        assert (calculated == expected).all(), 'calc: {}\nexpec: {}'.format(
            calculated, expected
        )

    def test_scaled_state_update(self):
        '''
        Correct calc of opinion update according to F&M (2011) Equation 2a
        '''
        neighbors = [self.a2_2, self.a3_2, self.a4_2]

        w_12 = calculate_weight(self.a1_2, self.a2_2)
        w_13 = calculate_weight(self.a1_2, self.a3_2)
        w_14 = calculate_weight(self.a1_2, self.a4_2)

        self.a1_2.weights[self.a2_2] = w_12
        self.a1_2.weights[self.a3_2] = w_13
        self.a1_2.weights[self.a4_2] = w_14

        raw_update_vec = raw_opinion_update_vec(self.a1_2, neighbors)

        expected_0 = \
            self.a1_2.opinions[0] + \
            ((1 + self.a1_2.opinions[0]) * raw_update_vec[0])

        expected_1 = \
            self.a1_2.opinions[1] + \
            ((1 - self.a1_2.opinions[1]) * raw_update_vec[1])

        calculated = opinion_update_vec(
            self.a1_2, neighbors
        )
        expected = np.array([expected_0, expected_1])

        assert (calculated == expected).all(), \
            'calculated: {}\nexpected: {}'.format(calculated, expected)

    def test_polarization(self):
        '''
        Polarization on graph equals expected manual calculation.
        '''

        d = np.zeros((4, 4))

        d[0, 1] = .4     ; d[0, 2] = .2     ; d[0, 3] = .2
        d[1, 0] = d[0, 1]; d[1, 2] = .3     ; d[1, 3] = .2
        d[2, 0] = d[0, 2]; d[2, 1] = d[1, 2]; d[2, 3] = .1
        d[3, 0] = d[0, 3]; d[3, 1] = d[1, 3]; d[3, 2] = d[2, 3]

        d_mean = d.sum() / (4 * 3)

        d_sub_mean = d - d_mean
        for i in range(4):
            d_sub_mean[i, i] = 0.0

        expected = np.sum((d_sub_mean)**2) / (4 * 3)

        g = nx.Graph()
        a1 = self.a1_2
        a2 = self.a2_2
        a3 = self.a3_2
        a4 = self.a4_2

        g.add_edges_from([
            (e1, e2, {'weight': calculate_weight(e1, e2)})
            for e1, e2 in [(a1, a2), (a1, a4), (a2, a3), (a3, a4)]
        ])

        calculated = polarization(g)

        np.testing.assert_approx_equal(
            expected, calculated,
            err_msg='calc: {}\nexpec: {}'.format(calculated, expected)
        )
