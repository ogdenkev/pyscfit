import unittest
import numpy as np
from pyscfit.pdf import W


class AsymptoticRTestCase(unittest.TestCase):
    def setUp(self):
        """Create Q matrix from Q-Matrix Cookbook p. 593"""
        self.Q = np.array(
            [
                [-3.05, 0.05, 0, 3, 0],
                [0.000666667, -0.500666667, 0.5, 0, 0],
                [0, 15, -19, 4, 0],
                [0.015, 0, 0.05, -2.065, 2],
                [0, 0, 0, 0.01, -0.01],
            ]
        )
        self.a_ind = np.array([0, 1], dtype="int32")
        self.f_ind = np.array([2, 3, 4], dtype="int32")
        self.td = 0.05
        self.s = 0
        self._W = W(self.s, self.Q, self.a_ind, self.f_ind, self.td)

    def test_W_shape(self):
        self.assertEqual(self._W.shape, (2, 2))

    def test_W(self):
        true_W = np.array(
            [
                [3.047862099413992, -0.052023367361842],
                [-0.000693644898158, 0.258571427794418],
            ]
        )
        self.assertTrue(np.allclose(self._W, true_W))

    def test_asymptotic_r_vals(self):
        pass

    def test_W_inverse_of_sI_minus_qFF_does_not_exist(self):
        pass

    def test_detW(self):
        pass

    def test_dWds(self):
        pass

    def test_chs_vectors(self):
        pass

    def test_reliability_function_R(self):
        pass

    def test_exact_pdf_with_missed_events(self):
        pass

