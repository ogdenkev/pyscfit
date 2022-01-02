import unittest
import numpy as np
from pyscfit.qmatrix import qmatvals, dvals, cvals


class QMatrixTestCase(unittest.TestCase):
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
        self.true_taus = np.array(
            [np.inf, 9.82145526, 0.49453067, 0.3232556, 0.05152461]
        )
        self.true_lambdas = np.array(
            [0.0, 0.10181791, 2.02211927, 3.09352724, 19.40820226]
        )
        self.true_A = np.array(
            [
                [
                    [
                        2.48271412e-05,
                        1.66996631e-05,
                        4.05101032e-02,
                        9.59448224e-01,
                        1.45533097e-07,
                    ],
                    [
                        1.86203557e-03,
                        3.49677662e-02,
                        -6.35589255e-02,
                        2.72025557e-02,
                        -4.73431990e-04,
                    ],
                    [
                        6.20678524e-05,
                        9.29734350e-04,
                        6.31175916e-03,
                        -7.90032303e-03,
                        5.96761662e-04,
                    ],
                    [
                        4.96542821e-03,
                        1.72811947e-03,
                        2.77879514e00,
                        -2.78535102e00,
                        -1.37669224e-04,
                    ],
                    [
                        9.93085641e-01,
                        -3.76423196e-02,
                        -2.76205808e00,
                        1.80660056e00,
                        1.41940188e-05,
                    ],
                ],
                [
                    [
                        2.48271412e-05,
                        4.66236993e-04,
                        -8.47452346e-04,
                        3.62700639e-04,
                        -6.31242696e-06,
                    ],
                    [
                        1.86203557e-03,
                        9.76263177e-01,
                        1.32962289e-03,
                        1.02833942e-05,
                        2.05348812e-02,
                    ],
                    [
                        6.20678524e-05,
                        2.59572032e-02,
                        -1.32039039e-04,
                        -2.98656262e-06,
                        -2.58842454e-02,
                    ],
                    [
                        4.96542821e-03,
                        4.82472742e-02,
                        -5.81310902e-02,
                        -1.05294748e-03,
                        5.97133528e-03,
                    ],
                    [
                        9.93085641e-01,
                        -1.05093389e00,
                        5.77809587e-02,
                        6.82950013e-04,
                        -6.15658627e-04,
                    ],
                ],
                [
                    [
                        2.48271412e-05,
                        3.71893829e-04,
                        2.52470366e-03,
                        -3.16012931e-03,
                        2.38704681e-04,
                    ],
                    [
                        1.86203557e-03,
                        7.78716095e-01,
                        -3.96117114e-03,
                        -8.95969073e-05,
                        -7.76527363e-01,
                    ],
                    [
                        6.20678524e-05,
                        2.07047570e-02,
                        3.93366597e-04,
                        2.60212502e-05,
                        9.78813787e-01,
                    ],
                    [
                        4.96542821e-03,
                        3.84844270e-02,
                        1.73182335e-01,
                        9.17409524e-03,
                        -2.25806285e-01,
                    ],
                    [
                        9.93085641e-01,
                        -8.38277173e-01,
                        -1.72139234e-01,
                        -5.95039027e-03,
                        2.32811559e-02,
                    ],
                ],
                [
                    [
                        2.48271412e-05,
                        8.64059937e-06,
                        1.38939757e-02,
                        -1.39267551e-02,
                        -6.88346163e-07,
                    ],
                    [
                        1.86203557e-03,
                        1.80927278e-02,
                        -2.17991586e-02,
                        -3.94855420e-04,
                        2.23925073e-03,
                    ],
                    [
                        6.20678524e-05,
                        4.81055335e-04,
                        2.16477919e-03,
                        1.14676187e-04,
                        -2.82257856e-03,
                    ],
                    [
                        4.96542821e-03,
                        8.94149054e-04,
                        9.53058844e-01,
                        4.04304271e-02,
                        6.51151413e-04,
                    ],
                    [
                        9.93085641e-01,
                        -1.94765727e-02,
                        -9.47318441e-01,
                        -2.62234927e-02,
                        -6.71352329e-05,
                    ],
                ],
                [
                    [
                        2.48271412e-05,
                        -9.41058212e-07,
                        -6.90514519e-05,
                        4.51650140e-05,
                        3.54850493e-10,
                    ],
                    [
                        1.86203557e-03,
                        -1.97050104e-03,
                        1.08339297e-04,
                        1.28053164e-06,
                        -1.15435993e-06,
                    ],
                    [
                        6.20678524e-05,
                        -5.23923231e-05,
                        -1.07587021e-05,
                        -3.71899380e-07,
                        1.45507224e-06,
                    ],
                    [
                        4.96542821e-03,
                        -9.73828637e-05,
                        -4.73659220e-03,
                        -1.31117464e-04,
                        -3.35676164e-07,
                    ],
                    [
                        9.93085641e-01,
                        2.12121728e-03,
                        4.70806306e-03,
                        8.50438175e-05,
                        3.46089973e-08,
                    ],
                ],
            ]
        )

        self.taus, self.lambdas, self.A = qmatvals(self.Q)

        self.a_ind = np.array([0, 1], dtype="int32")
        self.f_ind = np.array([2, 3, 4], dtype="int32")
        self.td = 0.05
        self.dvals = dvals(self.Q, self.a_ind, self.f_ind, self.td, self.A)
        self.mMax = 2
        self.C = cvals(self.Q, self.a_ind, self.f_ind, self.td, self.lambdas, self.A, self.mMax)

    def test_qmatvals_imaginary_eigenvalues_of_q_raises_error(self):
        # TODO: Create a Q matrix with imaginary eigenvalues ?!?
        pass

    def test_qmatvals_taus(self):
        self.assertTrue(np.allclose(self.taus, self.true_taus))

    def test_qmatvals_lambdas(self):
        self.assertTrue(np.allclose(self.lambdas, self.true_lambdas))

    def test_qmatvals_spectral_matrices(self):
        self.assertTrue(np.allclose(self.A, self.true_A))

    def test_dvals_shape(self):
        self.assertEqual(self.dvals.shape, (2, 2, 5))

    def test_dvals(self):
        true_D3 = np.array(
            [
                [0.037593031410235, 0.100012824437647],
                [-0.000786428567145, -0.002092221331137],
            ]
        )
        self.assertTrue(np.allclose(self.dvals[:, :, 2], true_D3))

    def test_cvals_shape(self):
        self.assertEqual(self.C.shape, (2, 2, 5, 3, 3))

    def test_cvals(self):
        true_C000 = np.array(
            [[2.48271412e-05, 1.86203557e-03], [2.48271412e-05, 1.86203557e-03]]
        )
        self.assertTrue(np.allclose(self.C[:, :, 0, 0, 0], true_C000))

    def test_eG(self):
        # TODO: Find a known correct output of this from a given Q matrix
        pass
    
    def test_phi(self):
        # TODO: Get known values of phi from a known input
        pass

    def test_equilibrium_occupancy(self):
        pass

