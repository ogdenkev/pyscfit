import unittest
import numpy as np
import scipy.linalg
from pyscfit.pdf import W, detW, dWds, asymptotic_r_vals, chs_vectors, R


class WTestCase(unittest.TestCase):
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
        self.true_W = np.array(
            [
                [3.047862099413992, -0.052023367361842],
                [-0.000693644898158, 0.258571427794418],
            ]
        )

    def test_W_shape(self):
        self.assertEqual(self._W.shape, (2, 2))

    def test_W(self):
        self.assertTrue(np.allclose(self._W, self.true_W))

    def test_detW(self):
        true_detW = (
            self.true_W[0, 0] * self.true_W[1, 1]
            - self.true_W[0, 1] * self.true_W[1, 0]
        )
        self.assertTrue(
            np.allclose(
                true_detW, detW(self.s, self.Q, self.a_ind, self.f_ind, self.td)
            )
        )

    def test_dWds(self):
        tau = 0.2
        kA = self.a_ind.shape[0]
        kF = self.f_ind.shape[0]
        I_AA = np.eye(kA)
        I_FF = np.eye(kF)
        qAA = self.Q[np.ix_(self.a_ind, self.a_ind)]
        qAF = self.Q[np.ix_(self.a_ind, self.f_ind)]
        qFF = self.Q[np.ix_(self.f_ind, self.f_ind)]
        qFA = self.Q[np.ix_(self.f_ind, self.a_ind)]
        LFF = scipy.linalg.expm(qFF * tau)
        SFF = I_FF - LFF
        GFA_star = scipy.linalg.inv(-qFF) @ qFA
        W_prime = I_AA + qAF @ (SFF @ scipy.linalg.inv(-qFF) - tau * LFF) @ GFA_star
        self.assertTrue(
            np.allclose(dWds(0, self.Q, self.a_ind, self.f_ind, tau), W_prime)
        )


class KatzQMatrixTest(unittest.TestCase):
    def setUp(self):
        """delCastillo and Katz Mechanism

        R <-> RA <-> RA* <-> RA**
        """

        self.Q = np.array([[-2, 2, 0, 0], [1, -6, 5, 0], [0, 4, -7, 3], [0, 0, 2, -2]])
        self.iA = np.array([2, 3])
        self.iF = np.array([0, 1])
        self.Qaa = self.Q[np.ix_(self.iA, self.iA)]
        self.Qaf = self.Q[np.ix_(self.iA, self.iF)]
        self.Qff = self.Q[np.ix_(self.iF, self.iF)]
        self.Qfa = self.Q[np.ix_(self.iF, self.iA)]
        self.Qaa_inv = np.array([[-1 / 4, -3 / 8], [-1 / 4, -7 / 8]])
        self.Qff_inv = np.array([[-3 / 5, -1 / 5], [-1 / 10, -1 / 5]])
        self.tau = 0.2
        self.tcrit = 2.0
        self.s1 = -1.162532
        self.s2 = -3.921
        self.c1 = np.array([[0.922398484], [0.38623961]])
        self.r1 = np.array([[0.76666, 0.64205]])
        self.c2 = np.array([[-0.7212], [0.69272]])
        self.r2 = np.array([[-0.461745, 0.8870128]])
        self.Wprime_s1 = np.array([[1, 0], [0, 1.203931192]])
        self.Wprime_s2 = np.array([[1, 0], [0, 1.288005677]])
        self.phiA = np.array([[0.523142, 0.476858]])
        self.phiF = np.array([[0.227596, 0.772404]])
        self.phiB = np.array([[0.5231416, 0.4768584]])
        self.eb = np.array([[0.1728713], [0.07286452]])
        (
            self.s_,
            self.areaR_,
            self.r_,
            self.c_,
            self.Wprime_,
            self.mu_,
            self.a_,
        ) = asymptotic_r_vals(self.Q, self.iF, self.iA, self.tau)

    def test_asymptotic_r_vals(self):
        s = self.s_.ravel()
        idx_sorted = np.argsort(s)[::-1]
        with self.subTest(msg="s"):
            self.assertTrue(np.allclose(s[idx_sorted], np.array([self.s1, self.s2])))
        with self.subTest(msg="r"):
            r = self.r_[idx_sorted, :]
            scale_factor = r[:, 0] / r[:, 1]
            true_r = np.concatenate([self.r1, self.r2], axis=0)
            scale_factor_true = true_r[:, 0] / true_r[:, 1]
            self.assertTrue(np.allclose(scale_factor, scale_factor_true))
        with self.subTest(msg="c"):
            c = self.c_[:, idx_sorted]
            scale_factor = c[0, :] / c[1, :]
            true_c = np.concatenate([self.c1, self.c2], axis=1)
            scale_factor_true = true_c[0, :] / true_c[1, :]
            self.assertTrue(np.allclose(scale_factor, scale_factor_true, atol=3e-04))
        with self.subTest(msg="Wprime 1"):
            self.assertTrue(
                np.allclose(self.Wprime_[:, :, idx_sorted[0]], self.Wprime_s1)
            )
        with self.subTest(msg="Wprime 2"):
            self.assertTrue(
                np.allclose(self.Wprime_[:, :, idx_sorted[1]], self.Wprime_s2)
            )
        with self.subTest(msg="areaR"):
            R1 = self.areaR_[:, :, idx_sorted[0]]
            self.assertTrue(
                np.allclose(
                    R1, (self.c1 @ self.r1) / (self.r1 @ self.Wprime_s1 @ self.c1)
                )
            )
            R2 = self.areaR_[:, :, idx_sorted[1]]
            self.assertTrue(
                np.allclose(
                    R2, (self.c2 @ self.r2) / (self.r2 @ self.Wprime_s2 @ self.c2)
                )
            )

    def test_asymptotic_areas(self):
        uA = np.ones((self.iA.size, 1))
        R1 = (self.c1 @ self.r1) / (self.r1 @ self.Wprime_s1 @ self.c1)
        mu1 = -1 / self.s1
        a1 = (
            mu1
            * self.phiF
            @ R1
            @ self.Qfa
            @ scipy.linalg.expm(self.Qaa * self.tau)
            @ uA
        )
        R2 = (self.c2 @ self.r2) / (self.r2 @ self.Wprime_s2 @ self.c2)
        mu2 = -1 / self.s2
        a2 = (
            mu2
            * self.phiF
            @ R2
            @ self.Qfa
            @ scipy.linalg.expm(self.Qaa * self.tau)
            @ uA
        )
        s = self.s_.ravel()
        idx_sorted = np.argsort(s)[::-1]
        exponential_area = self.a_.ravel()[idx_sorted]
        self.assertTrue(np.allclose(a1, exponential_area[0]))
        self.assertTrue(np.allclose(a2, exponential_area[1]))

    def test_W_inverse_of_sI_minus_qFF_does_not_exist(self):
        pass

    def test_chs_vectors(self):
        phib, ef = chs_vectors(
            self.Q, self.iA, self.iF, self.areaR_, self.mu_, self.tau, self.tcrit
        )
        self.assertTrue(np.allclose(phib, self.phiB))
        self.assertTrue(np.allclose(ef, self.eb))

    def test_reliability_function_R(self):
        # def R(t, C, lambdas, tau, s, areaR, mMax=2):
        t = 0.5
        result = R(t, None, None, self.tau, self.s_, self.areaR_)
        Rt_1 = (
            np.exp(self.s1 * 0.5)
            * self.c1
            @ self.r1
            / (self.r1 @ self.Wprime_s1 @ self.c1)
        )
        Rt_2 = (
            np.exp(self.s2 * 0.5)
            * self.c2
            @ self.r2
            / (self.r2 @ self.Wprime_s2 @ self.c2)
        )
        Rt = Rt_1 + Rt_2
        self.assertTrue(np.allclose(result, Rt))

    def test_reliability_function_with_time_less_than_zero(self):
        t = -0.2
        result = R(t, None, None, self.tau, self.s_, self.areaR_)
        self.assertTrue(np.allclose(result, 0))

    def test_exact_pdf_with_missed_events(self):
        pass
