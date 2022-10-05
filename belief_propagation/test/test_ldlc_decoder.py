import unittest
import numpy as np
from numpy.linalg import inv
from belief_propagation.ldlc_bp_decoding import ldlc_decoder


class TestLdlcDecodingCase(unittest.TestCase):
    def test_ldlc_decoder(self):
        H = [[1, 0, 0, 0.5, 0, 0, -0.25, 0],
             [0.25, 1, 0, 0, 0, 0, 0, -0.5],
             [0, 0.5, 1, 0, 0, 0, 0, 0.25],
             [0, 0, 0.25, 1, 0, 0, -0.5, 0],
             [0.5, 0, 0, 0, 1, -0.25, 0, 0],
             [0, 0, 0.5, 0.25, 0, 1, 0, 0],
             [0, -0.25, 0, 0, 0.5, 0, 1, 0],
             [0, 0, 0, 0, -0.25, 0.5, 0, 1]
             ]
        b = [1, 1, 3, 4, 2, 6, 7, 4]
        lattice_x = np.round(np.matmul(inv(np.array(H)), np.array(b)))
        sigma = 0.05
        lmax = 10
        np.random.seed(42)
        y = list(lattice_x + np.random.normal(0, sigma, 8))
        x_estimate = ldlc_decoder(y, sigma, H, lmax)
        self.assertListEqual(list(lattice_x), list(x_estimate))


if __name__ == '__main__':
    unittest.main()
