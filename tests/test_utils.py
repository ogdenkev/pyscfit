import unittest
import numpy as np
from pyscfit.utils import _match, _match_hash


class UtilsTestCase(unittest.TestCase):
    def setUp(self):
        self.x = np.array([19, 21, 11, 18, 46], dtype=np.int_)

    def test__match_single_query_value(self):
        y = 11
        self.assertEqual(_match(self.x, y), 2)

    def test__match_array_query(self):
        y = np.array([11, 18, 19], dtype=np.int_)
        np.testing.assert_array_equal(np.array([2, 3, 0]), _match(self.x, y))

    def test__match_hash_single_value_raises_TypeError(self):
        y = 11
        self.assertRaises(TypeError, _match_hash, self.x, y)

    def test__match_hash_array_query(self):
        y = [18, 11]
        self.assertEqual([3, 2], _match_hash(self.x, y))

    def test__match_hash_array_query_missing_in_reference(self):
        y = np.array([11, 12, 19, 25, 22])
        self.assertEqual([2, None, 0, None, None], _match_hash(self.x, y))
