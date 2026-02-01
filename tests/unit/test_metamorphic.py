import unittest
import numpy as np
from hypothesis import given, strategies as st

def mock_step(x, t, strict=False):
    if strict:
        return 1.0 if x >= t else 0.0
    return 1.0 / (1.0 + np.exp(-(x - t) * 10))

class TestMetamorphic(unittest.TestCase):
    @given(st.floats(min_value=-10, max_value=10), st.floats(min_value=-10, max_value=10))
    def test_step_monotonicity(self, t, x1):
        x2 = x1 + 0.01
        result1 = mock_step(x1, t)
        result2 = mock_step(x2, t)
        self.assertGreaterEqual(result2, result1 - 1e-6)  # Monotonic

    @given(st.floats(min_value=-5, max_value=5))
    def test_step_symmetry(self, delta):
        t = 0.0
        below = mock_step(t - abs(delta), t)
        above = mock_step(t + abs(delta), t)
        # Both should be equidistant from 0.5
        self.assertAlmostEqual(below + above, 1.0, delta=0.1)

    def test_step_threshold_boundary(self):
        t = 0.5
        for offset in [-0.001, 0.001]:
            result = mock_step(t + offset, t)
            if offset < 0:
                self.assertLess(result, 0.6)
            else:
                self.assertGreater(result, 0.4)

if __name__ == '__main__':
    unittest.main()
