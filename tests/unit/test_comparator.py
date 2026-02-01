import unittest
import numpy as np

def mock_step(x, t, strict=False):
    """Mock Step function for testing monotonicity."""
    if strict:
        return 1.0 if x >= t else 0.0
    # Approximate smooth step
    return 1.0 / (1.0 + np.exp(-(x - t) * 10))

class TestComparator(unittest.TestCase):
    def test_step_monotonicity(self):
        t = 0.5
        x_values = np.linspace(0, 1, 10)
        results = [mock_step(x, t) for x in x_values]
        
        # Verify monotonicity
        for i in range(len(results) - 1):
            self.assertGreaterEqual(results[i+1], results[i])

    def test_step_sharpness(self):
        t = 0.5
        # Close to threshold
        below = mock_step(0.49, t, strict=True)
        above = mock_step(0.51, t, strict=True)
        self.assertEqual(below, 0.0)
        self.assertEqual(above, 1.0)

if __name__ == '__main__':
    unittest.main()
