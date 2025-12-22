import unittest
import numpy as np

# Adjust import based on your folder structure. 
# Assuming test_math.py is in 'tests/' and source is in 'src/'
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.math_utils import CorrelationEngine, get_brownian_bridge_hit

class TestCorrelationEngine(unittest.TestCase):
    """
    Tests for Matrix Sanitization (Spectral Projection) and Cholesky Decomposition.
    """

    def setUp(self):
        # A valid, positive semi-definite matrix
        self.valid_matrix = np.array([
            [1.0, 0.5, 0.2],
            [0.5, 1.0, 0.3],
            [0.2, 0.3, 1.0]
        ])

        # A "Broken" Matrix (Violates Triangle Inequality)
        # A is highly corr with B (0.9), B with C (0.9), but A is anti-corr with C (-0.9)
        # This is mathematically impossible in Euclidean space.
        self.broken_matrix = np.array([
            [ 1.0,  0.9, -0.9],
            [ 0.9,  1.0,  0.9],
            [-0.9,  0.9,  1.0] 
        ])

    def test_valid_matrix_untouched(self):
        """Ensure valid matrices are not distorted by the repair logic."""
        repaired = CorrelationEngine.fix_non_psd(self.valid_matrix)
        
        # The repair might introduce tiny floating point diffs, but should be close
        np.testing.assert_array_almost_equal(
            self.valid_matrix, 
            repaired, 
            decimal=8,
            err_msg="Valid matrix was unnecessarily altered."
        )

    def test_broken_matrix_repair(self):
        """Ensure non-PSD matrix is projected to the nearest PSD matrix."""
        # 1. Verify input is indeed broken (has negative eigenvalues)
        evals_orig, _ = np.linalg.eigh(self.broken_matrix)
        self.assertTrue(np.any(evals_orig < -1e-9), "Input test matrix should be non-PSD")

        # 2. Repair
        repaired = CorrelationEngine.fix_non_psd(self.broken_matrix)

        # 3. Check Eigenvalues of result
        evals_new, _ = np.linalg.eigh(repaired)
        self.assertTrue(np.all(evals_new > -1e-15), "Repaired matrix must be PSD")

        # 4. Check Diagonal is 1.0 (Correlation property)
        diag = np.diag(repaired)
        np.testing.assert_array_almost_equal(diag, np.ones(3), decimal=8)
        
        # 5. Check Symmetry
        np.testing.assert_array_almost_equal(repaired, repaired.T, decimal=8)

    def test_cholesky_decomposition(self):
        """Ensure L * L.T reconstructs the original matrix."""
        L = CorrelationEngine.get_cholesky(self.valid_matrix)
        
        # Reconstruction
        reconstructed = L @ L.T
        np.testing.assert_array_almost_equal(
            self.valid_matrix, 
            reconstructed, 
            decimal=8,
            err_msg="Cholesky factor failed to reconstruct original matrix."
        )

    def test_cholesky_fallback(self):
        """Ensure get_cholesky automatically fixes a broken matrix without crashing."""
        try:
            L = CorrelationEngine.get_cholesky(self.broken_matrix)
        except np.linalg.LinAlgError:
            self.fail("get_cholesky raised LinAlgError instead of handling it internally.")
            
        # Check that L is lower triangular
        self.assertTrue(np.allclose(L, np.tril(L)))


class TestBrownianBridge(unittest.TestCase):
    """
    Tests for the Numba-compiled Brownian Bridge logic.
    """

    def test_immediate_breach(self):
        """If the path ends below barrier, it MUST return True."""
        s_old = 105.0
        s_new = 50.0 # Way below barrier
        barrier = 60.0
        sigma = 0.2
        dt = 1/252
        
        # Deterministic check
        hit = get_brownian_bridge_hit(s_old, s_new, barrier, sigma, dt)
        self.assertTrue(hit, "Failed to detect immediate barrier breach.")

    def test_safe_path(self):
        """If path is far above barrier, probability of hit is near 0."""
        s_old = 150.0
        s_new = 151.0
        barrier = 60.0 # Far away
        sigma = 0.2
        dt = 1/252
        
        # Run multiple times to ensure no random hits (prob is astronomically low)
        for _ in range(100):
            hit = get_brownian_bridge_hit(s_old, s_new, barrier, sigma, dt)
            self.assertFalse(hit, "Detected phantom barrier hit on safe path.")

    def test_near_miss_probability(self):
        """
        Statistical Test: Path starts and ends just above barrier. 
        Should hit roughly 50% of the time if volatility is high enough.
        """
        # Scenario: Start=101, End=101, Barrier=100. High Vol.
        s_old = 100.1
        s_new = 100.1
        barrier = 100.0
        sigma = 0.5 # High vol makes excursion likely
        dt = 0.1    # Long time step makes excursion likely
        
        hits = 0
        n_trials = 10000
        
        for _ in range(n_trials):
            if get_brownian_bridge_hit(s_old, s_new, barrier, sigma, dt):
                hits += 1
        
        hit_rate = hits / n_trials
        
        # Theoretical Probability check (Manual calc)
        # log(S/B) approx 0.001
        # exponent = -2 * 0.001 * 0.001 / (0.25 * 0.1) approx 0.0
        # exp(0) = 1.0 -> Wait, this setup implies 100% hit rate theoretically?
        # Let's use a setup with ~50% prob
        
        # Log(S/B) = x. 
        # want exp(-2x^2/var) = 0.5 => -2x^2/var = ln(0.5) = -0.693
        # 2x^2/var = 0.693 => x^2/var = 0.346
        # Let var = 1.0. x = sqrt(0.346) = 0.588
        # S = B * exp(0.588)
        
        # Refined Test Case for ~50% prob
        barrier = 100.0
        target_prob = 0.5
        variance = 1.0 # sigma=1, dt=1
        
        # Calculate x (log moneyness) that gives 50% hit prob
        # -2 * x^2 / 1 = ln(0.5) -> x = sqrt(-ln(0.5)/2) = 0.5887
        
        s_val = barrier * np.exp(0.5887) # approx 180.16
        
        hits = 0
        for _ in range(n_trials):
            if get_brownian_bridge_hit(s_val, s_val, barrier, 1.0, 1.0):
                hits += 1
                
        measured_prob = hits / n_trials
        
        # Allow wide tolerance because it's Monte Carlo (e.g. +/- 5%)
        self.assertTrue(0.45 < measured_prob < 0.55, 
                        f"Statistical Brownian Bridge test failed. Expected ~0.5, got {measured_prob}")

if __name__ == '__main__':
    unittest.main()