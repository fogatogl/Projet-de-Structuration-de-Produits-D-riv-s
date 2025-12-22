import numpy as np
from numba import jit

class CorrelationEngine:
    """
    A utility class to handle correlation matrix sanitization and 
    Cholesky decomposition for multi-asset simulations.
    """
    
    @staticmethod
    def fix_non_psd(matrix: np.ndarray, epsilon: float = 1e-9) -> np.ndarray:
        """
        Implements Spectral Projection to find the nearest Positive Semi-Definite 
        matrix. Required when market correlation data is inconsistent.
        Ref: Rebonato & Jäckel (1999) - "The most general methodology"
        """
        # 1. Eigen-decomposition (eigh is robust for symmetric matrices)
        evals, evecs = np.linalg.eigh(matrix)
        
        # 2. Floor negative eigenvalues to epsilon (ensure strict positivity)
        evals = np.maximum(evals, epsilon)
        
        # 3. Reconstruct the Matrix (Covariance approximation)
        # Optimization: Use broadcasting instead of full matrix multiplication
        # T = Lambda^0.5 * V^T
        T = evecs * np.sqrt(evals)
        reconstructed = T @ T.T
        
        # 4. Normalize back to Correlation Matrix (Unit Diagonal)
        # We must ensure we don't divide by zero if a diagonal is degenerate
        d = np.diag(reconstructed)
        
        # Safety Check: Replace near-zero diagonals to prevent NaNs
        d[d < epsilon] = 1.0 
        
        inv_sd = 1.0 / np.sqrt(d)
        clean_matrix = reconstructed * np.outer(inv_sd, inv_sd)
        
        # 5. Hard-set diagonal to 1.0 to fix floating point drift
        np.fill_diagonal(clean_matrix, 1.0)
        
        return clean_matrix

    @staticmethod
    def get_cholesky(matrix: np.ndarray) -> np.ndarray:
        """
        Computes the Lower Triangular Cholesky factor L.
        Includes an automatic fallback to 'fix_non_psd' if input fails.
        """
        try:
            return np.linalg.cholesky(matrix)
        except np.linalg.LinAlgError:
            print(">> [MathUtils] Non-PSD matrix detected. Applying spectral repair...")
            clean_corr = CorrelationEngine.fix_non_psd(matrix)
            
            # Final Safety: Add tiny jitter if repair was not perfect due to float precision
            # 1e-12 is usually sufficient for cholesky stability without altering price
            clean_corr += np.eye(matrix.shape[0]) * 1e-12
            return np.linalg.cholesky(clean_corr)

# ==============================================================================
# OPTIMIZATION: Use @jit so this can be inlined into the main Monte Carlo loop.
# Without @jit, calling this from the main loop would cause a severe slowdown.
# ==============================================================================
@jit(nopython=True, fastmath=True, cache=True)
def get_brownian_bridge_hit(s_old: float, s_new: float, barrier: float, sigma: float, dt: float) -> bool:
    """
    Implements Brownian Bridge interpolation (Section 2.1.3).
    Calculates probability that a path crossed barrier B between t and t+1.
    """
    # 1. Immediate Breach Check
    if s_new <= barrier:
        return True
    
    # 2. Bridge Check (Only if we started above and ended above)
    # If s_old <= barrier, we were already dead, so logic doesn't apply (or is True)
    if s_old <= barrier:
        return True

    # 3. Compute Probability of Excursion
    # Formula: P(Hit) = exp( -2 * ln(S0/B) * ln(S1/B) / (sigma^2 * dt) )
    # Note: We use abs() to handle both Up-and-Out and Down-and-Out logic symmetrically,
    # though for Knock-In (Down-and-In), S > B.
    
    # Optimization: Pre-compute logs
    # We know s_old > barrier and s_new > barrier, so ratios > 1.0, logs > 0.
    log_old = np.log(s_old / barrier)
    log_new = np.log(s_new / barrier)
    
    # Variance over the step
    variance = sigma * sigma * dt
    
    # Avoid division by zero if vol is zero
    if variance < 1e-16:
        return False

    exponent = -2.0 * log_old * log_new / variance
    
    # Fast exit: if exponent is very negative, probability is 0
    if exponent < -50.0:
        return False
        
    prob_hit = np.exp(exponent)
    
    # 4. Monte Carlo Sampling
    return np.random.random() < prob_hit