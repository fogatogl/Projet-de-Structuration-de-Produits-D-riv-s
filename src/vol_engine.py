import numpy as np
from scipy.interpolate import RectBivariateSpline
from dataclasses import dataclass
from typing import Tuple

@dataclass
class MarketData:
    """
    Container for raw market implied volatility surface.
    Ensure arrays are sorted by Expiry and Strike.
    """
    strikes: np.ndarray       # 1D Array (K)
    expiries: np.ndarray      # 1D Array (T)
    implied_vols: np.ndarray  # 2D Array (N_Expiries x N_Strikes)
    spot: float               # S0
    risk_free_rate: float     # r (continuous)
    div_yield: float          # q (continuous)

    def __post_init__(self):
        # Validation for spline requirements
        if not np.all(np.diff(self.strikes) > 0):
            raise ValueError("Strikes must be strictly increasing.")
        if not np.all(np.diff(self.expiries) > 0):
            raise ValueError("Expiries must be strictly increasing.")

class LocalVolEngine:
    """
    Constructs a Dupire Local Volatility Surface from Market Data.
    Uses Bicubic Splines on Total Variance coordinates for numerical stability.
    """

    def __init__(self, market_data: MarketData):
        self.md = market_data
        self.spline = None
        self._fit_variance_surface()

    def _fit_variance_surface(self):
        """
        Fits a smooth spline to the Total Variance surface w(y, T).
        Transformation:
            Forward F(T) = S0 * exp((r-q)T)
            Log-Moneyness y = ln(K / F(T))
            Total Variance w = sigma_imp^2 * T
        """
        # 1. Calculate Forwards for all expiries
        # Shape: (N_expiries,)
        forwards = self.md.spot * np.exp(
            (self.md.risk_free_rate - self.md.div_yield) * self.md.expiries
        )

        # 2. Construct Grid for Spline
        # We need a rectangular grid. Since y depends on T, strictly speaking, 
        # the grid is skewed. 
        # APPROACH: We map the input data to a normalized grid.
        # However, RectBivariateSpline requires orthogonal x, y inputs.
        # Standard Industry Trick: We fit w against (T, Log-Strike-Ratio) 
        # and handle the drift adjustment inside the Dupire formula derivatives.
        
        # Log-Strike-Ratio k = ln(K / S0)
        self.log_strikes = np.log(self.md.strikes / self.md.spot)
        
        # Total Variance w
        # Shape must be (N_expiries, N_strikes)
        self.w_surface = (self.md.implied_vols ** 2) * self.md.expiries[:, None]

        # 3. Fit the Spline
        # kx=3, ky=3 ensures C2 continuity (essential for second derivatives)
        self.spline = RectBivariateSpline(
            self.md.expiries,
            self.log_strikes,
            self.w_surface,
            kx=3, ky=3
        )

    def _get_dupire_local_vol(self, T: float, S: float) -> float:
        """
        Calculates the instantaneous local volatility sigma_loc(T, S).
        Formula: Gatheral's representation of Dupire in terms of Total Variance w.
        """
        # 1. Handle T=0 edge case (return closest vol)
        if T <= 1e-5:
            return self.md.implied_vols[0, len(self.md.strikes)//2]

        # 2. Current Log-Moneyness
        # k = ln(S / S0) (Note: S is the spot argument, not K)
        # We need to query the spline at the level corresponding to strike K = S
        k = np.log(S / self.md.spot)
        
        # 3. Extract Derivatives from Spline
        # w(T, k)
        w = self.spline(T, k)[0, 0]
        
        # Derivatives
        dw_dT = self.spline(T, k, dx=1, dy=0)[0, 0] # Time derivative
        dw_dk = self.spline(T, k, dx=0, dy=1)[0, 0] # Strike derivative (Slope)
        d2w_dk2 = self.spline(T, k, dx=0, dy=2)[0, 0] # Convexity (Butterfly)

        # 4. Apply Arbitrage Constraints (Sanity Checks)
        # Calendar Arbitrage: Variance must strictly increase with time
        if dw_dT < 1e-6:
            dw_dT = 1e-6
            
        # 5. Dupire Formula (Variance Form)
        # y_eff is the effective log-moneyness coordinate. 
        # In our spline basis k = ln(K/S0), the drift (r-q) is implicit.
        # We use the standard transformation for derivatives:
        
        # y = k
        # Denominator (Density function):
        # 1 - (y/w)*dw_dy + 0.25*(-0.25 - 1/w + y^2/w^2)*(dw_dy)^2 + 0.5*d2w_dy2
        
        # Term by term for clarity and debuggability:
        term1 = 1.0 - (k / w) * dw_dk
        term2 = 0.25 * (-0.25 - (1.0/w) + (k**2)/(w**2)) * (dw_dk**2)
        term3 = 0.5 * d2w_dk2
        
        denominator = term1 + term2 + term3
        
        # Butterfly Arbitrage Check (Density must be positive)
        if denominator <= 1e-6:
            denominator = 1e-6

        # Numerator adjustments for Drift (r-q)
        # If we interpolated on Forward Moneyness, this term vanishes.
        # Since we interpolated on Spot Moneyness k, we must adjust dw/dT.
        # Adjust: dw/dT_actual = dw/dT_spline + (r-q) * dw/dk
        mu = self.md.risk_free_rate - self.md.div_yield
        numerator = dw_dT + (mu * dw_dk)
        
        # Floor numerator
        if numerator < 0:
            numerator = 1e-6

        local_var = numerator / denominator
        
        return np.sqrt(local_var)

    def generate_grid(self, max_T: float, n_time: int = 100, n_spot: int = 200) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates the dense lookup table for the Numba engine.
        
        Args:
            max_T: Max maturity to generate (e.g., 3.0 years)
            n_time: Number of time steps (M)
            n_spot: Number of spot steps (N)
            
        Returns:
            Tuple (Time_Grid, Spot_Grid, Vol_Surface_Matrix)
        """
        # 1. Create Grids
        # Time: Small epsilon to max_T
        t_grid = np.linspace(1e-4, max_T, n_time)
        
        # Spot: From 1% of S0 to 250% of S0 (covers most crash/rally scenarios)
        # We use a log-space grid for Spot to give more resolution near the barrier (low spot)
        # Logspace from 1% to 250%
        # min_S = 0.01 * S0, max_S = 2.5 * S0
        s_min = 0.01 * self.md.spot
        s_max = 2.50 * self.md.spot
        s_grid = np.geomspace(s_min, s_max, n_spot)
        
        vol_surface = np.zeros((n_time, n_spot))
        
        # 2. Populate Grid (Vectorize where possible, but Dupire is point-wise)
        # Since the formula is complex, a simple double loop is sufficiently fast 
        # (this runs once at init, taking ~50-100ms)
        for i, t in enumerate(t_grid):
            for j, s in enumerate(s_grid):
                vol_surface[i, j] = self._get_dupire_local_vol(t, s)
                
        return t_grid, s_grid, vol_surface