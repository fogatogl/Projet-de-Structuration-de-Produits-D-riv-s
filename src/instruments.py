import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass(frozen=True)
class MARC_Instrument:
    """
    Data structure defining a Multi-Asset Autocallable Reverse Convertible.
    Using 'frozen=True' ensures the product definition cannot be accidentally 
    altered mid-simulation.
    """
    tickers: List[str]
    initial_spots: np.ndarray
    
    # Barrier Levels (expressed as decimals, e.g., 0.60 for 60%)
    barrier_autocall: float
    barrier_knock_in: float
    barrier_coupon: float
    
    # Financial Terms
    coupon_rate: float            # Annualized coupon (e.g., 0.08)
    memory_feature: bool = True   # Phoenix structure toggle
    principal: float = 1.0        # Normalized to 1.0 (100%)
    
    # Temporal Parameters
    observation_dates: np.ndarray = field(default_factory=lambda: np.array([]))
    maturity: float = 0.0
    
    def __post_init__(self):
        """Validation logic to ensure the instrument is mathematically sound."""
        if len(self.tickers) != len(self.initial_spots):
            raise ValueError("Number of tickers must match number of initial spots.")
        
        if self.barrier_knock_in >= 1.0:
            raise ValueError("Knock-In barrier is typically below 100% (initial spot).")
            
        if self.maturity <= 0:
            raise ValueError("Maturity must be a positive value.")

@dataclass
class MarketEnvironment:
    """
    Container for the stochastic parameters required by the pricing engine.
    """
    risk_free_rate: float
    dividend_yields: np.ndarray  # Array of q for each asset
    correlation_matrix: np.ndarray
    
    # Dictionary to hold the Local Vol Grids generated in Step 2
    # Key: Ticker String -> Value: Tuple(T_grid, S_grid, Vol_surface)
    vol_surfaces: dict = field(default_factory=dict)

    def get_rates_array(self) -> np.ndarray:
        """Returns a flat array [r, q1, q2, ... qN] for the Numba kernel."""
        return np.concatenate(([self.risk_free_rate], self.dividend_yields))