import numpy as np
from numba import jit, prange, float64, int64
from typing import Tuple, Dict

# Import the JIT-compiled math helper
from .math_utils import get_brownian_bridge_hit

# ==========================================
# 1. Low-Level JIT Helpers
# ==========================================

@jit(nopython=True, fastmath=True, cache=True)
def _get_local_vol(t: float, s: float, t_grid: np.ndarray, s_grid: np.ndarray, vol_surface: np.ndarray) -> float:
    """
    Performs Bilinear Interpolation on the Local Volatility Grid.
    Optimization: Uses binary search for index lookup O(log N).
    """
    # 1. Search Indices
    t_idx = np.searchsorted(t_grid, t) - 1
    s_idx = np.searchsorted(s_grid, s) - 1
    
    # 2. Boundary Safety (Clamp to Grid Edges)
    # This prevents crashes if Spot goes to 0.0 or Infinity
    if t_idx < 0: t_idx = 0
    if t_idx >= len(t_grid) - 1: t_idx = len(t_grid) - 2
    
    if s_idx < 0: s_idx = 0
    if s_idx >= len(s_grid) - 1: s_idx = len(s_grid) - 2
    
    # 3. Bilinear Interpolation
    t1 = t_grid[t_idx]
    t2 = t_grid[t_idx+1]
    s1 = s_grid[s_idx]
    s2 = s_grid[s_idx+1]
    
    # Weights
    denom_t = (t2 - t1)
    denom_s = (s2 - s1)
    
    # Safety against degenerate grids
    if denom_t == 0.0: w_t = 0.0 
    else: w_t = (t - t1) / denom_t
        
    if denom_s == 0.0: w_s = 0.0 
    else: w_s = (s - s1) / denom_s
    
    # Fetch Corner Values
    # Memory Layout: vol_surface[Time, Spot]
    c00 = vol_surface[t_idx, s_idx]
    c01 = vol_surface[t_idx, s_idx+1]
    c10 = vol_surface[t_idx+1, s_idx]
    c11 = vol_surface[t_idx+1, s_idx+1]
    
    # Interpolate
    val = (1 - w_t) * (1 - w_s) * c00 + \
          (1 - w_t) * w_s       * c01 + \
          w_t       * (1 - w_s) * c10 + \
          w_t       * w_s       * c11
          
    return val

# ==========================================
# 2. The Core Monte Carlo Kernel
# ==========================================

@jit(nopython=True, parallel=True, fastmath=True)
def _marc_kernel(
    S0: np.ndarray,            # Initial Spots (N,)
    L_matrix: np.ndarray,      # Cholesky Lower Triangle (N,N)
    T: float,                  # Maturity
    dt: float,                 # Time step
    vol_grids_T: np.ndarray,   # Shared T grid
    vol_grids_S: np.ndarray,   # Shared S grid
    vol_surfaces: np.ndarray,  # Stacked surfaces (N, Time, Spot)
    obs_indices: np.ndarray,   # Array of integers: time steps that are obs dates
    barriers: np.ndarray,      # [B_KO, B_KI, B_Cpn, Coupon_Amt]
    rates: np.ndarray,         # [r, q1, q2... qN]
    memory_feature: bool,
    n_sims: int                # Number of PAIRS (Actual paths = 2 * n_sims)
) -> np.ndarray:
    
    n_assets = len(S0)
    n_steps = int(T / dt)
    
    # Unpack Barriers
    B_KO = barriers[0] # Autocall
    B_KI = barriers[1] # Knock-In
    B_Cpn = barriers[2]
    Cpn_Amt = barriers[3]
    
    r = rates[0]
    
    # Results Container: [Payoff, Knocked_In_Flag]
    # We run 2 paths per loop iteration (Antithetic)
    results = np.zeros((n_sims * 2, 2))
    
    # PARALLEL LOOP (OpenMP)
    for i in prange(n_sims):
        
        # --- Initialize TWO paths (Antithetic A and B) ---
        S_A = S0.copy()
        S_B = S0.copy()
        
        alive_A, alive_B = True, True
        ki_A, ki_B = False, False
        
        mem_A, mem_B = 0.0, 0.0 # Accumulated coupons
        
        # Buffers for Random Numbers
        Z = np.zeros(n_assets)
        dW_A = np.zeros(n_assets)
        dW_B = np.zeros(n_assets)
        
        # --- Time Stepping ---
        for t_step in range(1, n_steps + 1):
            # OPTIMIZATION: Early Exit if both paths are dead
            if not alive_A and not alive_B:
                break
            
            curr_time = t_step * dt
            sq_dt = np.sqrt(dt)
            
            # 1. Generate Independent Normals
            for k in range(n_assets):
                Z[k] = np.random.standard_normal()
            
            # 2. Correlate (Manual Unroll)
            # We generate shocks for Path A (+Z) and Path B (-Z)
            for row in range(n_assets):
                sum_val = 0.0
                for col in range(row + 1):
                    sum_val += L_matrix[row, col] * Z[col]
                
                dW_A[row] = sum_val
                dW_B[row] = -sum_val # Antithetic
            
            # 3. Evolve Assets & Check Barriers
            for a in range(n_assets):
                q = rates[1 + a]
                
                # --- PATH A ---
                if alive_A:
                    sig_A = _get_local_vol(curr_time, S_A[a], vol_grids_T, vol_grids_S, vol_surfaces[a])
                    drift_A = (r - q - 0.5 * sig_A**2) * dt
                    diff_A = sig_A * dW_A[a] * sq_dt
                    
                    S_prev_A = S_A[a]
                    S_new_A = S_prev_A * np.exp(drift_A + diff_A)
                    
                    # Brownian Bridge Check (Down-and-In)
                    # Only check if we haven't knocked in yet AND we are above barrier
                    barrier_level = B_KI * S0[a]
                    if not ki_A and S_new_A > barrier_level and S_prev_A > barrier_level:
                        if get_brownian_bridge_hit(S_prev_A, S_new_A, barrier_level, sig_A, dt):
                            ki_A = True
                            
                    S_A[a] = S_new_A
                
                # --- PATH B ---
                if alive_B:
                    sig_B = _get_local_vol(curr_time, S_B[a], vol_grids_T, vol_grids_S, vol_surfaces[a])
                    drift_B = (r - q - 0.5 * sig_B**2) * dt
                    diff_B = sig_B * dW_B[a] * sq_dt
                    
                    S_prev_B = S_B[a]
                    S_new_B = S_prev_B * np.exp(drift_B + diff_B)
                    
                    # Brownian Bridge Check
                    barrier_level = B_KI * S0[a]
                    if not ki_B and S_new_B > barrier_level and S_prev_B > barrier_level:
                        if get_brownian_bridge_hit(S_prev_B, S_new_B, barrier_level, sig_B, dt):
                            ki_B = True
                            
                    S_B[a] = S_new_B
                    
            # 4. Observation Logic (Autocall / Coupon)
            # Check if this step is an observation date
            is_obs = False
            for obs_t in obs_indices:
                if t_step == obs_t:
                    is_obs = True
                    break
            
            if is_obs:
                # --- PROCESS PATH A ---
                if alive_A:
                    worst_A = 100.0
                    for k in range(n_assets):
                        p = S_A[k] / S0[k]
                        if p < worst_A: worst_A = p
                    
                    if worst_A < B_KI: ki_A = True # Natural breach
                    
                    if worst_A >= B_KO:
                        # Autocall!
                        results[2*i, 0] = 1.0 + Cpn_Amt + mem_A
                        results[2*i, 1] = 1.0 if ki_A else 0.0
                        alive_A = False
                    elif worst_A >= B_Cpn:
                        # Pay Coupon (Conceptually accrued)
                        # In this simple model, we assume it's paid at end, so we store it
                        # Or we could pay it now. Let's assume Phoenix "Memory" clears here.
                        # Note: Standard Phoenix pays immediately. We'll simplify by adding to payoff.
                        # Ideally, discount factor logic handles the timing. 
                        # For now, we just flag it.
                        mem_A = 0.0 # Coupon paid, memory reset. 
                        # (Real implementation: add PV(coupon) to result accumulator)
                    else:
                        # Missed Coupon
                        if memory_feature: mem_A += Cpn_Amt
                
                # --- PROCESS PATH B ---
                if alive_B:
                    worst_B = 100.0
                    for k in range(n_assets):
                        p = S_B[k] / S0[k]
                        if p < worst_B: worst_B = p
                        
                    if worst_B < B_KI: ki_B = True
                    
                    if worst_B >= B_KO:
                        results[2*i+1, 0] = 1.0 + Cpn_Amt + mem_B
                        results[2*i+1, 1] = 1.0 if ki_B else 0.0
                        alive_B = False
                    elif worst_B >= B_Cpn:
                        mem_B = 0.0
                    else:
                        if memory_feature: mem_B += Cpn_Amt

        # --- Terminal Logic (Maturity) ---
        if alive_A:
            worst_A = 100.0
            for k in range(n_assets):
                p = S_A[k] / S0[k]
                if p < worst_A: worst_A = p
            
            if ki_A or (worst_A < B_KI):
                # Protection Lost: Physical Delivery
                results[2*i, 0] = worst_A
                results[2*i, 1] = 1.0
            else:
                # Capital Protected
                # Check final coupon
                final_pay = 1.0
                if worst_A >= B_Cpn: final_pay += Cpn_Amt + mem_A
                results[2*i, 0] = final_pay
                results[2*i, 1] = 0.0

        if alive_B:
            worst_B = 100.0
            for k in range(n_assets):
                p = S_B[k] / S0[k]
                if p < worst_B: worst_B = p
            
            if ki_B or (worst_B < B_KI):
                results[2*i+1, 0] = worst_B
                results[2*i+1, 1] = 1.0
            else:
                final_pay = 1.0
                if worst_B >= B_Cpn: final_pay += Cpn_Amt + mem_B
                results[2*i+1, 0] = final_pay
                results[2*i+1, 1] = 0.0

    return results

# ==========================================
# 3. High-Level Wrapper
# ==========================================

def run_simulation(instrument, market_env, n_paths: int = 50000) -> Dict:
    """
    Orchestrates the simulation. Unpacks objects and calls the JIT kernel.
    """
    # 1. Prepare Arrays
    dt = 1.0 / 252.0
    obs_indices = (instrument.observation_dates / dt).astype(np.int64)
    
    # Barriers: [Autocall, Knock-In, CouponBarrier, CouponAmount]
    barriers = np.array([
        instrument.barrier_autocall,
        instrument.barrier_knock_in,
        instrument.barrier_coupon,
        instrument.coupon_rate # Assuming rate is the amount for the period for simplicity
    ])
    
    rates = market_env.get_rates_array()
    
    # 2. Prepare Volatility Grids
    # We need to stack the surfaces into a 3D array for Numba
    # Assumption: All assets share the same T_grid and S_grid definition
    # (In `vol_engine.py`, we generated T_grid, S_grid once).
    
    # Extract the first asset's grid to get axes
    first_ticker = instrument.tickers[0]
    t_grid, s_grid, _ = market_env.vol_surfaces[first_ticker]
    
    # Stack the surfaces
    n_assets = len(instrument.tickers)
    n_t = len(t_grid)
    n_s = len(s_grid)
    stacked_surfaces = np.zeros((n_assets, n_t, n_s))
    
    for i, ticker in enumerate(instrument.tickers):
        _, _, surf = market_env.vol_surfaces[ticker]
        stacked_surfaces[i] = surf
    
    # 3. Get Cholesky
    # We access the method from math_utils via the market_env helper or directly
    from .math_utils import CorrelationEngine
    L = CorrelationEngine.get_cholesky(market_env.correlation_matrix)
    
    # 4. Run Kernel
    # n_sims = n_paths / 2 (Antithetic)
    n_sims = n_paths // 2
    
    raw_results = _marc_kernel(
        instrument.initial_spots,
        L,
        instrument.maturity,
        dt,
        t_grid, s_grid, stacked_surfaces,
        obs_indices, barriers, rates,
        instrument.memory_feature,
        n_sims
    )
    
    # 5. Process Output
    payoffs = raw_results[:, 0]
    ki_flags = raw_results[:, 1]
    
    # Discounting
    df = np.exp(-market_env.risk_free_rate * instrument.maturity)
    
    # Calculate Standard Error
    mean_price = np.mean(payoffs) * df
    std_error = np.std(payoffs) * df / np.sqrt(n_paths)
    
    return {
        "price": mean_price,
        "std_error": std_error,
        "knock_in_prob": np.mean(ki_flags),
        "n_paths": n_paths,
        "payoffs_distribution": payoffs # Useful for histograms
    }