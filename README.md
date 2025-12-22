# Multi-Asset Autocallable Reverse Convertible (MARC) Pricing Engine

A high-performance quantitative finance library for pricing and risk-managing Multi-Asset Autocallable Reverse Convertibles (MARC). This engine utilizes a Multi-Asset Local Volatility framework and Numba-accelerated Monte Carlo simulations to deliver institutional-grade speed and accuracy.

## Key Features

* **Multi-Asset Stochastic Modeling:** Correlated Geometric Brownian Motion (GBM) using Cholesky Decomposition with automated PSD (Positive Semi-Definite) matrix correction.
* **Market Realism:** Implementation of Dupire’s Local Volatility to capture the volatility skew/smile, moving beyond the limitations of constant volatility models.
* **Complex Payoff Support:**
    * **"Worst-Of" Logic:** Performance linked to the laggard of a 3-5 stock basket.
    * **Memory Coupons (Snowball):** Path-dependent coupon accumulation.
    * **Barriers:** Supports both European (maturity-only) and American (continuous) downside barriers with Brownian Bridge corrections.
* **Numerical Optimizations:**
    * **Numba JIT Compilation:** Achieving near-C++ execution speeds.
    * **Variance Reduction:** Antithetic Variates and Control Variates (Geometric Basket Put) for superior convergence.
* **The Greeks:** Finite difference (Bump-and-Reval) calculations for Delta, Cross-Gamma, and Vega, specifically optimized for identifying "Negative Gamma" traps near barriers.

## Project Structure

```text

MARC-Pricing-Engine/
│
├── src/                          # THE COMPUTATIONAL CORE
│   ├── __init__.py
│   ├── instruments.py            # [Data Layer] Defines the 'MARC_Instrument' and 'MarketEnvironment' classes.
│   ├── math_utils.py             # [Math Layer] Spectral Projection, Cholesky, and JIT-compiled Brownian Bridge.
│   ├── vol_engine.py             # [Model Layer] Dupire Local Volatility construction (Splines & Skew).
│   └── simulation.py             # [Engine Layer] The high-performance Numba/LLVM Monte Carlo Kernel.
│
├── tests/                        # UNIT TESTING SUITE
│   ├── __init__.py
│   └── test_math.py              # Verifies Matrix Repair and Brownian Bridge probability logic.
│
├── notebooks/                    # ANALYSIS & DEMOS
│   ├── 01_Visual_Debugging.ipynb # (Optional) The visualization code to plot Surfaces and Paths.
│   └── 02_Pricing_Demo.ipynb     # The main entry point showing how to price a note.
│
├── requirements.txt              # Dependencies (numpy, scipy, numba, matplotlib, seaborn)
├── README.md                     # Documentation
└── .gitignore                    # Python cache and notebook checkpoints
