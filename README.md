Multi-Asset Autocallable Reverse Convertible (MARC) Pricing EngineA high-performance quantitative finance library for pricing and risk-managing Multi-Asset Autocallable Reverse Convertibles (MARC). This engine utilizes a Multi-Asset Local Volatility framework and Numba-accelerated Monte Carlo simulations to deliver institutional-grade speed and accuracy.Key FeaturesMulti-Asset Stochastic Modeling: Correlated Geometric Brownian Motion (GBM) using Cholesky Decomposition with automated PSD (Positive Semi-Definite) matrix correction.Market Realism: Implementation of Dupire’s Local Volatility to capture the volatility skew/smile, moving beyond the limitations of constant volatility models.Complex Payoff Support:"Worst-Of" Logic: Performance linked to the laggard of a 3-5 stock basket.Memory Coupons (Snowball): Path-dependent coupon accumulation.Barriers: Supports both European (maturity-only) and American (continuous) downside barriers with Brownian Bridge corrections.Numerical Optimizations:Numba JIT Compilation: Achieving near-C++ execution speeds.Variance Reduction: Antithetic Variates and Control Variates (Geometric Basket Put) for superior convergence.The Greeks: Finite difference (Bump-and-Reval) calculations for Delta, Cross-Gamma, and Vega, specifically optimized for identifying "Negative Gamma" traps near barriers.Project StructureMARC_Pricing_Engine/
├── src/
│   ├── math_core/         # GBM Kernels, Cholesky, Spline Surfaces
│   ├── instruments/       # MARC & Exotic Payoff Definitions
│   ├── analytics/         # Greeks & Variance Reduction Logic
│   └── utils/             # Market data handlers & Configuration
├── app/                   # Streamlit Dashboard for Visualization
├── tests/                 # Unit tests & Analytical benchmarks
├── notebooks/             # Research & Model Validation
└── requirements.txt
Mathematical Core1. The "Worst-Of" DynamicsThe engine tracks the relative performance of $N$ assets:$$P(t) = \min_{i=1, \dots, N} \left( \frac{S_i(t)}{S_i(0)} \right)$$The correlation structure $\rho_{ij}$ is managed via Cholesky decomposition $\Sigma = LL^T$, ensuring consistent joint movement across the basket.2. Local Volatility FrameworkTo price the downside protection (Knock-In barrier) accurately, we compute the local volatility surface $\sigma_{loc}(S, t)$ using Dupire's formula, calibrated to market-implied volatility splines.3. Numerical EfficiencyThe Monte Carlo kernel is decorated with @njit(parallel=True), allowing 100,000+ paths to be processed in under 5 seconds on standard multicore hardware.Visualization & DashboardThe project includes a Streamlit dashboard designed for Structurers and Traders to:Price at Par: Solve for the required coupon given specific barrier levels.Stress Testing: Visualize the "Negative Gamma" profile as assets approach the Autocall barrier.Convergence Analysis: Monitor simulation stability via standard error plots.Master the TheoryTo fully understand the implementation, refer to the following chapters in 'Options, Futures, and Other Derivatives' by John Hull:Chapter 15: BSM Model (Foundations)Chapter 19: Greek Letters (Dynamic Hedging)Chapter 20: Volatility Smiles (Local Vol Logic)Chapter 21: Basic Numerical Procedures (Monte Carlo & Cholesky)Chapter 26: Exotic Options (Barrier & Basket Payoffs)Installationgit clone [https://github.com/your-username/marc-pricing-engine.git](https://github.com/your-username/marc-pricing-engine.git)
cd marc-pricing-engine
pip install -r requirements.txt
Usagefrom src.math_core.gbm_numba import simulate_paths
from src.instruments.marc import MARCContract

# Define your basket and barriers
contract = MARCContract(
    assets=['AAPL', 'MSFT', 'GOOGL'],
    ko_barrier=1.0, 
    ki_barrier=0.6,
    coupon=0.08
)

# Run the accelerated engine
price = contract.price(n_paths=100000)
print(f"MARC Fair Value: {price:.4f}")
Disclaimer: This software is for educational and research purposes. It is not intended for live trading without further validation.