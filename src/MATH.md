# Mathematical Model Reference

This document details the quantitative framework used in the Multi-Asset Autocallable Reverse Convertible (MARC) Pricing Engine. It covers the stochastic dynamics, correlation structure, local volatility calibration, and the numerical methods used for pricing.

## 1. Asset Dynamics: Multi-Asset Local Volatility

We assume a risk-neutral measure $\mathbb{Q}$ where the price process of the $i$-th asset $S_i(t)$ (for $i=1, \dots, N$) evolves according to the following Stochastic Differential Equation (SDE):

$$
\frac{dS_i(t)}{S_i(t)} = (r(t) - q_i(t)) dt + \sigma_{loc, i}(S_i, t) dW_i(t)
$$

Where:
* $S_i(t)$: Spot price of asset $i$ at time $t$.
* $r(t)$: Risk-free interest rate (assumed deterministic).
* $q_i(t)$: Continuous dividend yield for asset $i$.
* $\sigma_{loc, i}(S, t)$: Local volatility function for asset $i$.
* $W_i(t)$: Standard Brownian motion for asset $i$.

### 1.1 Correlation Structure
The asset returns are correlated. The relationship between the Brownian motions is defined by the instantaneous correlation matrix $\rho$:

$$
d\langle W_i, W_j \rangle_t = \rho_{ij} dt
$$

To simulate this system, we require the correlation matrix $\Sigma$ to be Positive Semi-Definite (PSD). However, empirical correlation matrices are frequently non-PSD due to asynchronous market data or stress-testing overrides. To address this, we implement **Higham's Algorithm** (2002) to compute the nearest valid PSD correlation matrix $\Sigma_{PSD}$ in the Frobenius norm.

Once the valid matrix is obtained, we perform a **Cholesky Decomposition** to find the lower triangular matrix $L$ such that:

$$
\Sigma_{PSD} = L L^T
$$

During the Monte Carlo simulation, we generate a vector of independent standard normal random variables $Z \sim N(0, I)$ and transform them into correlated variables $Z_{corr}$:

$$
Z_{corr} = L \cdot Z
$$
## 2. Local Volatility Calibration (Dupire's Formula)

To ensure the model accurately prices vanilla European options (preserving the volatility smile/skew observed in the market), we use **Dupire's Local Volatility** model. The local volatility $\sigma_{loc}(K, T)$ is derived from the market's implied volatility surface $\sigma_{imp}(K, T)$:

$$
\sigma_{loc}^2(K, T) = \frac{\frac{\partial C}{\partial T} + (r - q)K \frac{\partial C}{\partial K} + qC}{K^2 \frac{\partial^2 C}{\partial K^2}}
$$

In terms of implied variance $w(K, T) = \sigma_{imp}^2(K, T) \cdot T$, this is often implemented as:

$$
\sigma_{loc}^2 = \frac{\frac{\partial w}{\partial T}}{1 - \frac{K}{w}\frac{\partial w}{\partial K} + \frac{1}{4}\left(-\frac{K^2}{w^2} + \frac{1}{w}\right)\left(\frac{\partial w}{\partial K}\right)^2 + \frac{1}{2}\frac{\partial^2 w}{\partial K^2}}
$$

*Note: In the implementation, we use bicubic spline interpolation on the implied volatility surface to ensure smooth partial derivatives.*

## 3. Payoff Logic: The "Worst-Of" Mechanism

The MARC product's performance is driven by the worst-performing asset in the basket. We define the performance ratio for asset $i$ at time $t$ as:

$$
Perf_i(t) = \frac{S_i(t)}{S_i(0)}
$$

The basket performance is defined as:

$$
P_{basket}(t) = \min_{i=1, \dots, N} \left( Perf_i(t) \right)
$$

### 3.1 Knock-In Barrier (Downside Risk)
The product has a barrier level $B_{KI}$ (e.g., 60% of initial fix).
* **European Barrier:** Checked only at maturity $T$. condition: $P_{basket}(T) < B_{KI}$.
* **American Barrier:** Checked continuously. condition: $\min_{t \in [0, T]} P_{basket}(t) < B_{KI}$.

### 3.2 Payoff Function
The payoff at maturity $T$ is generally:

$$
\text{Payoff}(T) = \text{Coupon} + \begin{cases} 
100\% & \text{if } \text{Barrier Not Breached} \\
P_{basket}(T) & \text{if } \text{Barrier Breached} 
\end{cases}
$$

*(Note: If the barrier is breached, the investor receives the physical delivery of the worst-performing stock, effectively $P_{basket}(T)$).*

## 4. Numerical Methods

### 4.1 Discretization (Euler-Maruyama)
We simulate the paths using the log-Euler scheme for stability. For a time step $\Delta t$:

$$
\ln S_i(t+\Delta t) = \ln S_i(t) + \left( r - q_i - \frac{1}{2}\sigma_{loc}^2 \right)\Delta t + \sigma_{loc} \sqrt{\Delta t} Z_{corr, i}
$$

### 4.2 Brownian Bridge (Barrier Correction)
For American barriers, discrete monitoring (e.g., daily) underestimates the probability of hitting the barrier compared to continuous monitoring. We apply a **Brownian Bridge** probability correction.

The probability that the asset hit the barrier $B$ in the interval $[t, t+\Delta t]$ given start point $S_t$ and end point $S_{t+\Delta t}$ (where both are above $B$) is:

$$
P(\text{Hit}) = \exp \left( \frac{-2 \ln(S_t/B) \ln(S_{t+\Delta t}/B)}{\sigma^2 \Delta t} \right)
$$

We treat a path as "knocked-in" if this probability exceeds a uniform random draw $U \sim [0,1]$.

### 4.3 Greeks Calculation (Bump and Reval)
We calculate sensitivities using Finite Differences:

* **Delta ($\Delta_i$):** $\frac{\partial V}{\partial S_i} \approx \frac{V(S_i + \epsilon) - V(S_i - \epsilon)}{2\epsilon}$
* **Vega ($\nu_i$):** $\frac{\partial V}{\partial \sigma_i}$ (Shift the entire vol surface).
* **Cross Gamma ($\Gamma_{ij}$):** $\frac{\partial^2 V}{\partial S_i \partial S_j}$ (Crucial for correlation risk).
