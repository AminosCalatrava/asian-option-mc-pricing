


# Exotic Option Pricing: Monte Carlo Simulation & Control Variate Variance Reduction

This repository implements a vectorized Python pricing engine for **discrete arithmetic Asian options** under the Black-Scholes-Merton framework, accompanied by a formal mathematical research note written in LaTeX$.


## 🎯 Overview

Unlike standard European vanilla options, the payoff of an Asian option is **path-dependent** and determined by the average price of the underlying asset over a discrete monitoring grid $(t_1, \dots, t_m)$:

$$H_A = \max\left( \frac{1}{m} \sum_{j=1}^m S_{t_j} - K, \, 0 \right)$$

Because the arithmetic average of log-normal random variables lacks a closed-form analytical distribution, numerical evaluation via **Monte Carlo simulation** is required. To overcome the slow $\mathcal{O}(N^{-1/2})$ convergence rate of naive Monte Carlo, we implement two variance reduction techniques:
1. **Antithetic Variates** (exploiting Gaussian symmetry and the FKG inequality).
2. **Control Variates** leveraging the exact closed-form analytical solution of the **Geometric Asian Call**, whose payoff exhibits a $>99.9\%$ correlation with the arithmetic payoff.

---

## 📊 Numerical Results ($N = 100,000$ total path evaluations, `seed = 42`)

**Market Parameters:** $S_0 = 100$, $K = 100$, $r = 5\%$, $\sigma = 20\%$, $T = 1.0\text{ yr}$, $m = 252\text{ steps}$.
* **Exact Discrete Geometric Call ($C_G$):** `5.5664` ($\widetilde{\sigma} = 0.11581$, $\widetilde{r} = 0.02177$)
* **Turnbull-Wakeman Analytical Benchmark ($C_{\text{TW}}$):** `5.7850`
* **Pilot Control Variate Coefficient ($\hat{\beta}^*$, $N_{\text{pilot}} = 10,000$):** `1.0414` (Out-of-sample correlation $\hat{\rho}_{X,Y} = 0.999212$)

| Estimation Method | Estimated Price | Standard Error | 95% Confidence Interval | Equal-Budget VRF | CPU Time | Time-Adjusted Gain |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Standard MC ($N$ paths) | `5.7888` | `0.0254` | `[5.7391, 5.8385]` | **1.0x** | `0.68s` | **1.0x** |
| Antithetic ($N/2$ pairs) | `5.7924` | `0.0178` | `[5.7575, 5.8273]` | **2.0x** | `0.70s` | **2.0x** |
| **Control Variate ($N$ paths)** | **`5.7830`** | **`0.0010`** | **`[5.7810, 5.7850]`** | **634.5x** | `0.99s` | **435.8x** |

### Convergence Visualization
![Monte Carlo Convergence](convergence_mc.png)

> **Key Takeaway:** The geometric control variate reduces estimator variance by a factor of **635x**, achieving with $10^5$ trajectories the same statistical precision that would require over $6.35 \times 10^7$ trajectories using standard Monte Carlo.

---

## 🚀 Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
