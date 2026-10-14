import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 300,
})





def closed_form_geometric_asian_call(S0, K, r, sigma, T, m):
  """Computes the exact analytical price of a discrete Geometric Asian Call"""
  sigma_tilde = sigma * np.sqrt((m + 1) * (2 * m + 1) / (6 * m**2))
  r_tilde = (r - 0.5 * sigma**2) * ((m + 1) / (2 * m)) + 0.5 * sigma_tilde**2

  d1 = (np.log(S0 / K) + (r_tilde + 0.5 * sigma_tilde**2) * T) / (
      sigma_tilde * np.sqrt(T)
  )
  d2 = d1 - sigma_tilde * np.sqrt(T)

  price = np.exp(-r * T) * (
      S0 * np.exp(r_tilde * T) * norm.cdf(d1) - K * norm.cdf(d2)
  )
  return price


def simulate_asian_options(S0, K, r, sigma, T, m, N, seed=42):
  """Simulates GBM paths and computes Standard MC, Antithetic, and Control Variate estimators"""
  np.random.seed(seed)
  dt = T / m

  Z = np.random.standard_normal((N, m))

  log_returns = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
  S_paths = S0 * np.exp(np.cumsum(log_returns, axis=1))

  log_returns_anti = (r - 0.5 * sigma**2) * dt - sigma * np.sqrt(dt) * Z
  S_paths_anti = S0 * np.exp(np.cumsum(log_returns_anti, axis=1))

  A_T = np.mean(S_paths, axis=1)
  Y = np.exp(-r * T) * np.maximum(A_T - K, 0)

  G_T = np.exp(np.mean(np.log(S_paths), axis=1))
  X = np.exp(-r * T) * np.maximum(G_T - K, 0)

  A_T_anti = np.mean(S_paths_anti, axis=1)
  Y_anti_pair = 0.5 * (Y + np.exp(-r * T) * np.maximum(A_T_anti - K, 0))

  C_G_exact = closed_form_geometric_asian_call(S0, K, r, sigma, T, m)
  cov_matrix = np.cov(X, Y)
  beta_opt = cov_matrix[0, 1] / cov_matrix[0, 0]
  corr_XY = cov_matrix[0, 1] / np.sqrt(cov_matrix[0, 0] * cov_matrix[1, 1])

  Y_cv = Y - beta_opt * (X - C_G_exact)

  return S_paths, Y, Y_anti_pair, Y_cv, C_G_exact, beta_opt, corr_XY


if __name__ == "__main__":

  S0, K = 100.0, 100.0
  r, sigma, T = 0.05, 0.20, 1.0
  m, N = 252, 100000

  S_paths, Y, Y_anti, Y_cv, C_G, beta, corr = simulate_asian_options(
      S0, K, r, sigma, T, m, N
  )

  for name, sample in [
      ("Standard MC", Y),
      ("Antithetic Variates", Y_anti),
      ("Control Variate", Y_cv),
  ]:
    mean = np.mean(sample)
    se = np.std(sample, ddof=1) / np.sqrt(N)
    vrf = np.var(Y, ddof=1) / np.var(sample, ddof=1)
    print(
        f"{name:22s} | Price: {mean:.4f} | SE: {se:.4f} | 95% CI:"
        f" [{mean-1.96*se:.4f}, {mean+1.96*se:.4f}] | VRF: {vrf:.1f}x"
    )

  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

  time_grid = np.linspace(T / m, T, m)
  for i in range(15):
    ax1.plot(time_grid, S_paths[i], alpha=0.5, lw=1)
  ax1.axhline(
      K, color="black", linestyle="--", lw=1.2, label="Strike $K = 100$"
  )
  ax1.set_title(r"Sample GBM Trajectories under $\mathbb{Q}$ ($m=252$)")
  ax1.set_xlabel("Time $t$ (years)")
  ax1.set_ylabel("Underlying Asset Price $S_t$")
  ax1.legend(loc="upper left")
  ax1.grid(True, alpha=0.3)

  n_grid = np.arange(500, N + 1, 500)
  cum_mc = np.cumsum(Y)[n_grid - 1] / n_grid
  cum_anti = np.cumsum(Y_anti)[n_grid - 1] / n_grid
  cum_cv = np.cumsum(Y_cv)[n_grid - 1] / n_grid

  se_mc_grid = np.std(Y) / np.sqrt(n_grid)
  se_cv_grid = np.std(Y_cv) / np.sqrt(n_grid)

  ax2.plot(
      n_grid, cum_mc, label="Standard Monte Carlo", color="#1f77b4", lw=1.2
  )
  ax2.fill_between(
      n_grid,
      cum_mc - 1.96 * se_mc_grid,
      cum_mc + 1.96 * se_mc_grid,
      color="#1f77b4",
      alpha=0.15,
  )

  ax2.plot(
      n_grid,
      cum_anti,
      label="Antithetic Variates",
      color="#ff7f0e",
      lw=1.2,
      alpha=0.8,
  )
  ax2.plot(
      n_grid,
      cum_cv,
      label="Control Variate (Geometric)",
      color="#d62728",
      lw=1.8,
  )
  ax2.fill_between(
      n_grid,
      cum_cv - 1.96 * se_cv_grid,
      cum_cv + 1.96 * se_cv_grid,
      color="#d62728",
      alpha=0.3,
  )

  ax2.set_title("Estimator Convergence & 95% Confidence Intervals")
  ax2.set_xlabel("Number of Simulations $N$")
  ax2.set_ylabel("Estimated Asian Call Price")
  ax2.set_ylim(5.60, 5.95)
  ax2.legend(loc="upper right")
  ax2.grid(True, alpha=0.3)

  plt.tight_layout()
  plt.savefig("convergence_mc.png")
  print("\nPlot successfully saved as 'convergence_mc.png'.")
