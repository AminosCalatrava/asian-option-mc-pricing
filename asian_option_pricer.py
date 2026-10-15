import time
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
  return price, sigma_tilde, r_tilde, d1, d2


def turnbull_wakeman_approximation(S0, K, r, sigma, T, m):
  """Computes the Turnbull-Wakeman (1991) log-normal moment-matching approximation for the discrete Arithmetic Asian Call as an independent benchmark"""
  dt = T / m
  t_grid = np.arange(1, m + 1) * dt

  M1 = (S0 / m) * np.sum(np.exp(r * t_grid))

  i_idx, j_idx = np.meshgrid(t_grid, t_grid)
  cov_terms = np.exp(r * (i_idx + j_idx) + sigma**2 * np.minimum(i_idx, j_idx))
  M2 = (S0**2 / m**2) * np.sum(cov_terms)

  sigma_A = np.sqrt(np.log(M2 / (M1**2)) / T)
  d1 = (np.log(M1 / K) + 0.5 * sigma_A**2 * T) / (sigma_A * np.sqrt(T))
  d2 = d1 - sigma_A * np.sqrt(T)

  return np.exp(-r * T) * (M1 * norm.cdf(d1) - K * norm.cdf(d2))


def verify_m1_consistency(S0, K, r, sigma, T):
  """Sanity check: for m=1, Asian Call reduces to European Black-Scholes Call"""
  c_g, _, _, _, _ = closed_form_geometric_asian_call(S0, K, r, sigma, T, m=1)
  d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
  d2 = d1 - sigma * np.sqrt(T)
  c_bs = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
  assert np.isclose(
      c_g, c_bs, atol=1e-12
  ), "Error: m=1 does not match Black-Scholes!"
  print(f"[Sanity Check m=1] C_G(m=1) = {c_g:.6f} == Black-Scholes = {c_bs:.6f}")


def run_experiments(S0, K, r, sigma, T, m, N=100000, N_pilot=10000, seed=42):
  """Runs Equal-Budget Monte Carlo, Antithetic (N/2 pairs), and Out-of-Sample Control Variate."""
  rng = np.random.default_rng(seed)
  dt = T / m
  C_G_exact, sigma_tilde, r_tilde, d1, d2 = closed_form_geometric_asian_call(
      S0, K, r, sigma, T, m
  )
  t0_cv = time.perf_counter()
  Z_pilot = rng.standard_normal((N_pilot, m))
  S_pilot = S0 * np.exp(
      np.cumsum((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z_pilot, axis=1)
  )
  Y_pilot = np.exp(-r * T) * np.maximum(np.mean(S_pilot, axis=1) - K, 0)
  X_pilot = np.exp(-r * T) * np.maximum(
      np.exp(np.mean(np.log(S_pilot), axis=1)) - K, 0
  )
  cov_pilot = np.cov(X_pilot, Y_pilot, ddof=1)
  beta_hat = cov_pilot[0, 1] / cov_pilot[0, 0]
  t0_mc = time.perf_counter()
  Z = rng.standard_normal((N, m))
  log_returns = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
  S_paths = S0 * np.exp(np.cumsum(log_returns, axis=1))
  A_T = np.mean(S_paths, axis=1)
  Y_mc = np.exp(-r * T) * np.maximum(A_T - K, 0)
  time_mc = time.perf_counter() - t0_mc
  t0_cv_eval = time.perf_counter()
  G_T = np.exp(np.mean(np.log(S_paths), axis=1))
  X_mc = np.exp(-r * T) * np.maximum(G_T - K, 0)
  Y_cv = Y_mc - beta_hat * (X_mc - C_G_exact)
  time_cv = (t0_mc - t0_cv) + time_mc + (time.perf_counter() - t0_cv_eval)
  corr_XY = np.corrcoef(X_mc, Y_mc)[0, 1]
  N_pairs = N // 2
  t0_anti = time.perf_counter()
  Z_half = Z[:N_pairs, :]
  S_pos = S0 * np.exp(
      np.cumsum((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z_half, axis=1)
  )
  S_neg = S0 * np.exp(
      np.cumsum((r - 0.5 * sigma**2) * dt - sigma * np.sqrt(dt) * Z_half, axis=1)
  )
  Y_pos = np.exp(-r * T) * np.maximum(np.mean(S_pos, axis=1) - K, 0)
  Y_neg = np.exp(-r * T) * np.maximum(np.mean(S_neg, axis=1) - K, 0)
  Y_anti_equal = 0.5 * (Y_pos + Y_neg)
  time_anti = time.perf_counter() - t0_anti

  return {
      "S_paths": S_paths,
      "Y_mc": Y_mc,
      "Y_anti": Y_anti_equal,
      "Y_cv": Y_cv,
      "C_G": C_G_exact,
      "sigma_tilde": sigma_tilde,
      "r_tilde": r_tilde,
      "d1": d1,
      "d2": d2,
      "beta_hat": beta_hat,
      "corr_XY": corr_XY,
      "times": (time_mc, time_anti, time_cv),
  }


if __name__ == "__main__":
  S0, K, r, sigma, T, m, N = 100.0, 100.0, 0.05, 0.20, 1.0, 252, 100000

  verify_m1_consistency(S0, K, r, sigma, T)
  tw_price = turnbull_wakeman_approximation(S0, K, r, sigma, T, m)
  res = run_experiments(S0, K, r, sigma, T, m, N=N, N_pilot=10000, seed=42)

  print(
      f"\n[Analytical Geometric Call] C_G = {res['C_G']:.4f} "
      f"(sigma_tilde={res['sigma_tilde']:.5f}, r_tilde={res['r_tilde']:.5f}, "
      f"d1={res['d1']:.4f}, d2={res['d2']:.4f})"
  )
  print(f"[Turnbull-Wakeman Benchmark] C_TW = {tw_price:.4f}")
  print(
      f"[Pilot Control Variate] beta_hat = {res['beta_hat']:.4f}, "
      f"Out-of-sample rho_XY = {res['corr_XY']:.6f}\n"
  )

  se_mc = np.std(res["Y_mc"], ddof=1) / np.sqrt(N)
  se_anti = np.std(res["Y_anti"], ddof=1) / np.sqrt(N // 2)
  se_cv = np.std(res["Y_cv"], ddof=1) / np.sqrt(N)

  t_mc, t_anti, t_cv = res["times"]

  for name, sample, se, t_exec in [
      ("Standard MC (N paths)", res["Y_mc"], se_mc, t_mc),
      ("Antithetic (N/2 pairs)", res["Y_anti"], se_anti, t_anti),
      ("Control Variate (N paths)", res["Y_cv"], se_cv, t_cv),
  ]:
    mean = np.mean(sample)
    vrf = (se_mc / se) ** 2
    eff_gain = vrf * (t_mc / t_exec)
    print(
        f"{name:26s} | Price: {mean:.4f} | SE: {se:.4f} | "
        f"95% CI: [{mean-1.96*se:.4f}, {mean+1.96*se:.4f}] | "
        f"Equal-Budget VRF: {vrf:6.1f}x | Time: {t_exec:.3f}s | Time-Adj Gain:"
        f" {eff_gain:5.1f}x"
    )

  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

  time_grid = np.linspace(T / m, T, m)
  for i in range(15):
    ax1.plot(time_grid, res["S_paths"][i], alpha=0.5, lw=1)
  ax1.axhline(
      K, color="black", linestyle="--", lw=1.2, label="Strike $K = 100$"
  )
  ax1.set_title(r"Sample GBM Trajectories under $\mathbb{Q}$ ($m=252$)")
  ax1.set_xlabel("Time $t$ (years)")
  ax1.set_ylabel("Underlying Asset Price $S_t$")
  ax1.legend(loc="upper left")
  ax1.grid(True, alpha=0.3)
  eval_grid = np.arange(1000, N + 1, 1000)
  pair_grid = eval_grid // 2

  cum_mc = np.cumsum(res["Y_mc"])[eval_grid - 1] / eval_grid
  cum_anti = np.cumsum(res["Y_anti"])[pair_grid - 1] / pair_grid
  cum_cv = np.cumsum(res["Y_cv"])[eval_grid - 1] / eval_grid

  se_mc_grid = np.std(res["Y_mc"], ddof=1) / np.sqrt(eval_grid)
  se_cv_grid = np.std(res["Y_cv"], ddof=1) / np.sqrt(eval_grid)

  ax2.plot(
      eval_grid, cum_mc, label="Standard Monte Carlo", color="#1f77b4", lw=1.2
  )
  ax2.fill_between(
      eval_grid,
      cum_mc - 1.96 * se_mc_grid,
      cum_mc + 1.96 * se_mc_grid,
      color="#1f77b4",
      alpha=0.15,
  )
  ax2.plot(
      eval_grid,
      cum_anti,
      label="Antithetic Variates (Equal Budget)",
      color="#ff7f0e",
      lw=1.2,
      alpha=0.85,
  )
  ax2.plot(
      eval_grid,
      cum_cv,
      label="Control Variate (Geometric)",
      color="#d62728",
      lw=1.8,
  )
  ax2.fill_between(
      eval_grid,
      cum_cv - 1.96 * se_cv_grid,
      cum_cv + 1.96 * se_cv_grid,
      color="#d62728",
      alpha=0.3,
  )

  ax2.set_title("Estimator Convergence vs. Total Path Evaluations")
  ax2.set_xlabel("Total Path Evaluations ($N$)")
  ax2.set_ylabel("Estimated Asian Call Price")
  ax2.set_ylim(5.60, 5.95)
  ax2.legend(loc="upper right")
  ax2.grid(True, alpha=0.3)

  plt.tight_layout()
  plt.savefig("convergence_mc.png")
