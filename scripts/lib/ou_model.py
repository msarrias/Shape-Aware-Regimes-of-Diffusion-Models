import numba as nb
import numpy as np
import torch
from scipy.special import softmax


from functools import partial
from scipy import integrate

# most code comes from here https://github.com/tbonnair/Dynamical-Regimes-of-Diffusion-Models
def forward(x_0, t):
    """
    Computes the state x_t of the Ornstein-Uhlenbeck process at any time t.
    This is the closed-form solution to dx = -xdt + sqrt(2)dB.
    """
    epsilon = torch.randn_like(x_0) # N(0,1)
    x_t = x_0*np.exp(-t) + np.sqrt(1-np.exp(-2*t))*epsilon
    return x_t, epsilon


def backward(x_t, t, dt, mu_star, std, model_score, epsilon=None):
    """
    Moves the Ornstein-Uhlenbeck process one step backward in time (t -> t-dt).
    This implements the discrete version of:
    -dy_i(t) = [y_i + 2 * Score_i] dt + sqrt(2) * dB_t
    Which is discretized as:
    x_{t-dt} = x_t + [x_t + 2 * Score(x_t, t)] * dt + sqrt(2 * dt) * epsilon
    """
    if epsilon is None:
        epsilon = torch.randn_like(x_t)     # N(0,1)
    s = model_score(x_t, t, mu_star, std, **score_kwargs)
    s = torch.as_tensor(s, dtype=x_t.dtype, device=x_t.device)
    f = -x_t - 2 * s
    dW = np.sqrt(2 * dt) * epsilon                     # N(0, dt)
    x_tm1 = x_t - dt * f + dW
    return x_tm1, epsilon


def bimodal_score(x_t, t, mu_star, std):
    """Bimodal model in R^d.  x_t: (d, N), mu_star: (d,)."""
    alpha = np.exp(-t)
    sigma2_t = alpha ** 2 * std ** 2 + (1 - alpha ** 2)
    mu_t = alpha * mu_star  # (d,)
    term1 = -x_t / sigma2_t
    term2 = torch.tanh(mu_t @ x_t / sigma2_t)[None, :] * mu_t[:, None] / sigma2_t
    return term1 + term2


def embedded_score(x_t, t, mu, std, Q):
    """Embedded bimodal model.
    x_t : (d, N) noisy samples, columns are samples
    t   : float, time
    mu  : (m,) mode center in the latent space R^m
    std : float, within-mode std sigma
    Q   : (d, m) embedding with orthonormal columns (Q^T Q = I_m)
    """
    alpha = np.exp(-t)
    sigma2_t = alpha ** 2 * std ** 2 + (1 - alpha ** 2)  # in-subspace variance
    h_t = 1 - alpha ** 2  # off-subspace variance

    z_t = Q.T @ x_t  # (m, N) latent coordinates
    Px = Q @ z_t  # (d, N) projection P x
    mu_emb = alpha * (Q @ mu)  # (d,)  alpha_t Q mu

    term1a = -Px / sigma2_t  # pull on the subspace part
    term1b = -(x_t - Px) / h_t  # pull on the padding part
    w = torch.tanh(alpha * (mu @ z_t) / sigma2_t)  # (N,) E[c | x_t]
    term2 = w[None, :] * mu_emb[:, None] / sigma2_t  # pull toward the mode
    return term1a + term1b + term2


def embedded_score_canonical(x_t, t, mu, std):
    """Same score for Q = [I_m; 0]: first m coordinates are z_t, the rest n_t."""
    m = mu.shape[0]
    alpha = np.exp(-t)
    h_t = 1 - alpha ** 2
    z_t, n_t = x_t[:m], x_t[m:]
    return torch.cat([bimodal_score(z_t, t, mu, std), -n_t / h_t], dim=0)


def canonical_Q(d, m):
    return torch.cat([torch.eye(m), torch.zeros(d - m, m)], dim=0)


def random_Q(d, m, seed=0):
    g = torch.Generator().manual_seed(seed)
    Q, R = torch.linalg.qr(torch.randn(d, m, generator=g, dtype=torch.float64))
    return Q * torch.sign(torch.diagonal(R))


def theoretical_bimodal_gaussian_ts(mu_star, std, times):
    """
    computes the theoretical speciation time t_s proposed by Biroli et. al,
    and finds its index in the time array.
    :param mu_star: The cluster center vector.
    :param std: The standard deviation (internal variance) of the clusters.
    :param times: The array of time steps used in the simulation.
    """
    if torch.is_tensor(mu_star):
        mu_star_np = mu_star.detach().cpu().numpy()
    else:
        mu_star_np = mu_star
    # Total variance of the mixture at t=0
    # Lambda = ||mu||^2 + sigma^2
    norm_sq = np.linalg.norm(mu_star_np)**2
    Lambda = norm_sq + (std**2)
    # t_s = 1/2 * ln(Lambda)
    # This is the point where the Hessian of the log-density crosses zero
    t_s = np.log(Lambda) / 2.0
    t_s_idx = np.abs(times - t_s).argmin()
    
    return t_s, t_s_idx


def find_third_phase_onset(
        CTDs_raw,
        time_snaps
):
    def minmax(x):
        return (x - x.min()) / (x.max() - x.min())

    mu_hat = [CTDs_raw[t]['norm_ctds'].mean() for t in time_snaps]
    std_hat = [CTDs_raw[t]['norm_ctds'].std() for t in time_snaps]

    t_arr = np.array(time_snaps)

    mu_norm = minmax(np.array(mu_hat))
    std_norm = minmax(np.array(std_hat))

    diff = std_norm - mu_norm
    crossings = np.where(np.diff(np.sign(diff)))[0]

    if len(crossings) == 0:
        print("No crossing found")
        return None

    idx = crossings[-1]
    t_cross = t_arr[idx + 1]

    return t_cross


def classify(x, mu_star):
    """
    Predicts which of the classes a particle belongs to.
    It projects the particle onto the axis defined by the class centers.
    """
    # m = np.dot(mu_star, x)
    # return np.sign(m)
    m = torch.matmul(mu_star, x)
    return torch.sign(m)


def same_cluster_prob(dim, mu, std, times):
    """
    Estimates the probability that two clones will be grouped at the same cluster
    :param dim: Data dimension
    :param mu: The cluster center vector is torch.ones(d) * args.mu
    :param std: The standard deviation (internal variance) of the clusters.
    :param times: The array of time steps used in the simulation.
    """
    phi = np.zeros(len(times))
    m = mu * np.sqrt(dim)
    for (it, t) in enumerate(times):
        c = m*np.exp(-t)
        x0 = -c-5*std
        part = partial(integrand, m, std, t)
        integral = integrate.quad(part, x0, -x0, epsrel=1e-4)[0]
        if np.isnan(integral):
            x0, x1, x2, x3 = -c-5*std, -c+5*std, c-5*std, c+5*std
            integral = integrate.quad(part, x0, x1, epsrel=1e-4)[0]
            integral += integrate.quad(part, x2, x3, epsrel=1e-4)[0]
        phi[it] = 1/2*integral
    return phi


def pos_cluster_prob(y, t, dim, mu, std):
    """
    Calculates the probability that the backward process ends in +mu center
    knowing that it was in y in given time t
    :param dim: Data dimension
    :param mu: The cluster center vector is torch.ones(d) * args.mu
    :param std: The standard deviation (internal variance) of the clusters.
    :param times: The array of time steps used in the simulation.
    """
    delta_t = 1 - np.exp(-2*t)
    Gamma_t = delta_t + std**2*np.exp(-2*t)
    mu_vec = torch.ones(dim) * mu
    y_m = torch.matmul(mu_vec, y)
    x = 2 * np.exp(-t) / Gamma_t * y_m
    return 1 / (1 + np.exp(-x))


@nb.njit(error_model="numpy",fastmath=True)
def gaussian(m, s, x):
    return 1/np.sqrt(2*np.pi*s) * np.exp(-(x-m)**2 / 2 / s)

@nb.njit(error_model="numpy",fastmath=True)
def integrand(m, s, t, x):
    dt = 1 - np.exp(-2*t)
    gt = dt+s*np.exp(-2*t)
    Gp = gaussian(m*np.exp(-t), gt, x)
    Gm = gaussian(-m*np.exp(-t), gt, x)
    itgrd = (Gp**2+Gm**2) / (Gp + Gm)
    return itgrd