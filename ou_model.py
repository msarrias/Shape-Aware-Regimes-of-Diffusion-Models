import numpy as np
import matplotlib as mpl
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import torch
import sys
from scipy.special import softmax


# most code comes from here https://github.com/tbonnair/Dynamical-Regimes-of-Diffusion-Models
def forward(x_0, t):
    """
    Computes the state x_t of the Ornstein-Uhlenbeck process at any time t.
    This is the closed-form solution to dx = -xdt + sqrt(2)dB.
    """
    epsilon = torch.randn_like(x_0) # N(0,1)
    x_t = x_0*np.exp(-t) + np.sqrt(1-np.exp(-2*t))*epsilon
    return x_t, epsilon


def backward(x_t, t, dt, mu_star, std, model, weights=None, epsilon=None):
    """
    Moves the Ornstein-Uhlenbeck process one step backward in time (t -> t-dt).
    This implements the discrete version of:
    -dy_i(t) = [y_i + 2 * Score_i] dt + sqrt(2) * dB_t
    Which is discretized as:
    x_{t-dt} = x_t + [x_t + 2 * Score(x_t, t)] * dt + sqrt(2 * dt) * epsilon
    """
    if epsilon is None:                                # For reproducibility
        epsilon = torch.randn_like(x_t)     # N(0,1)
    s = score(x_t, t, mu_star, std, model, weights)
    s = torch.tensor(s, dtype=x_t.dtype)
    f = -x_t - 2 * s
    dW = np.sqrt(2 * dt) * epsilon                     # N(0, dt)
    x_tm1 = x_t - dt * f + dW
    return x_tm1, epsilon


def centers(d, mu_micro, mu_macro, n_inner=3):
    angles = np.array([2 * np.pi * k / n_inner for k in range(n_inner)])

    scale = np.sqrt(d)  # scale separation with sqrt(d)

    micro_offsets = np.zeros((n_inner, d))
    micro_offsets[:, 0] = mu_micro * np.cos(angles) * scale
    micro_offsets[:, 1] = mu_micro * np.sin(angles) * scale

    macro_offset = np.zeros(d)
    macro_offset[-1] = mu_macro * scale

    mu = np.concatenate([
        micro_offsets - macro_offset,
        micro_offsets + macro_offset
    ], axis=0)
    return mu


def score(x_t, t, mu_star, std, model='bimodal', weights=None):
    """
    Score function for three models, all working in arbitrary dimension d.
    x_t   : (d, N) tensor
    t     : float
    mu_star: 
        bimodal    — (d,) tensor, the single mean
        hierarchical — (K, d) array, the K cluster means
        rings      — ignored, rings defined by Q
    std   : 
        bimodal      — scalar tensor
        hierarchical — (K,) array, per-cluster std
        rings        — scalar, manifold noise level
    Q     : (d, d) orthogonal matrix for rings embedding (optional).
              If None, rings live in dims 0,1,2 with zeros elsewhere.
    """

    if model == 'bimodal':
        ns = x_t.shape[1]
        delta_t = 1 - np.exp(-2*t)
        Gamma_t = delta_t + std**2 * np.exp(-2*t)
        mu_t = mu_star * np.exp(-t)
        m = torch.matmul(mu_t, x_t)
        mu_t = mu_t.repeat(ns).reshape(-1, ns)
        return torch.tanh(m / Gamma_t) * mu_t / Gamma_t - x_t / Gamma_t

    elif model == 'hierarchical':
        X = x_t.T.detach().numpy() if isinstance(x_t, torch.Tensor) else x_t.T  # (N, d)                     # (N, d)
        d = X.shape[1]
        sigmas  = np.array(std)
        mu_t = mu_star * np.exp(-t)           # (K, d)
        Delta_t = 1 - np.exp(-2*t) + sigmas**2 * np.exp(-2*t)
        Delta_t = np.clip(Delta_t, 1e-8, None)  # (K,)
        if weights is None:
            weights = np.ones(len(mu_star)) / len(mu_star)
        diff = X[:, None, :] - mu_t[None, :, :]           # (N, K, d)
        # divide by d to prevent softmax collapse in high dimensions
        log_weights = np.log(weights)[None, :] - 0.5 * np.sum(diff**2, axis=2) / (Delta_t[None, :] * d)
        
        post = softmax(log_weights, axis=1)                # (N, K)
        x_hat = np.einsum('nk,kd->nd', post, mu_t)         # (N, d)
        Delta_t_avg = np.einsum('nk,k->n',   post, Delta_t)      # (N,)
        s_np = -(X - x_hat) / Delta_t_avg[:, None]        # (N, d)
        return s_np.T                                             # (d, N)
    else:
        raise ValueError('Unknown model: {}'.format(model))
        
def classify(x, mu_star):
    """
    Predicts which of the classes a particle belongs to.
    It projects the particle onto the axis defined by the class centers.
    """
    # m = np.dot(mu_star, x)
    # return np.sign(m)
    m = torch.matmul(mu_star, x) 
    return torch.sign(m)

def classify_hierarchical(X, mu_star):
    """
    X       : (N, d)
    mu_star : (K, d) cluster centers
    Returns : (N,) integer labels 0..K-1
    """
    # (N, 1, d) - (1, K, d) -> (N, K, d)
    diff   = X[:, None, :] - mu_star[None, :, :]   # (N, K, d)
    dists  = np.linalg.norm(diff, axis=2)           # (N, K)
    labels = dists.argmin(axis=1)                   # (N,)
    return labels

def theoretical_ts(mu_star, std, times):
    """
    computes the theoretical speciation time t_s and finds its index in the time array.
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
    