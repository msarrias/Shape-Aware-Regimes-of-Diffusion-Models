import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import torch
import sys

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
        epsilon = torch.randn_like(x_t)                # N(0,1)
    x_np = x_t.detach().numpy()
    s = score(x_np, t, mu_star, std, model, weights)
    s = torch.tensor(s, dtype=x_np.dtype)
    f = -x_t - 2 * s
    dW = np.sqrt(2 * dt) * epsilon                     # N(0, dt)
    x_tm1 = x_t - dt * f + dW_t
    return x_tm1, epsilon


def centers():
    mu_micro = 3.5  # Distance from XY origin
    mu_macro = 12.0 # Total distance between macro groups (6 - (-6))
    
    angles = np.array([0, 2 * np.pi / 3, 4 * np.pi / 3])
    micro_offsets = np.array([
        [mu_micro * np.cos(a), mu_micro * np.sin(a), 0.0]
        for a in angles
    ])
    
    macro_offset = np.array([0.0, 0.0, mu_macro / 2])
    mu = np.concatenate([
        micro_offsets - macro_offset,
        micro_offsets + macro_offset
    ], axis=0)
    
    return mu
    
def score(x_t, t, mu_star, std, model='bimodal', weights=None):
    """
    Score function of 2D, bimodal multivariate Gaussian
    x_t: Current particle positions. Shape (batch_size, d).
    t: Current diffusion time.
    mu_star: symmetric stationary mean. Shape (d,).
    std: The initial standard deviation clusters.
    """
    if model=='bimodal':
        ns = x_t.shape[1]
        delta_t = 1 - np.exp(-2*t)
        Gamma_t = delta_t + std**2*np.exp(-2*t)
        mu_t = mu_star * np.exp(-t)
        m = torch.matmul(mu_t, x_t)
        mu_t = mu_t.repeat(ns).reshape(-1, ns)
        return torch.tanh(m/Gamma_t)*mu_t/Gamma_t - x_t/Gamma_t
    elif model=='hierarchical':
        sigmas = np.array(std)
        mu_t = mu_star * np.exp(-t)
        Delta_t = 1 - np.exp(-2*t) + sigmas**2 * np.exp(-2*t)
        Delta_t = np.clip(Delta_t, 1e-8, None)
        if weights is None:
            weights = np.ones(len(mu_star)) / len(mu_star)
        else:
            weights = np.array(weights)
            weights = weights / weights.sum()  # normalize to sum to 1
        diff = X[:, None, :] - mu_t[None, :, :]
        log_weights = (np.log(weights)[None, :] - 0.5 * np.sum(diff**2, axis=2) / Delta_t[None, :])  # (n, K)
        post = softmax(log_weights, axis=1)
        x_hat = np.einsum('nk,kd->nd', post, mu_t)
        Delta_t_avg = np.einsum('nk,k->n', post, Delta_t)
        return -(X - x_hat) / Delta_t_avg[:, None]
        
def classify(x, mu_star):
    """
    Predicts which of the classes a particle belongs to.
    It projects the particle onto the axis defined by the class centers.
    """
    # m = np.dot(mu_star, x)
    # return np.sign(m)
    m = torch.matmul(mu_star, x) 
    return torch.sign(m)


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
    