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


def backward(x_t, t, dt, mu_star, std, epsilon=None):
    """
    Moves the Ornstein-Uhlenbeck process one step backward in time (t -> t-dt).
    This implements the discrete version of:
    -dy_i(t) = [y_i + 2 * Score_i] dt + sqrt(2) * dB_t
    Which is discretized as:
    x_{t-dt} = x_t + [x_t + 2 * Score(x_t, t)] * dt + sqrt(2 * dt) * epsilon
    """
    if epsilon is None:                                # For reproducibility
        epsilon = torch.randn_like(x_t)                # N(0,1)
    f = -x_t - 2*score(x_t, t, mu_star, std)           # Drift term
    dW_t = np.sqrt(2*dt)*epsilon                       # N(0, dt)
    x_tm1 = x_t - dt*f + dW_t
    return x_tm1, epsilon


def score(x_t, t, mu_star, std):
    """
    Score function of 2D, bimodal multivariate Gaussian
    x_t: Current particle positions. Shape (batch_size, d).
    t: Current diffusion time.
    mu_star: symmetric stationary mean. Shape (d,).
    std: The initial standard deviation clusters.
    """
    ns = x_t.shape[1]
    delta_t = 1 - np.exp(-2*t)
    Gamma_t = delta_t + std**2*np.exp(-2*t)
    mu_t = mu_star * np.exp(-t)
    m = torch.matmul(mu_t, x_t)
    mu_t = mu_t.repeat(ns).reshape(-1, ns)
    return torch.tanh(m/Gamma_t)*mu_t/Gamma_t - x_t/Gamma_t


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
    