import numpy as np
import pandas as pd
from scipy.special import rel_entr

def read_data(file_path):
    """Reads scatter data from a file and returns X and Y arrays."""
    data = pd.read_csv(file_path)
    return data['X'].values, data['Y'].values

def compute_histogram(x, y, bins):
    """Computes a 2D histogram for the scatter data."""
    hist, xedges, yedges = np.histogram2d(x, y, bins=bins, density=True)
    return hist

def js_divergence(P, Q):
    """Computes the Jensen-Shannon divergence between two probability distributions."""
    # Ensure the probability distributions sum to 1
    P = P / np.sum(P)
    Q = Q / np.sum(Q)
    # Compute the mid-point distribution
    M = 0.5 * (P + Q)
    # Compute the KL divergence between P and M, and Q and M
    Dkl_PM = np.sum(rel_entr(P, M))
    Dkl_QM = np.sum(rel_entr(Q, M))
    # Compute the JS divergence
    Djs = 0.5 * Dkl_PM + 0.5 * Dkl_QM
    return Djs

# File paths
file1 = r'..\original.csv'
file2 = r'..\moment.csv'

# Read data from files
x1, y1 = read_data(file1)
x2, y2 = read_data(file2)

# Define the number of bins for the histogram
bins = 5
# Compute histograms
hist1 = compute_histogram(x1, y1, bins)
hist2 = compute_histogram(x2, y2, bins)

# Flatten the histograms to 1D arrays
P = hist1.flatten()
Q = hist2.flatten()

# Compute JS divergence
jsd = js_divergence(P, Q)
jsd = jsd**0.5

print(f"Jensen-Shannon Divergence: {jsd}")
