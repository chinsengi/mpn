import matplotlib.pyplot as plt
import torch
from utility import *
from sklearn.decomposition import PCA

def plot_norm(db, batch, save_dir, save_name):
    M_hist = db["M"] # shape: [B, T, Nh, Nx]
    

