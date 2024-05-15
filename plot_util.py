import matplotlib.pyplot as plt
import torch
from utility import *
from sklearn.decomposition import PCA

c_vals = [
    "firebrick",
    "darkgreen",
    "blue",
    "darkorange",
    "m",
    "deeppink",
    "r",
    "gray",
    "g",
    "navy",
    "y",
    "purple",
    "cyan",
    "olive",
    "skyblue",
    "pink",
    "tan",
]

task_names = (
    'Context DM',
    'Delay Comparison',
    # 'Delay Match Category', # No diff from FF
    'Delay Match Sample',
    'Delay Match Sample Dist.',
    'Delay Paired Association',
    'Dual Delay Match Sample',
    'Go No-go',
    # 'Hierarchical Reasoning', # RL task
    'Interval Discrimination',
    'Motor Timing',
    'Multi-Sensory Integration',
    'One Two Three Go',
    'Perceptual DM',
    'Perc. DM Delay Response',
    'Probabilistic Reasoning',
    # 'Pulse DM', # No diff from FF
    # 'ReachingDelayResponse-v0', # Different type of input
    'Ready-Set-Go',
    'Single Context DM',
)

def plot_acc(load_types, tasks, accs, n_trials):

    # load_colors = (c_vals[5], c_vals[3])
    load_idx_names = []
    for load_type in load_types:
        param_type = load_type.split("/")[0]
        filename = load_type.split("/")[1]
        net_type = filename.split("[")[0]
        load_idx_names.append(f"{net_type} {param_type}")
        
    # load_order = (
    #     2,
    #     1,
    # )  # Puts MPN first

    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 7))

    for load_idx, _ in enumerate(load_types):
        ax1.plot(
            np.arange(len(tasks)),
            np.mean(accs[load_idx], axis=-1),
            color=c_vals[load_idx],
            label=load_idx_names[load_idx],
            marker=".",
            # zorder=load_order[load_idx],
        )

        for task_idx, task in enumerate(tasks):
            ax1.scatter(
                task_idx * np.ones((n_trials,)),
                accs[load_idx, task_idx],
                color=c_vals[load_idx],
                marker=".",
                zorder=-1,
                alpha=0.3,
                linewidth=0,
            )

    ax1.set_xticks(np.arange(len(tasks)))
    ax1.set_xticklabels(task_names, rotation=45, fontsize=8, ha="right")

    ax1.set_ylim((0.4, 1.1))
    ax1.set_yticks((0.5, 0.6, 0.7, 0.8, 0.9, 1.0))
    ax1.set_yticklabels((0.5, None, None, None, None, 1.0))
    ax1.set_ylabel("Accuracy")
    ax1.legend()

    for it in range(0, len(tasks), 2):
        ax1.axvline(it, color="grey", alpha=0.2, zorder=-1)

    # jetplot.breathe(ax=ax1)
    plt.tight_layout()
    savefig("./figures", "ngym_accs", "png")


def plot_norm(net_type, db, batch, save_dir, save_name):
    if net_type in ["GRU"]:
        return
    n_batch = 10
    if net_type == "FreeNet":
        M_hist = db["M"][:n_batch]  # shape: [B, T, Nh, Nx]
    elif net_type == "HebbNet_M":
        breakpoint()
    patterns = M_hist[:, -1, :, :].squeeze().reshape(-1, M_hist.shape[-1])
    cluster = np.ones((M_hist.shape[0], M_hist.shape[2])) * np.array(
        [i for i in range(M_hist.shape[0])]
    ).reshape(-1, 1)
    cluster = cluster.flatten()
    pca = PCA(n_components=2)
    pca.fit(patterns)
    patterns_pca = pca.transform(patterns)
    patterns_pca = patterns_pca + np.random.normal(0, 0.05, patterns_pca.shape)
    for i in range(n_batch):
        plt.scatter(
            patterns_pca[cluster==i, 0],
            patterns_pca[cluster==i, 1],
            cmap="viridis",
        )
        
    # plt.scatter(patterns_pca[:, 0], patterns_pca[:, 1], c=cluster.flatten(), cmap="viridis")
    savefig(save_dir, save_name, "png")