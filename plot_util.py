import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from sklearn.decomposition import PCA

from utility import create_dir, savefig

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

c_vals_dl = [
    "#c53030",
    "#2b6cb0",
    "#2f855a",
    "#6b46c1",
    "#c05621",
    "#2c7a7b",
    "#4a5568",
    "#b83280",
    "#b7791f",
]

task_names = {
    "ContextDecisionMaking-v0": "Context DM",
    "DelayComparison-v0": "Delay Comparison",
    "DelayMatchSample-v0": "Delay Match Sample",
    "DelayMatchSampleDistractor1D-v0": "Delay Match Sample Dist.",
    "DelayPairedAssociation-v0": "Delay Paired Association",
    "DualDelayMatchSample-v0": "Dual Delay Match Sample",
    "GoNogo-v0": "Go No-go",
    "IntervalDiscrimination-v0": "Interval Discrimination",
    "MotorTiming-v0": "Motor Timing",
    "MultiSensoryIntegration-v0": "Multi-Sensory Integration",
    "OneTwoThreeGo-v0": "One Two Three Go",
    "PerceptualDecisionMaking-v0": "Perceptual DM",
    "PerceptualDecisionMakingDelayResponse-v0": "Perc. DM Delay Response",
    "ProbabilisticReasoning-v0": "Probabilistic Reasoning",
    "ReadySetGo-v0": "Ready-Set-Go",
    "SingleContextDecisionMaking-v0": "Single Context DM",
}


def plot_acc(load_idx_names, tasks, accs, n_trials):
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams.update({"font.size": 28})
    # load_colors = (c_vals[5], c_vals[3])
    # load_colors = (c_vals[5], c_vals[3], c_vals_dl[8], "#ecc94b", c_vals[6], c_vals[2])

    _, ax1 = plt.subplots(1, 1, figsize=(24, 16))

    bar_width = 0.8 / len(load_idx_names)
    for load_idx, _ in enumerate(load_idx_names):
        ax1.bar(
            np.arange(len(tasks))
            + bar_width * (load_idx - len(load_idx_names) / 2 + 0.5),
            np.mean(accs[load_idx], axis=-1),
            bar_width,
            # color=load_colors[load_idx],
            label=load_idx_names[load_idx],
        )

    ax1.set_xticks(np.arange(len(tasks)))
    ax1.set_xticklabels(
        [task_names[task] for task in tasks], rotation=45, ha="right", fontsize=36
    )

    ax1.set_ylim((0.4, 1.2))
    ax1.set_yticks((0.5, 0.6, 0.7, 0.8, 0.9, 1.0))
    ax1.set_yticklabels((0.5, None, None, None, None, 1.0), fontsize=36)
    ax1.set_ylabel("Accuracy")
    ax1.legend(fontsize=36)

    for it in range(0, len(tasks), 2):
        ax1.axvline(it, color="grey", alpha=0.2, zorder=-1)

    # jetplot.breathe(ax=ax1)
    plt.tight_layout()
    savefig("./figures", "ngym_accs", "pdf")


"""
Plot the patterns in the FreeNet model for decision making task.
Parameters:
    - batch: input data
"""


def plot_norm_dm(net_type, db, batch, save_dir, save_name, task):
    assert task in ["ContextDecisionMaking-v0"]
    plt.clf()
    plt.rcParams["font.family"] = "DejaVu Sans"
    n_batch = 2
    if net_type == "FreeNet":
        M_hist = db["M"][:n_batch]  # shape: [B, T, Nh, Nx]
    elif net_type in ["HebbNet_M", "GRU"]:
        raise NotImplementedError("plot norm not implemented for HebbNet_M")
    cue_time = (
        np.nonzero(batch[1][:n_batch, :])[:, :2].cpu().numpy()
    )  # when label is non-zero, it is a cue time
    # cue_time = cue_time[0:1, :]
    patterns = (
        M_hist[cue_time[:, 0], cue_time[:, 1], :, :]
        .squeeze()
        .reshape(-1, M_hist.shape[-1])
    )
    # cluster = cue_time[:, 0]
    cluster = np.arange(cue_time.shape[0])
    cluster = np.repeat(cluster, M_hist.shape[2])
    pca = PCA(n_components=2)
    pca.fit(patterns)
    patterns_pca = pca.transform(patterns)
    patterns_pca = patterns_pca + np.random.normal(0, 0.01, patterns_pca.shape)
    # for i in range(n_batch):
    #     plt.scatter(
    #         patterns_pca[cluster == i, 0],
    #         patterns_pca[cluster == i, 1],
    #         s=10,
    #     )
    # breakpoint()
    plt.scatter(
        patterns_pca[:, 0],
        patterns_pca[:, 1],
        s=10,
        c=cluster.flatten(),
        cmap="Set1",
        alpha=0.5,
        marker=".",
    )
    # plt.xlim(-3, 3)
    # plt.ylim(-3, 3)
    savefig(save_dir, save_name, "png")


def plot_norm_readysetgo(net_type, db, batch, raw_input, save_dir, save_name):
    plt.clf()
    plt.rcParams["font.family"] = "DejaVu Sans"
    n_batch = 1
    raw_input = raw_input[:n_batch]
    if net_type == "FreeNet":
        M_hist = db["M"][:n_batch].cpu().numpy()  # shape: [B, T, Nhidden, Ninput]
    elif net_type in ["HebbNet_M", "GRU"]:
        raise NotImplementedError("plot norm not implemented for HebbNet_M")
    # breakpoint()
    # raw_input shape: [B, T, 3]
    ready_time = np.nonzero(raw_input[:n_batch, :, 1])[1]
    set_time = np.nonzero(raw_input[:n_batch, :, 2])[1]
    go_time = (
        (np.nonzero(batch[1][:n_batch, :, 0])[:, 1]).cpu().numpy()
    )  # when label is non-zero, it is a cue time
    t = np.arange(M_hist.shape[1])
    norms = np.linalg.norm(M_hist, axis=-1)
    # breakpoint()
    plt.plot(t, norms.mean(-1).flatten(), label="norm")
    plt.vlines(
        ready_time,
        *plt.ylim(),
        color="red",
        label="ready time",
    )
    plt.vlines(
        set_time,
        *plt.ylim(),
        color="blue",
        label="set time",
    )
    plt.vlines(
        go_time,
        *plt.ylim(),
        color="green",
        label="go time",
    )
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Norm")
    savefig(save_dir, save_name, "png")


def plot_pattern_gif(net_type, db, batch, save_dir, save_name):
    plt.rcParams["font.family"] = "DejaVu Sans"
    if net_type in ["GRU"]:
        return
    n_batch = 1
    if net_type == "FreeNet":
        M_hist = db["M"][:n_batch]  # shape: [B, T, Nh, Nx]
    elif net_type == "HebbNet_M":
        raise NotImplementedError("plot pattern gif not implemented for HebbNet_M")
    T = M_hist.shape[1]
    fig = plt.figure()

    def update(frame):
        plt.clf()
        patterns = M_hist[:, frame, :, :].squeeze().reshape(-1, M_hist.shape[-1])
        pca = PCA(n_components=2)
        pca.fit(patterns)
        patterns_pca = pca.transform(patterns)
        # patterns_pca = patterns_pca + np.random.normal(0, 0.05, patterns_pca.shape)
        plt.scatter(
            patterns_pca[:, 0],
            patterns_pca[:, 1],
        )

    ani = FuncAnimation(fig, update, frames=range(T), repeat=False, interval=100)
    create_dir(save_dir)
    ani.save(f"{save_dir}/{save_name}.gif", writer="imagemagick", fps=3)
    logging.info(f"Saved gif to {save_dir}/{save_name}.gif")
    # plt.scatter(patterns_pca[:, 0], patterns_pca[:, 1], c=cluster.flatten(), cmap="viridis")
    # savefig(save_dir, save_name, "png")
