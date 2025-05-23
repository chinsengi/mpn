import copy
import logging
import os
import pickle
import time

import matplotlib.pyplot as plt
import neurogym as ngym
import numpy as np
import torch

import context_data as context
import int_data as syn
import networks as nets
from data import convert_serialized_mnist
from FreeNet import FreeNet
from net_utils import random_weight_init

NET_TYPES = [
    "nnLSTM",
    "GRU",
    "VanillaRNN",
    "HebbNet",
    "HebbNet_M",
    "rHebbNet",
    "rHebbNet_M",
    "GHU",
    "GHU_M",
    "FreeNet",
]


def default_params(net_params):
    # Applies default parameters. Note this was implemented only when writing
    # rebuttal, so if certain parameters are not in saved nets they are default

    if "modulation_bounds" not in net_params:
        net_params["modulation_bounds"] = False
    if "mod_bound_val" not in net_params:
        net_params["mod_bound_val"] = 1.0

    return net_params


def init_net(net_params, verbose=True, device="cpu"):
    # Set defaults
    net_params = default_params(net_params)

    # initialize net with default values
    if net_params["netType"] == NET_TYPES[0]:  # "nnLSTM"
        net = nets.nnLSTM(
            [net_params["n_inputs"], net_params["n_hidden"], net_params["n_outputs"]],
            batch_size=net_params["batch_size"],
            device=device,
        )
    elif net_params["netType"] == NET_TYPES[1]:  # "GRU"
        net = nets.GRU(
            [net_params["n_inputs"], net_params["n_hidden"], net_params["n_outputs"]],
            trainableState0=net_params["trainable_state0"],
            hiddenBias=net_params["hidden_bias"],
            roBias=net_params["ro_bias"],
            fAct=net_params["f_act"],
            fOutAct=net_params["f0_act"],
            freezeInputs=net_params["freeze_inputs"],
            sparsification=net_params["sparsification"],
            noiseType=net_params["noise_type"],
            noiseScale=net_params["noise_scale"],
            verbose=verbose,
        )
    elif net_params["netType"] == NET_TYPES[2]:  # "VanillaRNN"
        net = nets.VanillaRNN(
            [net_params["n_inputs"], net_params["n_hidden"], net_params["n_outputs"]],
            trainableState0=net_params["trainable_state0"],
            hiddenBias=net_params["hidden_bias"],
            roBias=net_params["ro_bias"],
            fAct=net_params["f_act"],
            fOutAct=net_params["f0_act"],
            freezeInputs=net_params["freeze_inputs"],
            sparsification=net_params["sparsification"],
            noiseType=net_params["noise_type"],
            noiseScale=net_params["noise_scale"],
            verbose=verbose,
        )
    elif net_params["netType"] in (
        NET_TYPES[3],  # "HebbNet"
        NET_TYPES[4],  # "HebbNet_M"
        NET_TYPES[5],  # "rHebbNet"
        NET_TYPES[6],  # "rHebbNet_M"
        NET_TYPES[7],  # "GHU"
        NET_TYPES[8],  # "GHU_M"
        NET_TYPES[9],  # "FreeNet"
    ):
        if net_params["netType"] in (
            NET_TYPES[4],
            NET_TYPES[6],
            NET_TYPES[8],
        ):  # "HebbNet_M", "rHebbNet_M", "GHU_M"
            stpType = "mult"
        elif net_params["netType"] in (
            NET_TYPES[3],
            NET_TYPES[5],
            NET_TYPES[7],
        ):  # "HebbNet", "rHebbNet", "GHU"
            stpType = "add"
        elif net_params["netType"] == NET_TYPES[9]:  # "FreeNet"
            stpType = "free"

        if net_params["netType"] in (
            NET_TYPES[3],
            NET_TYPES[4],
        ):  # "HebbNet", "HebbNet_M"
            netClass = nets.HebbNet
        elif net_params["netType"] in (
            NET_TYPES[5],
            NET_TYPES[6],
        ):  # "rHebbNet", "rHebbNet_M"
            netClass = nets.RecHebbNet
        elif net_params["netType"] in (NET_TYPES[7], NET_TYPES[8]):  # "GHU", "GHU_M"
            netClass = nets.GHU
        elif net_params["netType"] == NET_TYPES[9]:  # "FreeNet"
            netClass = FreeNet

        net = netClass(
            [net_params["n_inputs"], net_params["n_hidden"], net_params["n_outputs"]],
            trainableState0=net_params["trainable_state0"],
            stpType=stpType,
            hiddenBias=net_params["hidden_bias"],
            roBias=net_params["ro_bias"],
            updateType="hebb",
            lamClamp=net_params["lam_clamp"],
            lamType=net_params["lam_type"],
            layerBias=net_params["layer_bias"],
            hebbType=net_params["hebb_type"],
            mpType=net_params["mp_type"],
            etaType=net_params["eta_type"],
            freezeInputs=net_params["freeze_inputs"],
            sparsification=net_params["sparsification"],
            noiseType=net_params["noise_type"],
            noiseScale=net_params["noise_scale"],
            AAct=net_params["A_act"],
            modulation_bounds=net_params["modulation_bounds"],
            mod_bound_val=net_params["mod_bound_val"],
            fAct=net_params["f_act"],
            fOutAct=net_params["f0_act"],
            verbose=verbose,
            batch_size=net_params["batch_size"],
            outputLayer=net_params["output_layer"],
            winpRank=net_params.get("winp_rank", -1),
            device=device,
        )

        if net_params["eta_force"] == "Hebb":
            net.forceHebb = torch.tensor(True)
            net.init_hebb(
                eta=net.eta.item(), lam=net.lam.item()
            )  # need to re-init for this to work
        elif net_params["eta_force"] == "Anti":
            net.forceAnti = torch.tensor(True)
            net.init_hebb(eta=net.eta.item(), lam=net.lam.item())
        elif net_params["eta_force"] is not None:
            raise ValueError
    else:
        raise ValueError("Network type {} not recognized".format(net_params["netType"]))

    return net.to(device)


def save_net(
    net, net_params, dataset_details, root_path, overwrite=False, verbose=True
):
    """
    Saves the network as well as the dicts net_params and toy_params which
    will allow the network be reinitialized. Unfortunately couldn't figure
    out how to do this without saving to two different files.

    INPUTS:
    dataset_details: either toy_params or a dictionary of neuroGym parameters

    """

    save_file = True

    if "data_type" in dataset_details:  # toy_params case
        filename = os.path.join(
            root_path,
            "{}[{},{},{}]_train={}_task={}_{}class_{}len_seed{}.pkl".format(
                net_params["netType"],
                net_params["n_inputs"],
                net_params["n_hidden"],
                net_params["n_outputs"],
                net_params["train_mode"],
                dataset_details["data_type"],
                dataset_details["n_classes"],
                dataset_details["phrase_length"],
                net_params["seed"],
            ),
        )
    elif "dataset" in dataset_details:  # neruoGym case
        del dataset_details[
            "dataset"
        ]  # Sometimes cant pickle this, so just remove the dataset
        dataset_details = copy.deepcopy(dataset_details)
        filename = os.path.join(
            root_path,
            "{}[{},{},{}]_train={}_task={}_{}len_seed{}.pkl".format(
                net_params["netType"],
                net_params["n_inputs"],
                net_params["n_hidden"],
                net_params["n_outputs"],
                net_params["train_mode"],
                dataset_details["dataset_name"],
                dataset_details["seq_length"],
                net_params["seed"],
            ),
        )
    else:
        raise ValueError("dataset_details not recognized")

    directory = os.path.split(filename)[0]
    if directory != "" and not os.path.exists(directory):
        os.makedirs(directory)

    if os.path.exists(filename):
        print("  File already exists at:", filename)
        # override = input('Override? (Y/N):')
        override = "Y"
        if override.upper() == "N":
            save_file = False
        elif override.upper() != "Y":
            raise ValueError(f"Input {override} not recognized!")

    # if not overwrite:
    #     base, ext = os.path.splitext(filename)
    #     n = 2
    #     while os.path.exists(filename):
    #         filename = '{}_({}){}'.format(base, n, ext)
    #         n+=1

    state_filename = filename[: filename.find(".pkl")] + "_netstate.pkl"

    if save_file:
        if verbose:
            print("  Params filename:", filename)
        with open(filename, "wb") as save_file:
            # Saves additional parameters so the network can be re-initialized
            pickle.dump(net_params, save_file, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(dataset_details, save_file, protocol=pickle.HIGHEST_PROTOCOL)

        # Calls a reset state to set batch size to default (=1)
        # (doesn't affect anything, just prevents a warning when loading)
        if net_params["netType"] in (
            "HebbNet",
            "HebbNet_M",
        ):
            net.reset_state()

        if verbose:
            print("  State filename:", state_filename)
        with open(state_filename, "wb") as save_file:
            # Uses Pytorch's internal saving protocal for the network last
            net.save(save_file)
    else:
        print("  File not saved!")


def get_data_type_fns(toy_params, verbose=True):
    # Sets some dataset dependent parameters/functions
    special_data_gen_fn = None
    if toy_params["data_type"] == "int":
        if verbose:
            print("Toy integration data...")
        data_gen_fn = syn.generate_data
        special_data_gen_fn = syn.generate_special_data
    elif toy_params["data_type"] in ("context", "retro_context"):
        if verbose:
            print("Contextual integration data (continuous)...")
        assert toy_params["n_classes"] == 2  # Only defined for 2 outputs at the moment
        data_gen_fn = context.generate_data
        special_data_gen_fn = context.generate_special_data
    elif toy_params["data_type"] in (
        "context_int",
        "retro_context_int",
    ):
        if verbose:
            print("Contextual integration data (discrete)...")
        data_gen_fn = syn.generate_data_context
    elif toy_params["data_type"] in (
        "context_anti",
        "retro_context_anti",
    ):
        if verbose:
            print("Contextual anti-integration task...")
        data_gen_fn = syn.generate_data_anti_context
    elif toy_params["data_type"] in ("cont_int",):
        if verbose:
            print("Continuous integration task (no context)...")
        data_gen_fn = context.generate_cont_int_data
    elif toy_params["data_type"] in (
        "smnist_rows",
        "smnist_columns",
    ):
        if verbose:
            print("Serialized MNIST...")
        data_gen_fn = convert_serialized_mnist
    else:
        raise ValueError("Data_type {} not recognized".format(toy_params["data_type"]))

    run_special = False if special_data_gen_fn is None else True

    return data_gen_fn, special_data_gen_fn, run_special


def load_net(filename, seed, device="cpu"):
    """
    Loads the net_params and toy_params, and uses this to re-init a network.

    filename: should be the filename of the pkl file where the net_params
        and toy_params are located WITHOUT SEED OR .pkl
    """
    filename = filename + "_seed{}.pkl".format(seed)

    logging.info(f"Loading network from {filename}")
    if not os.path.exists(filename):
        raise ValueError("No file at path:", filename)
    else:
        state_filename = filename[: filename.find(".pkl")] + "_netstate.pkl"

        with open(filename, "rb") as load_file:
            net_params_load = pickle.load(load_file)
            dataset_details_load = pickle.load(
                load_file
            )  # toy_params or dataset_details

        # Using loaded net and toy parameters, creates new network
        net_load = init_net(net_params_load, verbose=True, device=device)
        with open(state_filename, "rb") as load_file:
            net_load.load(load_file)
        net_load.to(device)
        return net_load, net_params_load, dataset_details_load


def get_align_angle(x, y):
    dot = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    if dot > 1.0:
        dot = 1.0
    elif dot < -1.0:
        dot = -1.0

    return 180 / np.pi * np.arccos(dot)


def get_running_scores(word_labels, toy_params):
    """
    Returns the running score for a dataset

    OUTPUTS:
    running_score: (n_batch, seq_len, score_vec_size)
    """

    running_score = np.zeros(
        (
            word_labels.shape[0],
            word_labels.shape[1],
            toy_params["base_word_vals"]["evid0"].shape[0],
        )
    )

    for batch_idx in range(word_labels.shape[0]):
        batch_score = np.zeros((running_score.shape[-1],))
        for seq_idx in range(word_labels.shape[1]):
            current_word = toy_params["words"][word_labels[batch_idx, seq_idx]]
            if current_word in toy_params["base_word_vals"]:
                batch_score = batch_score + toy_params["base_word_vals"][current_word]
            running_score[batch_idx, seq_idx] = batch_score

    return running_score


def HebbNet_word_proj(word, As, net, toy_params):
    if toy_params["data_type"] == "int":
        word_input = toy_params["word_to_input_vector"][word]
    elif toy_params["data_type"] == "retro_context_int":
        assert word == "<eos>"
        word_input = np.concatenate(
            (
                toy_params["toy_params_1"]["word_to_input_vector"]["<eos>"],
                toy_params["toy_params_2"]["word_to_input_vector"]["<eos>"],
            ),
            axis=0,
        )
    else:
        assert word == "<eos>"
        word_input = np.array([0, 0, 0.5, 0.5])

    if net.stpType == "add":
        Aword = np.matmul(As, word_input)
        Winpword = np.matmul(net.w1.detach().numpy(), word_input)
        if len(As.shape) == 4:  # sequence and batch idx
            Winpword = Winpword[np.newaxis, np.newaxis, :]
        elif len(As.shape) == 3:  # only one idx
            Winpword = Winpword[np.newaxis, :]
        else:
            raise ValueError("As shape not implemented")

        a1 = Aword + Winpword + net.b1.detach().numpy()
        if not torch.is_tensor(a1):
            a1 = torch.tensor(a1)
        h_word = net.f(a1) if net.f is not None else a1

    elif net.stpType == "mult":
        if len(As.shape) == 4:  # sequence and batch idx
            w1 = net.w1.detach().cpu().numpy()[np.newaxis, np.newaxis, :, :]
        elif len(As.shape) == 3:  # only one idx
            w1 = net.w1.detach().cpu().numpy()[np.newaxis, :, :]
        else:
            raise ValueError("As shape not implemented")
        AWword = np.matmul(As * w1, word_input)
        h_word = (
            net.f(torch.tensor(AWword) + net.b1.detach().cpu())
            if net.f is not None
            else torch.tensor(AWword) + net.b1.detach()
        )

    return h_word.numpy()


def convert_ngym_dataset(
    dataset_params,
    set_size=None,
    device="cpu",
    mask_type=None,
    output_unconverted_data=False,
):
    """
    This converts a neuroGym dataset into one that the code can use.

    Mostly just transposes the batch and sequence dimensions, then combines
    them into a TensorDataset.

    Params:
    dataset_params: dictionary of parameters for the dataset
        dataset_name: name of the neuroGym dataset
        dt: time step size
        seq_length: length of the sequence
        convert_inputs: whether to convert inputs to a different dimension
        input_dim: dimension to convert inputs to
        convert_mat: conversion matrix (optional)
        convert_b: conversion bias (optional)
    set_size: size of the dataset (default is None, which means use the default size)
    device: device to use (default is "cpu")
    mask_type: type of mask to use (default is None, which means no mask)
        None: all True
        label: True when labels are nonzero
        no_fix: True when fixation is zero
    """

    dataset = ngym.Dataset(
        dataset_params["dataset_name"],
        env_kwargs={"dt": dataset_params["dt"]},
        batch_size=set_size,
        seq_len=dataset_params["seq_length"],
    )
    inputs, labels = dataset()

    # Default in our setup (batch, seq_idx, :) so need to swap dims
    inputs = np.transpose(inputs, axes=(1, 0, 2))
    labels = np.transpose(
        labels,
        axes=(
            1,
            0,
        ),
    )[:, :, np.newaxis]

    act_size = dataset.env.action_space.n
    if (
        mask_type is None
    ):  # Mask is always just all time steps, so creates all True array
        masks = np.ones((inputs.shape[0], inputs.shape[1], act_size))
    elif mask_type == "label":  # Masks on when labels are nonzero
        masks_flat = (labels > 0.0).astype(np.int32)  # (B, seq_len)
        masks = np.repeat(masks_flat, act_size, axis=-1)  # (B, seq_len, act_size)
    elif (
        mask_type == "no_fix"
    ):  # Masks on when fixation is zero, assumes fixation is zeroth input
        masks_flat = (inputs[:, :, 0:1] == 0.0).astype(np.int32)  # (B, seq_len)
        if np.sum(masks_flat) == 0:
            raise ValueError("Mask is all zeros!")
        masks = np.repeat(masks_flat, act_size, axis=-1)  # (B, seq_len, act_size)
    else:
        raise ValueError("mask type {} not recoginized".format(mask_type))

    unconverted_inputs = None
    if output_unconverted_data:
        unconverted_inputs = np.array(inputs)
    if dataset_params["convert_inputs"]:
        # If the conversion is not already generated, create it
        if "convert_mat" not in dataset_params:
            W, b = random_weight_init(
                [inputs.shape[-1], dataset_params["input_dim"]], bias=True
            )
            dataset_params["convert_mat"] = W[0]
            dataset_params["convert_b"] = b[0][np.newaxis, np.newaxis, :]

        inputs = (
            np.matmul(inputs, dataset_params["convert_mat"].T)
            + dataset_params["convert_b"]
        )
    inputs = (
        torch.from_numpy(inputs).type(torch.float).to(device)
    )  # inputs.shape (16, 100, 3)
    labels = (
        torch.from_numpy(labels).type(torch.long).to(device)
    )  # labels.shape = (16, 100, 1)
    masks = torch.tensor(masks, dtype=torch.bool, device=device)

    trainData = torch.utils.data.TensorDataset(inputs, labels)

    if output_unconverted_data:
        return trainData, masks, dataset_params, unconverted_inputs
    return trainData, masks, dataset_params


def use_gpu(gpu_id: int = 0):
    num_of_gpus = torch.cuda.device_count()
    if num_of_gpus > 0:
        assert gpu_id < num_of_gpus
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    return device


def setup_logger(save_dir, filename="stdout.txt", level=logging.INFO):
    create_dir(save_dir)
    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(os.path.join(save_dir, filename))
    formatter = logging.Formatter(
        "%(levelname)s - %(filename)s:%(lineno)d - %(asctime)s - %(message)s"
    )
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    logger.setLevel(level)


def create_dir(path="./model"):
    isExist = os.path.exists(path)
    if not isExist:
        print(f"Creating directory: {path}")
        os.makedirs(path)


def savefig(path="./image", filename="image", format="png", include_timestamp=True):
    create_dir(path)
    if include_timestamp:
        t = time.localtime()
        current_time = time.strftime("%H_%M_%S", t)
    else:
        current_time = ""
    plt.savefig(
        os.path.join(path, current_time + filename + "." + format),
        dpi=300,
        format=format,
    )
    logging.info(
        f"Saving figure to {os.path.join(path, current_time + filename + '.' + format)}"
    )
