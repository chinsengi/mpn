from net_utils import xe_classifier_accuracy
from paper.data import convert_serialized_mnist
import os
import pickle
import copy
import numpy as np
import networks as nets
import torch
import int_data as syn
import context_data as context
from net_utils import random_weight_init
from FreeNet import FreeNet


def default_params(net_params):
    # Applies default parameters. Note this was implemented only when writing
    # rebuttal, so if certain parameters are not in saved nets they are default

    if 'modulation_bounds' not in net_params: net_params['modulation_bounds'] = False
    if 'mod_bound_val' not in net_params: net_params['mod_bound_val'] = 1.0

    return net_params

def init_net(net_params, verbose=True):
    
    # Set defaults
    net_params = default_params(net_params)

    # initialize net with default values
    if net_params['netType'] == 'nnLSTM':
        net = nets.nnLSTM([net_params['n_inputs'], net_params['n_hidden'], net_params['n_outputs']])
    elif net_params['netType'] == 'GRU':
        net = nets.GRU([net_params['n_inputs'], net_params['n_hidden'], net_params['n_outputs']],
                       trainableState0=net_params['trainable_state0'], 
                       hiddenBias=net_params['hidden_bias'], roBias=net_params['ro_bias'], 
                       fAct=net_params['f_act'], fOutAct=net_params['f0_act'],
                       freezeInputs=net_params['freeze_inputs'],
                       sparsification=net_params['sparsification'], noiseType=net_params['noise_type'],
                       noiseScale=net_params['noise_scale'], verbose=verbose)
    elif net_params['netType'] == 'VanillaRNN':
        net = nets.VanillaRNN([net_params['n_inputs'], net_params['n_hidden'], net_params['n_outputs']],
                              trainableState0=net_params['trainable_state0'], 
                              hiddenBias=net_params['hidden_bias'], roBias=net_params['ro_bias'], 
                              fAct=net_params['f_act'], fOutAct=net_params['f0_act'],
                              freezeInputs=net_params['freeze_inputs'],
                              sparsification=net_params['sparsification'], noiseType=net_params['noise_type'],
                              noiseScale=net_params['noise_scale'], verbose=verbose)
    elif net_params['netType'] in ('HebbNet', 'HebbNet_M', 'rHebbNet', 'rHebbNet_M', 'GHU', 'GHU_M', 'FreeNet'):
        if net_params['netType'] in ('HebbNet_M', 'rHebbNet_M', 'GHU_M'):
            stpType = 'mult'
        elif net_params['netType'] in ('HebbNet', 'rHebbNet', 'GHU'):
            stpType = 'add'
        elif net_params['netType'] in ('FreeNet'):
            stpType = 'free'
        if net_params['netType'] in ('HebbNet', 'HebbNet_M'):
            netClass = nets.HebbNet
        elif net_params['netType'] in ('rHebbNet', 'rHebbNet_M'):
            netClass = nets.RecHebbNet
        elif net_params['netType'] in ('GHU', 'GHU_M'):
            netClass = nets.GHU
        elif net_params['netType'] in ('FreeNet'):
            netClass = FreeNet

        net = netClass([net_params['n_inputs'], net_params['n_hidden'], net_params['n_outputs']],
                       trainableState0=net_params['trainable_state0'], stpType=stpType, 
                       hiddenBias=net_params['hidden_bias'], roBias=net_params['ro_bias'], updateType='hebb', 
                       lamClamp=net_params['lam_clamp'], hebbType=net_params['hebb_type'],
                       etaType=net_params['eta_type'], freezeInputs=net_params['freeze_inputs'], 
                       sparsification=net_params['sparsification'], noiseType=net_params['noise_type'],
                       noiseScale=net_params['noise_scale'], AAct=net_params['A_act'], 
                       modulation_bounds=net_params['modulation_bounds'], mod_bound_val=net_params['mod_bound_val'],
                       fAct=net_params['f_act'], fOutAct=net_params['f0_act'], verbose=verbose)

        if net_params['eta_force'] == 'Hebb':
            net.forceHebb = torch.tensor(True)
            net.init_hebb(eta=net.eta.item(), lam=net.lam.item()) #need to re-init for this to work
        elif net_params['eta_force'] == 'Anti':
            net.forceAnti = torch.tensor(True)
            net.init_hebb(eta=net.eta.item(), lam=net.lam.item())
        elif net_params['eta_force'] is not None:
            raise ValueError
    else:
        raise ValueError

    return net

def save_net(net, net_params, dataset_details, root_path, overwrite=False, verbose=True):
    """ 
    Saves the network as well as the dicts net_params and toy_params which 
    will allow the network be reinitialized. Unfortunately couldn't figure 
    out how to do this without saving to two different files.

    INPUTS:
    dataset_details: either toy_params or a dictionary of neuroGym parameters

    """

    save_file = True

    if 'data_type' in dataset_details: # toy_params case
        filename = root_path + '{}[{},{},{}]_train={}_task={}_{}class_{}len_seed{}.pkl'.format(
            net_params['netType'], net_params['n_inputs'], net_params['n_hidden'],
            net_params['n_outputs'], net_params['train_mode'],
            dataset_details['data_type'], dataset_details['n_classes'], 
            dataset_details['phrase_length'], net_params['seed'],
        )
    elif 'dataset' in dataset_details: # neruoGym case
        dataset_details = copy.deepcopy(dataset_details)
        del dataset_details['dataset'] # Sometimes cant pickle this, so just remove the dataset
        filename = root_path + '{}[{},{},{}]_train={}_task={}_{}len_seed{}.pkl'.format(
            net_params['netType'], net_params['n_inputs'], net_params['n_hidden'],
            net_params['n_outputs'], net_params['train_mode'],
            dataset_details['dataset_name'], 
            dataset_details['seq_length'], net_params['seed'],
        )
    else:
        raise ValueError('dataset_details not recognized')

    directory = os.path.split(filename)[0]
    if directory != '' and not os.path.exists(directory):
        os.makedirs(directory)

    if os.path.exists(filename):
        print('  File already exists at:', filename)
        override = input('Override? (Y/N):')
        if override == 'N':
            save_file = False
        elif override != 'Y':
            raise ValueError(f'Input {override} not recognized!')

    # if not overwrite:
    #     base, ext = os.path.splitext(filename)
    #     n = 2
    #     while os.path.exists(filename):
    #         filename = '{}_({}){}'.format(base, n, ext)
    #         n+=1

    state_filename = filename[:filename.find('.pkl')] + '_netstate.pkl'

    if save_file:
        if verbose:
            print('  Params filename:', filename)
        with open(filename, 'wb') as save_file:
            # Saves additional parameters so the network can be re-initialized 
            pickle.dump(net_params, save_file, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(dataset_details, save_file, protocol=pickle.HIGHEST_PROTOCOL)

        # Calls a reset state to set batch size to default (=1)
        # (doesn't affect anything, just prevents a warning when loading)
        if net_params['netType'] in ('HebbNet', 'HebbNet_M',):
            net.reset_state()

        if verbose:
            print('  State filename:', state_filename)
        with open(state_filename, 'wb') as save_file:    
            # Uses Pytorch's internal saving protocal for the network last
            net.save(save_file)
    else:
        print('  File not saved!')

def get_data_type_fns(toy_params, verbose=True):
     # Sets some dataset dependent parameters/functions
    special_data_gen_fn = None
    if toy_params['data_type'] == 'int':
        if verbose: print('Toy integration data...')
        data_gen_fn = syn.generate_data
        special_data_gen_fn = syn.generate_special_data
    elif toy_params['data_type'] in ('context', 'retro_context'):
        if verbose: print('Contextual integration data (continuous)...')
        assert toy_params['n_classes'] == 2 # Only defined for 2 outputs at the moment
        data_gen_fn = context.generate_data
        special_data_gen_fn = context.generate_special_data
    elif toy_params['data_type'] in ('context_int', 'retro_context_int',):
        if verbose: print('Contextual integration data (discrete)...')
        data_gen_fn = syn.generate_data_context
    elif toy_params['data_type'] in ('context_anti', 'retro_context_anti',):
        if verbose: print('Contextual anti-integration task...')
        data_gen_fn = syn.generate_data_anti_context
    elif toy_params['data_type'] in ('cont_int',):
        if verbose: print('Continuous integration task (no context)...')
        data_gen_fn = context.generate_cont_int_data
    elif toy_params['data_type'] in ('smnist_rows', 'smnist_columns',):
        if verbose: print('Serialized MNIST...')
        data_gen_fn = convert_serialized_mnist
    else:
        raise ValueError('Data_type {} not recognized'.format(toy_params['data_type']))

    run_special = False if special_data_gen_fn is None else True
    
    return data_gen_fn, special_data_gen_fn, run_special

def load_net(filename, seed):
    """
    Loads the net_params and toy_params, and uses this to re-init a network.

    filename: should be the filename of the pkl file where the net_params
        and toy_params are located WITHOUT SEED OR .pkl
    """
    filename = filename + '_seed{}.pkl'.format(seed)

    if not os.path.exists(filename):
        raise ValueError('No file at path:', filename)
    else:
        state_filename = filename[:filename.find('.pkl')] + '_netstate.pkl'

        with open(filename, 'rb') as load_file:
            net_params_load = pickle.load(load_file)
            dataset_details_load = pickle.load(load_file) # toy_params or dataset_details
        
        # Using loaded net and toy parameters, creates new network
        net_load = init_net(net_params_load, verbose=False)

        with open(state_filename, 'rb') as load_file:
            net_load.load(load_file)

        return net_load, net_params_load, dataset_details_load

def get_align_angle(x, y):
    dot = np.dot(x,y)/(
         np.linalg.norm(x) * np.linalg.norm(y)
     )
    if dot > 1.0:
         dot = 1.0
    elif dot < -1.0:
        dot = -1.0
    
    return 180/np.pi * np.arccos(dot)

def get_running_scores(word_labels, toy_params):
    """
    Returns the running score for a dataset 

    OUTPUTS:
    running_score: (n_batch, seq_len, score_vec_size)
    """

    running_score = np.zeros((word_labels.shape[0],
                              word_labels.shape[1],
                              toy_params['base_word_vals']['evid0'].shape[0]))

    for batch_idx in range(word_labels.shape[0]):
        batch_score = np.zeros((running_score.shape[-1],))
        for seq_idx in range(word_labels.shape[1]):
            current_word = toy_params['words'][word_labels[batch_idx, seq_idx]]
            if current_word in toy_params['base_word_vals']:
                batch_score = batch_score + toy_params['base_word_vals'][current_word]
            running_score[batch_idx, seq_idx] = batch_score

    return running_score

def HebbNet_word_proj(word, As, net, toy_params):
    if toy_params['data_type'] == 'int':
        word_input = toy_params['word_to_input_vector'][word]
    elif toy_params['data_type'] == 'retro_context_int':
        assert word == '<eos>'
        word_input = np.concatenate((
            toy_params['toy_params_1']['word_to_input_vector']['<eos>'],
            toy_params['toy_params_2']['word_to_input_vector']['<eos>']
        ), axis=0)
    else:
        assert word == '<eos>'
        word_input = np.array([0, 0, 0.5, 0.5])

    if net.stpType == 'add':
        Aword = np.matmul(As, word_input)
        Winpword = np.matmul(net.w1.detach().numpy(), word_input)
        if len(As.shape) == 4: # sequence and batch idx
            Winpword = Winpword[np.newaxis, np.newaxis, :]
        elif len(As.shape) == 3: # only one idx
            Winpword = Winpword[np.newaxis, :]
        else:
            raise ValueError('As shape not implemented')
        
        a1 = Aword + Winpword + net.b1.detach().numpy()
        if not torch.is_tensor(a1):
            a1 = torch.tensor(a1)
        h_word = net.f(a1) if net.f is not None else a1

    elif net.stpType == 'mult':
        if len(As.shape) == 4: # sequence and batch idx
            w1 = net.w1.detach().cpu().numpy()[np.newaxis, np.newaxis, :, :]
        elif len(As.shape) == 3: # only one idx
            w1 = net.w1.detach().cpu().numpy()[np.newaxis, :, :]
        else:
            raise ValueError('As shape not implemented')
        AWword = np.matmul(As*w1, word_input)
        h_word = net.f(torch.tensor(AWword) + net.b1.detach().cpu()) if net.f is not None else torch.tensor(AWword) + net.b1.detach()

    return h_word.numpy()

def convert_ngym_dataset(dataset_params, set_size=None, device='cpu'):
    """
    This converts a neroGym dataset into one that the code can use. 

    Mostly just transposes the batch and sequence dimensions, then combines
    them into a TensorDataset. Also creates a mask of all trues.
    """

    dataset = dataset_params['dataset']

    if set_size is not None:
        dataset.batchsize = set_size # Just create as a single large batch for now

    inputs, labels = dataset()

    # Default in our setup (batch, seq_idx, :) so need to swap dims
    inputs = np.transpose(inputs, axes=(1, 0, 2))
    labels = np.transpose(labels, axes=(1, 0,))[:, :, np.newaxis]

    if dataset_params['convert_inputs']:
        # If the conversion is not already generated, create it
        if 'convert_mat' not in dataset_params:
            W,b = random_weight_init([inputs.shape[-1], dataset_params['input_dim']], bias=True)
            dataset_params['convert_mat'] = W[0]
            dataset_params['convert_b'] = b[0][np.newaxis, np.newaxis, :]

        inputs = np.matmul(inputs, dataset_params['convert_mat'].T) + dataset_params['convert_b']
    
    # Mask is always just all time steps, so creates all True array
    act_size = dataset_params["dataset"].env.action_space.n
    masks = np.ones((inputs.shape[0], inputs.shape[1], act_size))

    inputs = torch.from_numpy(inputs).type(torch.float).to(device) # inputs.shape (16, 100, 3)
    labels = torch.from_numpy(labels).type(torch.long).to(device) # labels.shape = (16, 100, 1)
    masks = torch.tensor(masks, dtype=torch.bool, device=device)

    trainData = torch.utils.data.TensorDataset(inputs, labels)

    return trainData, masks, dataset_params