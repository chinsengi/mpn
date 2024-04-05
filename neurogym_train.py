import neurogym as ngym
import torch
import numpy as np
from utility import *

def train_network_ngym(net_params, dataset_params, current_net=None, save=False, save_root='', 
                        set_seed=True, verbose=True):
        """ 
        Code to train a single network using a neuroGym dataset
        
        OUTPUTS:
        net: the trained network

        """

        # Sets the random seed for reproducibility (this affects both data generation and network)
        if 'seed' in net_params and set_seed:
            if net_params['seed'] is not None: 
                np.random.seed(seed=net_params['seed'])
                torch.manual_seed(net_params['seed'])
        
        # Intitializes network and puts it on device
        if net_params['cuda']:
            if verbose: print('Using CUDA...')
            device = torch.device('cuda')
        else:
            if verbose: print('Using CPU...')
            device = torch.device('cpu')
        if current_net is None: # Creates a new network
            net = init_net(net_params, verbose=verbose)
            # print('w1:', net.w1.detach().numpy()[:2,:2])
            # print('b1:', net.b1.detach().numpy()[:2])
            # print('wr:', net.wr.detach().numpy()[:2,:2])
            # print('wri:', net.wri.detach().numpy()[:2,:2])
            net.to(device)
        else:
            net = current_net

        # Creats the data and trains the network, this is done different for different training setups
        if net_params['train_mode'] == 'seq_inf': # Continually generates new data to train on
            # This will iterate in loop so that it only sees each type of data a set amount of times
            net_params['epochs'] = 0 if current_net is None else net.hist['epoch']
            
            validData, validOutputMask, dataset_params = convert_ngym_dataset(
                dataset_params, set_size=net_params['valid_set_size'], device=device
            )
            
            early_stop = False
            new_thresh = True # Triggers threshold setting for first call of .fit, but turns off after first call

            while not early_stop:
                net_params['epochs'] += 10 # Number of times it sees each example
                trainData, trainOutputMask, dataset_params = convert_ngym_dataset(
                    dataset_params, set_size=net_params['train_set_size'], device=device
                )

                breakpoint()
                early_stop = net.fit('sequence', epochs=net_params['epochs'], 
                                    trainData=trainData, batchSize=net_params['batch_size'],
                                    validBatch=validData[:,:,:], learningRate=1e-3,
                                    newThresh=new_thresh, monitorFreq=50, 
                                    trainOutputMask=trainOutputMask, validOutputMask=validOutputMask,
                                    validStopThres=net_params['accEarlyStop'], weightReg=net_params['weight_reg'], 
                                    regLambda=net_params['reg_lambda'], gradientClip=net_params['gradient_clip'],
                                    earlyStopValid=net_params['validEarlyStop'], minMaxIter=net_params['minMaxIter']) 
                new_thresh = False   
        else:
            raise ValueError

        if save:
            save_net(net, net_params, dataset_params, save_root, overwrite=False, verbose=verbose)
        else:
            print('Not saving network.')

        return net, net_params
    
def main():
    # All supervised tasks:
    tasks = (
        # 'ContextDecisionMaking-v0', 
        # 'DelayComparison-v0', 
        # 'DelayMatchCategory-v0',
        # 'DelayMatchSample-v0',
        # 'DelayMatchSampleDistractor1D-v0',
        # 'DelayPairedAssociation-v0',
        # 'DualDelayMatchSample-v0',
        # 'GoNogo-v0',
        # 'HierarchicalReasoning-v0',
        # 'IntervalDiscrimination-v0',
        # 'MotorTiming-v0',
        # 'MultiSensoryIntegration-v0',
        # 'OneTwoThreeGo-v0',
        # 'PerceptualDecisionMaking-v0',
        # 'PerceptualDecisionMakingDelayResponse-v0',
        'ProbabilisticReasoning-v0',
        # 'PulseDecisionMaking-v0',
        # 'ReachingDelayResponse-v0', # Different input type, so omitted
        # 'ReadySetGo-v0',
        # 'SingleContextDecisionMaking-v0',
    )

    kwargs = {'dt': 100}
    seq_len = 100

    datasets_params = []

    for task_idx, task in enumerate(tasks):

        # Make supervised dataset
        dataset = ngym.Dataset(task, env_kwargs=kwargs, batch_size=16,
                            seq_len=seq_len)
        
        dataset_params = {
            'dataset_name': task,
            'seq_length': seq_len,
            'dt': kwargs['dt'],
            'dataset': dataset,
            'convert_inputs': True, # Remap inputs through a random matrix + bias
            'input_dim': 10,
        }

        env = dataset.env
        ob_size = env.observation_space.shape[0]
        act_size = env.action_space.n

        print('Task {}: {}'.format(task_idx, task))
        print('  Observation size: {}, Action size: {}'.format(ob_size, act_size))

        datasets_params.append(dataset_params)

    train = True
    save = True
    save_root = './saved_nets/'

    net_params = {
        'netType': 'HebbNet', # HebbNet, HebbNet_M, VanillaRNN, GRU, rHebbNet, rHebbNet_M, GHU, GHU_M
        'n_inputs': ob_size,           # input dim
        'n_hidden': 100,                                # hidden dim
        'n_outputs': act_size,         # output dim
        'f_act': 'tanh',         # 1st layer actiivation function # linear, sigmoid, tanh, relu
        'f0_act': 'linear',          # 2nd layer actiivation function
        'trainable_state0': False,
        
        # STPN Features
        'A_act': None,            # Activation on the A update (tanh or None)
        'lam_type': 'matrix',
        'lam_clamp': 0.95,          # Maximum lambda value
        'eta_type': 'matrix',       # scalar or vector
        'eta_force': None,        # ensure either Hebbian or anti-Hebbian plasticity
        'hebb_type': 'inputOutput',      # input, output, inputOutput
        'modulation_bounds': False, # bound modulations 
        'mod_bound_val': 0.1, 
        "trainable_state0": True, # Train the initial weights
        'mp_type': 'free',

        # Train parameters
        'train_mode': 'seq_inf',   # 'seq' or 'seq_inf'
        'weight_reg': 'L1',
        'reg_lambda': 1e-4,
        'hidden_bias': True,
        'ro_bias': False,           # use readout bias or not
        'gradient_clip': 10,
        'freeze_inputs': False,      # freeze the input layer (and hidden bias)
        'sparsification': 0.0,      # Amount to sparsify network's weights (0.0 = no sparsification)
        'noise_type': 'input',      # None, 'input', 'hidden'
        'noise_scale': 0.1,         # Expected magnitude of noise vector
        'cuda': True,
        'validEarlyStop': True,     # Early stop when average validation loss saturates
        'accEarlyStop': None,       # Accuracy to stop early at (None to ignore)
        'minMaxIter': (100, 200), #(2000, 10000),  # Bounds on training time
        'seed': 1003,               # This seed is used to generate training/valid data too

        'train_set_size': 3200,
        'valid_set_size': 250,
        'batch_size': 32,
        'epochs': 40,
    }

    if dataset_params['convert_inputs']:
        net_params['n_inputs'] = dataset_params['input_dim']

    if train:
        dataset_params = datasets_params[0]
        _, net_params = train_network_ngym(net_params, dataset_params, save=save, save_root=save_root)
    else:
        print('Not training (set train=True if you want to train).')

    # Evaluate!
    load_root_start = './saved_nets/ngym_'

    # n_trials = 5 # For main text figure
    n_trials = 10 # For eta > 0 and eta < 0 figure

    load_types = [
        'convert10_2factor2_HebbNet_M[10,100,{}]_train=seq_inf_task={}_{}len', # note extra 2 here, original did not have acc cap
        'convert10_1factorsmall_HebbNet_M[10,100,{}]_train=seq_inf_task={}_{}len',
        # 'convert10_1factor_GRU[10,100,{}]_train=seq_inf_task={}_{}len',
        # 'convert10_1factor_VanillaRNN[10,100,{}]_train=seq_inf_task={}_{}len',
    ]

    init_seed = 1000
    test_set_size = 250

    accs = np.zeros((len(load_types), len(tasks), n_trials,))
    accs_pos_eta = [[[] for _ in range(len(tasks))] for _ in range(len(load_types))]
    accs_neg_eta = [[[] for _ in range(len(tasks))] for _ in range(len(load_types))]

    for task_idx, task in enumerate(tasks):

        # Uses the dataset_params array created above just to specify loading name
        dataset_params = datasets_params[task_idx]
        print('Task {}: {}'.format(task_idx, dataset_params['dataset_name']))
        
        # Have to recreate the dataset in dataset_params_load since its not saved
        kwargs = {'dt': dataset_params['dt']}
        dataset = ngym.Dataset(
            dataset_params['dataset_name'], env_kwargs=kwargs, 
            batch_size=16, seq_len=dataset_params['seq_length'])

        env = dataset.env
        ob_size = env.observation_space.shape[0]
        act_size = env.action_space.n

        for load_idx, load_type in enumerate(load_types):
            
            for trial_idx in range(n_trials):
                load_path = load_root_start + load_type.format(
                    act_size, dataset_params['dataset_name'], dataset_params['seq_length']
                )

                net_load, net_params_load, dataset_params_load = load_net(load_path, init_seed+trial_idx)
                # Have to recreate the dataset in dataset_params_load since its not saved
                kwargs = {'dt': dataset_params_load['dt']}
                dataset_params_load['dataset'] = ngym.Dataset(
                    dataset_params_load['dataset_name'], env_kwargs=kwargs, 
                    batch_size=16, seq_len=dataset_params_load['seq_length'])

                if 'convert_inputs' not in dataset_params_load:
                    dataset_params_load['convert_inputs'] = False

                testData, testOutputMask, _ = convert_ngym_dataset(
                    dataset_params_load, set_size=net_params_load['valid_set_size'], device=torch.device('cpu')
                )
                
                db_load = net_load.evaluate_debug(testData[:,:,:], batchMask=testOutputMask)
                accs[load_idx, task_idx, trial_idx] = db_load['acc']

                # Special eta signed accuracies
                if load_idx in (0, 1,): # MPN or MPNpre
                    eta = net_load.eta.detach().cpu().numpy()[0, 0, 0]
                    if eta > 0:
                        accs_pos_eta[load_idx][task_idx].append(db_load['acc'])
                    else:
                        accs_neg_eta[load_idx][task_idx].append(db_load['acc'])
            
            print('  Acc: {:.3f}'.format(np.mean(accs[load_idx, task_idx, :], axis=-1)))
            if load_idx in (0, 1,):
                print('    Pos eta: {}, Neg eta: {}'.format(
                    len(accs_pos_eta[load_idx][task_idx]), len(accs_neg_eta[load_idx][task_idx]))
                )


if __name__ == "__main__":
    main()