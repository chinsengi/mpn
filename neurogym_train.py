import neurogym as ngym
import torch
import numpy as np
from utility import *
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
    save_root = '/content/drive/MyDrive/neuro_research/stp_networks/saved_nets/'
    # save_root = '/content/drive/MyDrive/neuro_research/stp_networks/saved_nets/anti_task_'

    net_params = {
        'netType': 'HebbNet_M', # HebbNet, HebbNet_M, VanillaRNN, GRU, rHebbNet, rHebbNet_M, GHU, GHU_M
        'n_inputs': ob_size,           # input dim
        'n_hidden': 100,                                # hidden dim
        'n_outputs': act_size,         # output dim
        'f_act': 'tanh',         # 1st layer actiivation function # linear, sigmoid, tanh, relu
        'f0_act': 'linear',          # 2nd layer actiivation function
        'trainable_state0': False,
        
        # STPN Features
        'A_act': None,            # Activation on the A update (tanh or None)
        'lam_clamp': 0.95,          # Maximum lambda value
        'eta_type': 'scalar',       # scalar or vector
        'eta_force': None,        # ensure either Hebbian or anti-Hebbian plasticity
        'hebb_type': 'inputOutput',      # input, output, inputOutput
        'modulation_bounds': False, # bound modulations 
        'mod_bound_val': 0.1, 

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

        if train:
            dataset_params = datasets_params[0]
            net, net_params = train_network_ngym(net_params, dataset_params, save=save, save_root=save_root)
        else:
            print('Not training (set train=True if you want to train).')

if __name__ == "__main__":
    main()