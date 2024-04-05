import itertools
import torch
from torch import nn
from torch.utils.data import TensorDataset
import torch.nn.functional as F
import numpy as np
import scipy.special as sps

from net_utils import StatefulBase, check_dims, random_weight_init, binary_classifier_accuracy, xe_classifier_accuracy

class FreeLayer(nn.Module):  
    """
    Layer with multiplasticity

    """
    def __init__(self, init, verbose=True, **mpnArgs):
        super().__init__()   

        if all([type(x)==int for x in init]) and len(init) == 2:
            Nx, Ny = init
            W, b = random_weight_init([Nx,Ny], bias=True)
            
            self.n_inputs = Nx
            self.n_outputs = Ny 
        else:
            raise ValueError('Input format not recognized')

        self.verbose = verbose
        init_string = 'MP Layer parameters:'

        self.layerAct = mpnArgs.get('layerAct', 'sigmoid') # layer activation
        if self.layerAct  == 'linear':
            self.f = None
        elif self.layerAct  == 'sigmoid':
            self.f = torch.sigmoid
        elif self.layerAct  == 'tanh':
            self.f = torch.tanh
        elif self.layerAct  == 'relu':
            self.f = torch.relu
        else:
            raise ValueError('f activaiton not recognized')

        self.mpType = mpnArgs.get('mpType', 'add')
        if self.mpType == 'add':
            init_string += '\n  MP Type: Additive //'
        elif self.mpType == 'mult':
            init_string += '\n  MP Type: Multiplicative //'
        else:
            raise ValueError('mpnType not recognized')

        init_string += ' Activation: {} // (Nx, Ny) = ({}, {})'.format(
            self.layerAct, self.n_inputs, self.n_outputs)

        # Forces all weights belonging to input neurons to be the same sign (80/20 split for +/-)
        self.useCellTypes = mpnArgs.get('useCellTypes', False)
        if self.useCellTypes:
            # Generates random vector of 1s and -1s with expected 80/20 split, rep Exc/Inh cells
            # note this starts as a numpy array b/c W[0] is a numpy array, but after first use converted to tensor
            cellTypes_np = 2*(np.random.rand(1, self.n_inputs) > 0.2)-1 # dim (1, Nx)
            # Adjusts initial W to match allowed signs
            W[0] = np.abs(W[0]) * cellTypes_np
            self.register_buffer('cellTypes', torch.tensor(cellTypes_np, dtype=torch.int32))
            init_string += '\n  W: Exc-Inh // '
        else:
            init_string += '\n  '

        # Determines whether or not layer weights are trainable parameters
        self.freezeLayer = mpnArgs.get('freezeLayer', False)
        if self.freezeLayer: # Does not train input layer 
            init_string += 'W: Frozen // '
            self.register_buffer('w1', torch.tensor(W[0], dtype=torch.float))
        else:
            self.w1 = nn.Parameter(torch.tensor(W[0], dtype=torch.float))
        
        # Determines if layer bias is present and trainable (can overwhelm noise during delay).
        self.layerBias = mpnArgs.get('layerBias', True)
        if self.layerBias and not self.freezeLayer:
            init_string += 'Layer bias: trainable // '
            self.b1 = nn.Parameter(torch.tensor(b[0], dtype=torch.float))
        elif self.layerBias and self.freezeLayer:
            init_string += 'Layer bias: frozen // '
            self.b1 = nn.Parameter(torch.tensor(b[0], dtype=torch.float))
        else: # No hidden bias
            init_string += 'No layer bias // '
            self.register_buffer('b1', torch.zeros_like(torch.tensor(b[0], dtype=torch.float)))

        # Sparisfies the layer by creating masks for weights
        self.sparsification = mpnArgs.get('sparsification', 0.0)
        if self.sparsification > 0:
            self.register_buffer('w1Mask', torch.bernoulli((1-self.sparsification)*torch.ones_like(self.w1)))
        init_string += 'Sparsification: {:.2f} // '.format(self.sparsification)

        # Injects noise into the layer
        self.noiseType = mpnArgs.get('noiseType', None)
        self.noiseScale = mpnArgs.get('noiseScale', 0.0)
        if self.noiseType in ('layer',):
            init_string += 'Noise type: {} (scl: {:.1e})'.format(self.noiseType, self.noiseScale)
        else:
            init_string += 'Layer Noise: None'

        ###### SM matrix-related specs ########
        init_string += '\n  SM matrix parameters:'

        # Change the type of update
        self.updateType = mpnArgs.get('updateType', 'hebb')
        if self.updateType == 'oja':
            init_string += '\n    M update: Oja-like // '
        elif self.updateType == 'hebb_norm':
            init_string += '\n    M update: Normalized // '
        elif self.updateType == 'hebb':
            init_string += '\n    M update: Hebbian // '
        else:
            raise ValueError('updateType: {} not recognized'.format(self.updateType))

        # Change the type of hebbian update
        self.hebbType = mpnArgs.get('hebbType', 'inputOutput')
        if self.hebbType == 'input':
            init_string += 'Pre-Syn Only // '
        elif self.hebbType == 'output':
            init_string += 'Post-Syn Only // '
        elif self.hebbType != 'inputOutput':
            raise ValueError('hebbType: {} not recognized'.format(self.hebbType))

        # Activation for the M update
        self.MAct = mpnArgs.get('MAct', None)
        if self.MAct == 'tanh':
            init_string += 'M Act: tanh // '
        elif self.MAct is None:
            init_string += 'M Act: linear // '
        else:
            raise ValueError('M activation not recognized')

        # Initial SM matrix of the network
        self.trainableState0 = mpnArgs.get('trainableState0', False)
        W2, _ = random_weight_init([self.n_inputs, self.n_outputs], bias=False)        
        if self.trainableState0:
            init_string += 'M0: trainable'
            self.M0 = nn.Parameter(torch.tensor(W2[0], dtype=torch.float))
        else:
            init_string += 'M0: zeros'            
            self.register_buffer('M0', torch.zeros_like(torch.tensor(W2[0], dtype=torch.float)))

        # Change the type of eta parameter
        self.etaType = mpnArgs.get('etaType', 'scalar')
        if self.etaType in ('scalar', 'pre_vector', 'post_vector', 'matrix'):
            init_string += '\n    Eta: {} // '.format(self.etaType)
        else:
            raise ValueError('etaType: {} not recognized'.format(self.etaType))

        # Change the type of lambda parameter
        self.lamType = mpnArgs.get('lamType', 'scalar')
        if self.lamType in ('scalar', 'pre_vector', 'post_vector', 'matrix'):
            init_string += 'Lam: {} // '.format(self.lamType)
        else:
            raise ValueError('lamType: {} not recognized'.format(self.lamType))

        # Maximum lambda values
        self.lamClamp = mpnArgs.get('lamClamp', 1.0)
        init_string += 'Lambda_max: {:.2f}'.format(self.lamClamp)

        if self.verbose: # Full summary of network parameters
            print(init_string)

        # Register_buffer                     
        self.register_buffer('M', None) 
        try:          
            self.reset_state() # Sets Hebbian weights to zeros
        except AttributeError as e:
            print('Warning: {}. Not running reset_state() in mpnNet.__init__'.format(e))
        
        self.register_buffer('plastic', torch.tensor(True))    
        self.register_buffer('forceAnti', torch.tensor(False))        # Forces eta to be negative
        self.register_buffer('forceHebb', torch.tensor(False))        # Forces eta to be positive
        self.register_buffer('groundTruthPlast', torch.tensor(False))
        
        self.init_sm_matrix()

    def reset_state(self, batchSize=1):

        self.M = torch.ones(batchSize, *self.w1.shape, device=self.w1.device) #shape=[B,Ny,Nx]   
        self.M = self.M * self.M0.unsqueeze(0) # (B, Ny, Nx) x (1, Ny, Nx)

    def init_sm_matrix(self):

        # Initialize different forms of eta parameter
        if self.etaType == 'scalar':
            _, b_eta = random_weight_init((1, 1), bias=True) # Xavier init between [-sqrt(3), sqrt(3)]
            eta = [[[b_eta[0][0]]]] # shape (1, 1, 1)
            # eta = [[[-5./self.w1.shape[1]]]] #eta*d = -5, shape (1, 1, 1)
        elif self.etaType == 'pre_vector':
            _, b_eta = random_weight_init([self.n_inputs, self.n_inputs], bias=True)
            eta = b_eta[0]
            eta = eta[np.newaxis, np.newaxis, :] # makes (1, 1, Nx)
        elif self.etaType == 'post_vector':
            _, b_eta = random_weight_init([self.n_inputs, self.n_outputs], bias=True)
            eta = b_eta[0]
            eta = eta[np.newaxis, :, np.newaxis] # makes (1, Ny, 1)   
        elif self.etaType == 'matrix':
            w_eta, _ = random_weight_init([self.n_inputs, self.n_outputs], bias=True)
            eta = w_eta[0]
            eta = eta[np.newaxis, :, :] # makes (1, Ny, Nx)  

        # Initialize different forms of lambda parameter
        if self.lamType == 'scalar': # For scalar and scalar only, lam is automatically initialized to the clamped value, otherwise uniform
            lam = [[[self.lamClamp]]] # shape (1, 1, 1)
            # eta = [[[-5./self.w1.shape[1]]]] #eta*d = -5, shape (1, 1, 1)
        elif self.lamType == 'pre_vector':
            lam = np.random.uniform(low=0.0, high=self.lamClamp, size=(self.n_inputs,))
            lam = lam[np.newaxis, np.newaxis, :] # makes (1, 1, Nx)
        elif self.lamType == 'post_vector':
            lam = np.random.uniform(low=0.0, high=self.lamClamp, size=(self.n_outputs,))
            lam = lam[np.newaxis, :, np.newaxis] # makes (1, Ny, 1)   
        elif self.lamType == 'matrix':
            lam = np.random.uniform(low=0.0, high=self.lamClamp, size=(self.n_inputs, self.n_outputs,))
            lam = lam[np.newaxis, :, :] # makes (1, Ny, Nx) 
            
        #Hebbian learning rate 
        if self.forceAnti:
            if self.eta: 
                del self.eta
            self._eta = nn.Parameter(torch.log(torch.abs(torch.tensor(eta, dtype=torch.float)))) #eta = exp(_eta)
            self.eta = -torch.exp(self._eta)
        elif self.forceHebb: 
            if self.eta: 
                del self.eta
            self._eta = nn.Parameter(torch.log(torch.abs(torch.tensor(eta, dtype=torch.float)))) #eta = exp(_eta)
            self.eta = torch.exp(self._eta)
        else: # Unconstrained eta
            if self.freezeLayer:
                if self.etaType != 'scalar':
                    raise NotImplementedError('Still need to set defaults for non-scalar eta')
                # self.register_buffer('_eta', torch.tensor(-1.0, dtype=torch.float)) # Anti-hebbian
                self.register_buffer('_eta', torch.tensor(1.0, dtype=torch.float))
                self.eta = self._eta
            else:
                self._eta = nn.Parameter(torch.tensor(eta, dtype=torch.float))    
                self.eta = self._eta.data    

        # Setting lambda parameter
        if self.freezeLayer:
            if self.lamType != 'scalar':
                raise NotImplementedError('Still need to set defaults for non-scalar lambda')
            self.register_buffer('_lam', torch.tensor(self.lamClamp))
            self.lam = self._lam
        else:
            self._lam = nn.Parameter(torch.tensor(lam, dtype=torch.float))
            self.lam = self._lam.data

    def update_sm_matrix(self, pre, post, stateOnly=False):
        """
        Updates the synaptic modulation matrix from one time step to the next.
        Should only be called in the forward pass once.

        Note that this directly updates self.M, unless stateOnly=True in which case
        it returns the updated state.

        """

        self._lam.data = torch.clamp(self._lam.data, min=0., max=self.lamClamp) 
        self.lam = self._lam 
        
        if self.forceAnti:
            self.eta = -torch.exp(self._eta)
        elif self.forceHebb: 
            self.eta = torch.exp(self._eta)
        else:
            self.eta = self._eta
        
        # Changes to post and pre if ignoring respective neurons for update
        if self.hebbType == 'input':
            post = torch.ones_like(post)
        elif self.hebbType == 'output':
            pre = torch.ones_like(pre)

        if self.plastic: 
            if self.groundTruthPlast: #and isFam: # only decays hebbian weights 
                raise NotImplementedError('Broke this functionality to get around something earlier.')
                M = self.lam*self.M
            elif self.updateType == 'hebb': # normal hebbian update
                if self.MAct is None:
                    M = self.lam*self.M + self.eta*torch.bmm(post, pre.unsqueeze(1))
                elif self.MAct == 'tanh':
                    M = torch.tanh(self.lam*self.M + self.eta*torch.bmm(post, pre.unsqueeze(1))) # [B, Ny, 1] x [B, 1, Nx] = [B, Ny, Nx]
            elif self.updateType == 'hebb_norm': # normal hebbian update
                M_tilde = self.lam*self.M + self.eta*torch.bmm(post, pre.unsqueeze(1)) # Normalizes over input dimension 
                M = M_tilde / torch.norm(M_tilde, dim=2, keepdim=True) # [B, Ny, Nx] / [B, Ny, 1]
            elif self.updateType == 'oja': # For small eta, effectively normalizes M matrix.
                M = self.lam*self.M + self.eta*(torch.bmm(post, pre.unsqueeze(1)) + post**2 * self.M) # [B, Ny, 1] x [B, 1, Nx] + [B, Ny, 1] * [B, Ny, Nx]
                
            if stateOnly: # This option is for functionality for finding fixed points
                return M
            else:
                self.M = M
                return None

    def forward(self, x, debug=False, stateOnly=False):
        """
        Passes inputs through the network and also modifies the internal state of the model (self.M). 
        Don't call twice in a row unless you want to update self.M twice!

        x.shape: [B, Nx]
        b1.shape: [Ny]
        w1.shape=[Ny,Nx], 
        M.shape=[B,Ny,Nx], 

        """
                
        # w1 = self.g1*self.w1 if not torch.isnan(self.g1) else self.w1
        
        # print('M', self.M.shape)
        # print('x', x.shape)
        # print('b1', self.b1.shape)
        # print('w1', self.w1.shape)

        # b1 + (w1 + A) * x
        # (Nh, 1) + [(B, Nh, Nx) x (B, Nx, 1) = (B, Nh, 1)] = (B, Nh, 1) -> (B, Nh)

        # Clamps w1 before it is used in forward pass if cellTypes are being used
        if self.useCellTypes: # First multiplication removes signs, then clamps, then restores them
            self.w1.data = torch.clamp(self.w1.data*self.cellTypes, min=0., max=1e6) * self.cellTypes

        if self.sparsification == 0.0:
            if self.mpType == 'add':
                y_tilde = torch.baddbmm(self.b1.unsqueeze(1), self.w1+self.M, x.unsqueeze(2))
            elif self.mpType == 'mult':
                y_tilde = torch.baddbmm(self.b1.unsqueeze(1), self.w1*(
                        self.M + torch.ones_like(self.M)), x.unsqueeze(2))
                # y_tilde = torch.baddbmm(self.b1.unsqueeze(1), self.w1*self.M, x.unsqueeze(2))
        else: # Applies masking of weights to sparsify network
            if self.mpType == 'add':
                y_tilde = torch.baddbmm(self.b1.unsqueeze(1), self.w1Mask*(self.w1+self.M), x.unsqueeze(2))
            elif self.mpType == 'mult':
                y_tilde = torch.baddbmm(self.b1.unsqueeze(1), self.w1Mask*self.w1*(
                        self.M + torch.ones_like(self.M)), x.unsqueeze(2))

        # Adds noise to the preactivations
        if self.noiseType in ('layer',):
            batch_noise = self.noiseScale*torch.normal(
                torch.zeros_like(y_tilde), 1/np.sqrt(self.n_outputs)*torch.ones_like(y_tilde)
                )
            y_tilde = y_tilde + batch_noise 

        y = self.f(y_tilde) if self.f is not None else y_tilde
        
        M = self.update_sm_matrix(x, y, stateOnly=stateOnly) # Returned M is only used if finding fixed points        
        
        if stateOnly:
            return M                 
        elif debug:
            return y_tilde, y, self.M
        else:
            return y  

class FreeNet(StatefulBase):     
    def __init__(self, init, verbose=True, **mpnArgs):        
        super().__init__()        
        
        if all([type(x)==int for x in init]) and len(init) == 3:
            Nx,Nh,Ny = init
            # For readouts
            # W,b = random_weight_init([Nh,Ny], bias=True)
            readout = nn.Linear(in_features=Nh, out_features=Ny, bias=True)
            nn.init.xavier_uniform_(readout.weight)
            if readout.bias is not None:
                nn.init.xavier_uniform_(readout.bias)

            self.n_inputs = Nx
            self.n_hidden = Nh
            self.n_outputs = Ny 
        else:
            raise ValueError('Init type not recognized.')
        
        self.verbose = verbose

        self.loss_fn = F.cross_entropy # Reductions is mean by default
        self.acc_fn = xe_classifier_accuracy 

        # Creates the input MP layer
        self.mp_layer = FreeLayer((self.n_inputs, self.n_hidden), verbose=verbose, **mpnArgs)

        init_string = 'Network parameters:'

        self.fOutAct = mpnArgs.pop('fOutAct', 'linear') # output activiation   
        if self.fOutAct == 'linear':
            self.fOut = None # No activiation function for output
        else:
            raise ValueError('f0 activaiton not recognized')
        
        init_string += '\n  Readout act: {} // '.format(self.fOutAct)

        # Readout layer (always trainable)
        self.w2 = nn.Parameter(torch.tensor(W[0], dtype=torch.float))
        
        # Determines if readout bias is trainable or simply not used (easier interpretting readouts in the latter)
        self.roBias = mpnArgs.pop('roBias', True)
        if self.roBias:
            init_string += 'Readout bias: trainable // '
            self.b2 = nn.Parameter(torch.tensor(b[0], dtype=torch.float))
        else:
            init_string += 'No readout bias // '
            self.register_buffer('b2', torch.zeros_like(torch.tensor(b[0])))

        # Injects noise into the network
        self.noiseType = mpnArgs.get('noiseType', None)
        self.noiseScale = mpnArgs.get('noiseScale', 0.0)
        if self.noiseType is None:
            init_string += 'Noise type: None'
        elif self.noiseType in ('input',):
            init_string += 'Noise type: {} (scl: {:.1e})'.format(self.noiseType, self.noiseScale)

        if self.verbose: # Full summary of readout parameters (MP layer prints out internally)
            print(init_string)      
        
    def load(self, filename):
        super(FreeNet, self).load(filename)
        # self.update_hebb(torch.tensor([0.]),torch.tensor([0.])) # to get self.eta right if forceHebb/forceAnti used        
    
    def reset_state(self, batchSize=1):
        """
        Resets states of all internal layer SM matrices
        """
        self.mp_layer.reset_state(batchSize=batchSize)   

    def forward(self, x, debug=False, stateOnly=False):
        """
        This modifies the internal state of the model (self.M). 
        Don't call twice in a row unless you want to update self.M twice!

        x.shape: [B, Nx]
        b1.shape: [Nh]
        w1.shape=[Nx,Nh], 
        A.shape=[B,Nh,Nx], 

        """
                
        # w1 = self.g1*self.w1 if not torch.isnan(self.g1) else self.w1
        
        # print('M', self.M.shape)
        # print('x', x.shape)
        # print('b1', self.b1.shape)
        # print('w1', self.w1.shape)

        # b1 + (w1 + A) * x
        # (Nh, 1) + [(B, Nh, Nx) x (B, Nx, 1) = (B, Nh, 1)] = (B, Nh, 1) -> (B, Nh)

        # Adds noise to the input
        if self.noiseType in ('input',):
            batch_noise = self.noiseScale*torch.normal(
                torch.zeros_like(x), 1/np.sqrt(self.n_inputs)*torch.ones_like(x)
                )
            x = x + batch_noise 

        outs = self.mp_layer(x, debug=debug, stateOnly=stateOnly)

        if stateOnly:
            M = outs
            return M
        elif debug:
            h_tilde, h, M = outs
        else:
            h = outs
        # (Ny) + (B, Nh) x (Nh, Ny) = (B, Ny)
        # print('h', h.shape)
        # print('b2', self.b2.shape)
        # print('w2', self.w2.shape)

        # (1, Ny) + [(B, Nh,) x (Nh, Ny) = (B, Ny)] = (B, Ny)
        y_tilde = self.b2.unsqueeze(0) + torch.mm(h.squeeze(dim=2), torch.transpose(self.w2, 0, 1)) #output layer activation
        y = self.fOut(y_tilde) if self.fOut is not None else y_tilde  
                           
        if debug: # Make a1 and h size [B, Nh]
            return h_tilde.squeeze(dim=2), h.squeeze(dim=2), y_tilde, y, M 
        else:
            return y   
     
