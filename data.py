import random
import numpy as np
import torch
import int_data as syn #synthetic data
from torch.utils.data import TensorDataset

               
def generate_recog_data(T=2000, d=50, R=1, P=0.5, interleave=True, multiRep=False, xDataVals='+-', softLabels=False):
    """Generates "image recognition dataset" sequence of (x,y) tuples. 
    x[t] is a d-dimensional random binary vector, 
    y[t] is 1 if x[t] has appeared in the sequence x[0] ... x[t-1], and 0 otherwise
    
    if interleave==False, (e.g. R=3) ab.ab.. is not allowed, must have a..ab..b.c..c (dots are new samples)
    if multiRep==False a pattern will only be (intentionally) repeated once in the trial
    
    T: length of trial
    d: length of x
    R: repeat interval
    P: probability of repeat
    """  
    if np.isscalar(R):
        Rlist = [R]
    else:
        Rlist = R
    
    data = []
    repeatFlag = False
    r=0 #countdown to repeat
    for t in range(T): 
        #decide if repeating
        R = Rlist[np.random.randint(0, len(Rlist))]
        if interleave:
            repeatFlag = np.random.rand()<P
        else:
            if r>0:
                repeatFlag = False
                r-=1
            else:
                repeatFlag = np.random.rand()<P 
                if repeatFlag:
                    r = R
                
        #generate datapoint
        if t>=R and repeatFlag and (multiRep or data[t-R][1].round()==0):
            x = data[t-R][0]
            y = 1
        else:
            if xDataVals == '+-': #TODO should really do this outside the loop...
                x = 2*np.round(np.random.rand(d))-1
            elif xDataVals.lower() == 'normal':
                x = np.sqrt(d)*np.random.randn(d)    
            elif xDataVals.lower().startswith('uniform'):
                upper, lower = parse_xDataVals_string(xDataVals)
                x = np.random.rand(d)*(upper-lower)+lower
            elif xDataVals == '01':
                x = np.round(np.random.rand(d))
            else:
                raise ValueError('Invalid value for "xDataVals" arg')           
            y = 0
            
        if softLabels:
            y*=(1-2*softLabels); y+=softLabels               
        data.append((x,np.array([y]))) 
        
    return data_to_tensor(data)

 
def generate_recog_data_batch(T=2000, batchSize=1, d=25, R=1, P=0.5, interleave=True, multiRep=False, softLabels=False, xDataVals='+-', device='cpu'):
    """Faster version of recognition data generation. Generates in batches and uses torch directly    
    Note: this is only faster when approx batchSize>4
    """  
    if np.isscalar(R):
        Rlist = [R]
    else:
        Rlist = R
    
    if xDataVals == '+-':
        x = 2*torch.rand(T,batchSize,d, device=device).round()-1 #faster than (torch.rand(T,B,d)-0.5).sign()
    elif xDataVals.lower() == 'normal':
        x = torch.randn(T,batchSize,d, device=device)    
    elif xDataVals.lower().startswith('uniform'):
        upper, lower = parse_xDataVals_string(xDataVals)
        x = torch.rand(T,batchSize,d, device=device)*(upper-lower)+lower
    elif xDataVals == '01':
        x = torch.rand(T,batchSize,d, device=device).round()
    else:
        raise ValueError('Invalid value for "xDataVals" arg')  
    
    y = torch.zeros(T,batchSize, dtype=torch.bool, device=device)
    
    for t in range(max(Rlist), T):
        R = Rlist[np.random.randint(0, len(Rlist))] #choose repeat interval   
        
        if interleave:
            repeatMask = torch.rand(batchSize)>P
        else:
            raise NotImplementedError
        
        if not multiRep:
            repeatMask = repeatMask*(~y[t-R]) #this changes the effective P=n/m to P'=n/(n+m)
          
        x[t,repeatMask] = x[t-R,repeatMask]            
        y[t,repeatMask] = 1
        
    y = y.unsqueeze(2).float()
    if softLabels:
        y = y*0.98 + 0.01

    # x.shape = torch.Size([T, batchSize, d])
    # y.shape = torch.Size([T, batchSize, 1]) 

    return TensorDataset(x, y)


class GenRecogClassifyData():
    def __init__(self, d=None, teacher=None, datasize=int(1e4), sampleSpace=None, save=False, device='cpu'):
        if sampleSpace is None:
            x = torch.rand(datasize,d, device=device).round()*2-1
            if teacher is None:
                c = torch.randint(2,(datasize,1), device=device, dtype=torch.float)
            else:
                c = torch.empty(datasize,1, device=device, dtype=torch.float)
                for i,xi in enumerate(x):
                    c[i] = teacher(xi)
                c = (c-c.mean()+0.5).round()
            self.sampleSpace = TensorDataset(x,c)
            if save:
                if type(save) == str:
                    fname = save
                else:
                    fname = 'sampleSpace.pkl'
                torch.save(self.sampleSpace, fname)
        elif type(sampleSpace) == str:
            self.sampleSpace = torch.load(sampleSpace) 
        elif type(sampleSpace) == TensorDataset:
            self.sampleSpace = sampleSpace
            
        self.datasize, self.d = self.sampleSpace.tensors[0].shape            
        
        
    def __call__(self, T, R, P=0.5, batchSize=-1, multiRep=False, device='cpu'):
        if np.isscalar(R):
            Rlist = [R]
        else:
            Rlist = R
        
        squeezeFlag=False
        if batchSize is None:
            batchSize=1
            squeezeFlag=True
        elif batchSize < 0:
            batchSize = self.datasize/T
            
        randomSubsetIdx = torch.randperm(len(self.sampleSpace))[:T*batchSize]
        x,c = self.sampleSpace[randomSubsetIdx]
        x = x.reshape(T,batchSize,self.d)
        c = c.reshape(T,batchSize,1)
        y = torch.zeros(T,batchSize, dtype=torch.bool, device=device)
        for t in range(max(Rlist), T):    
            R = Rlist[np.random.randint(0, len(Rlist))] #choose repeat interval   
                   
            repeatMask = torch.rand(batchSize)>P   
            if not multiRep:
                repeatMask = repeatMask*(~y[t-R]) #this changes the effective P
              
            x[t,repeatMask] = x[t-R,repeatMask] 
            c[t,repeatMask] = c[t-R,repeatMask]            
            y[t,repeatMask] = 1
         
        y = y.unsqueeze(2).float()
        y = torch.cat((y,c), dim=-1)        
        data = TensorDataset(x,y)
        
        if squeezeFlag:
            data = TensorDataset(*data[:,0,:])
    
        return data
    

#%%############
### Helpers ###   
###############
def parse_xDataVals_string(xDataVals):
    assert xDataVals.lower().startswith('uniform')
    delimIdx = xDataVals.find('_')
    if delimIdx > 0:
        assert delimIdx==7
        lims = xDataVals[delimIdx+1:]
        lower = float(lims[:lims.find('_')])
        upper = float(lims[lims.find('_')+1:])
    else:
        lower = -1
        upper = 1
    return upper, lower


def prob_repeat_to_frac_novel(P, multiRep=False):
    if multiRep:
        return P
    n,m = P.as_integer_ratio()
    return 1 - float(n)/(m+n)
    

def check_recognition_data(data, R):
    """Make sure there are no spurious repeats"""
    if len(data) == 0:
        return False
    for i in range(len(data)):
        for j in range(0,i-1):
            if all(data[i][0] == data[j][0]):   
                if i-j != R:
                    print( 'bad R', i, j )
                    return False
                if not data[i][1]:
                    print( 'unmarked', i, j )
                    return False
    return True

             
def recog_chance(data):
    """Calculates expected performance if network simply guesses based on output statistics
    i.e. the number of zeroes in the data"""
    return 1-np.sum([xy[1] for xy in data], dtype=np.float)/len(data) 


def batch(generate_data, batchsize=1, batchDim=1, **dataKwargs):
    dataList = []
    for b in range(batchsize):
        dataList.append( generate_data(**dataKwargs) )
    x = torch.cat([data.tensors[0].unsqueeze(batchDim) for data in dataList], dim=batchDim)
    y = torch.cat([data.tensors[0].unsqueeze(batchDim) for data in dataList], dim=batchDim) 
    return TensorDataset(x,y)


def data_to_tensor(data, y_dtype=torch.float, device='cpu'):
    '''Convert from list of (x,y) tuples to TensorDataset'''
    x,y = zip(*data)
    return TensorDataset(torch.as_tensor(x, dtype=torch.float, device=device), 
                   torch.as_tensor(y, dtype=y_dtype, device=device))
    
    
from torch.utils.data import TensorDataset

def convert_serialized_mnist(dataset, smnist_params, out_size, auto_balance=False, verbose=True, raw_data=False, device='cpu'):
    """
    Prunes classes in MNIST and converts into serialized data
    """

    MNIST_DIM = 28 # Number of pixels of an MNIST image
    
    mnist_classes = smnist_params['smnist_classes']
    n_classes = len(mnist_classes)
    n_seq = smnist_params['phrase_length'] # Used only for input data, not labels or mask
    if smnist_params['include_eos']:
        n_seq = n_seq - 1
    n_smnist_input = int((MNIST_DIM*MNIST_DIM)/n_seq)
    
    n_inputs = smnist_params['input_size']

    batch_size = 64 # Just used internally, doesn't really matter what size this is

    ### Prunes classes that are not in 'mnist_classes'
    prune_bools = np.zeros((n_classes, len(dataset.targets)))
    for digit_idx in range(len(mnist_classes)):
        prune_bools[digit_idx] = dataset.targets==mnist_classes[digit_idx]

    idx = []
    for mnist_idx in range(len(dataset.targets)):
        if prune_bools[:, mnist_idx].any(): idx.append(mnist_idx)

    data_subset = torch.utils.data.Subset(dataset, idx)
    pruned_dataset = torch.utils.data.DataLoader(data_subset, batch_size=64, shuffle=True)

    # Converts dataset into the type that can be used by our code 
    # There are probably better ways to do this but this is just adapting old code...
      
    images_np = [] # list where elements are: batch_index x n_seq x n_smnist_input, concatanated later
    labels_np = [] # elements are batch_index x n_seq x label

    counter = 0
    for train_vals, idx in zip(pruned_dataset, range(len(pruned_dataset))):
        images, labels = train_vals[0], train_vals[1]
        squeezed_images = np.squeeze(images.numpy())
        
        batch_images_squeezed = np.squeeze(images.numpy())
        batch_size = squeezed_images.shape[0]

        if smnist_params['data_type'] == 'smnist_rows':
            images_np.append(batch_images_squeezed.reshape(batch_size, n_seq, n_smnist_input))
        elif smnist_params['data_type'] == 'smnist_columns':
            batch_images_columns = np.swapaxes(batch_images_squeezed, 1, 2)
            images_np.append(batch_images_columns.reshape(batch_size, n_seq, n_smnist_input))
        # Note this uses 'phrase_length' so correct size with or without eos
        labels_batch = np.zeros((squeezed_images.shape[0], smnist_params['phrase_length'], 1), dtype=np.int32)
        for label, batch_idx in zip(labels, range(len(labels))):
            labels_batch[batch_idx, -1, 0] = mnist_classes.index(label)
        labels_np.append(labels_batch)            
    
    # Concatanates over batch dimension
    images_np = np.concatenate(images_np, axis=0)
    labels_np = np.concatenate(labels_np, axis=0)

    # Now converts to full n_input size via usual word mapping
    if 'words'not in smnist_params: # should only be called on first pass
        smnist_params['words'] = ['{}'.format(idx) for idx in range(MNIST_DIM)]
        if smnist_params['include_eos']:
            smnist_params['words'].append('<eos>')
        smnist_params['word_to_input_vector'] = syn.generateInputVectors(smnist_params)
        smnist_params['input_norm'] = np.mean(np.sum(images_np, axis=-1))
        print('Average input sum: {:.2f}'.format(smnist_params['input_norm']))

    # Normalized inputs so expected input sum is 1
    images_np = images_np / smnist_params['input_norm']

    input_matrix = np.zeros((MNIST_DIM, n_inputs))
    for idx in range(MNIST_DIM):
        input_matrix[idx] = smnist_params['word_to_input_vector']['{}'.format(idx)]

    inputs = np.matmul(images_np, input_matrix)
    if smnist_params['include_eos']:
        eos_inputs = np.zeros((inputs.shape[0], 1, n_inputs))
        eos_inputs[:, 0] = smnist_params['word_to_input_vector']['<eos>']

        inputs = np.concatenate((inputs, eos_inputs), axis=1)


    masks_np = np.zeros((images_np.shape[0], smnist_params['phrase_length'], n_classes), dtype=np.int32)
    masks_np[:, -1, :] = np.ones((images_np.shape[0], n_classes))

    inputs_torch = torch.tensor(inputs, dtype=torch.float, device=device)
    labels_torch = torch.tensor(labels_np, dtype=torch.long, device=device)
    masks_torch = torch.tensor(masks_np, dtype=torch.bool, device=device)

    data = TensorDataset(inputs_torch, labels_torch)

    if raw_data:
        return data, masks_torch, None, smnist_params
    else:
        return data, masks_torch, smnist_params