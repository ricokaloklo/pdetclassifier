import os, copy
import pickle
from tqdm import tqdm
from schwimmbad import MultiPool

import numpy as np
import astropy.cosmology

import pycbc.psd
import pycbc.detector
import pycbc.waveform
import pycbc.filter

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.utils import data

# Specify the parameterization
massvar = ['mtot','q','z']
spinvar = ['chi1x','chi1y','chi1z','chi2x','chi2y','chi2z']
intrvar = massvar + spinvar
extrvar = ['iota','ra','dec','psi']
train_variables = intrvar + extrvar

def lookup_limits():
    '''
    Define the limits in all variables. If you want to change this, please check generate_binaries() and pdet() as well.
    '''

    limits={
        'mtot'  : [2,1000],
        'q'     : [0.1,1],
        'z'     : [1e-4,4],
        'chi1x'  : [-1,1],
        'chi1y'  : [-1,1],
        'chi1z'  : [-1,1],
        'chi2x'  : [-1,1],
        'chi2y'  : [-1,1],
        'chi2z'  : [-1,1],
        'iota'  : [0,np.pi],
        'ra'    : [-np.pi,np.pi],
        'dec'   : [-np.pi/2,np.pi/2],
        'psi'   : [0,np.pi]
        }

    return limits

def sperical_to_cartesian(mag,theta,phi):
    '''
    Convert spherical to cartesian coordinates
    '''

    coordx = mag * np.cos(phi) * np.sin(theta)
    coordy = mag * np.sin(phi) * np.sin(theta)
    coordz = mag * np.cos(theta)

    return coordx,coordy,coordz

def generate_binaries(N):
    '''
    Generate a sample of N binaries. Edit here to specify a different traning/validation distribution.
    '''

    N=int(N)
    limits=lookup_limits()

    binaries={}
    binaries['N']=N

    for var in ['mtot','q','z']:
        binaries[var]=np.random.uniform(min(limits[var]),max(limits[var]),N)
    binaries['iota']= np.arccos(np.random.uniform(-1,1,N))
    binaries['ra']= np.pi*np.random.uniform(-1,1,N)
    binaries['psi']= np.pi*np.random.uniform(0,1,N)
    binaries['dec']= np.arccos(np.random.uniform(-1,1,N))- np.pi/2

    mag = np.random.uniform(0,1,N)
    theta = np.arccos(np.random.uniform(-1,1,N))
    phi = np.pi*np.random.uniform(-1,1,N)
    binaries['chi1x'],binaries['chi1y'],binaries['chi1z'] = sperical_to_cartesian(mag,theta,phi)

    mag = np.random.uniform(0,1,N)
    theta = np.arccos(np.random.uniform(-1,1,N))
    phi = np.pi*np.random.uniform(-1,1,N)
    binaries['chi2x'],binaries['chi2y'],binaries['chi2z'] = sperical_to_cartesian(mag,theta,phi)

    return binaries

def get_psds(ifos, flow, delta_f, flen, noisecurve="design"):
    psds=[]
    for ifo in ifos:
        if noisecurve=="design" or noisecurve=="Design":
            if ifo == 'V1':
                psds.append(pycbc.psd.AdVDesignSensitivityP1200087(flen, delta_f, flow) )
            elif ifo == 'H1' or ifo == 'L1':
                psds.append(pycbc.psd.aLIGODesignSensitivityP1200087(flen, delta_f, flow) )

        elif noisecurve=="O1O2":
            if ifo == 'V1':
                psds.append(pycbc.psd.AdVEarlyHighSensitivityP1200087(flen, delta_f, flow) )
            elif ifo == 'H1' or ifo == 'L1':
                psds.append(pycbc.psd.aLIGOEarlyHighSensitivityP1200087(flen, delta_f, flow) )

        elif noisecurve=="O3":
            if ifo == 'H1':
                psds.append(pycbc.psd.from_txt('T2000012_aligo_O3actual_H1.txt', flen, delta_f,flow, is_asd_file=True) )
            elif ifo=='L1':
                psds.append(pycbc.psd.from_txt('T2000012_aligo_O3actual_L1.txt', flen, delta_f,flow, is_asd_file=True) )
            elif ifo=='V1':
                psds.append(pycbc.psd.from_txt('T2000012_avirgo_O3actual.txt', flen, delta_f,flow, is_asd_file=True) )

        elif noisecurve=="O4":
            if ifo == 'V1':
                psds.append(pycbc.psd.from_txt('T2000012_avirgo_O4high_NEW.txt', flen, delta_f,flow, is_asd_file=True) )
            elif ifo == 'H1' or ifo == 'L1':
                psds.append(pycbc.psd.from_txt('T2000012_aligo_O4high.txt', flen, delta_f,flow, is_asd_file=True) )
        else:
            raise ValueError

    return psds

def calculate_snr(args):
    binaries, ifos, approximant, noisecurve = args
    dets = []
    for ifo in ifos:
        dets.append(pycbc.detector.Detector(ifo))
    psds = get_psds(ifos, binaries['flow'], binaries['delta_f'], binaries['flen'], noisecurve)

    # Waveform generator
    hp, hc = pycbc.waveform.get_fd_waveform(approximant = approximant,
                        mass1       = binaries['m1z'],
                        mass2       = binaries['m2z'],
                        spin1x      = binaries['chi1x'],
                        spin1y      = binaries['chi1y'],
                        spin1z      = binaries['chi1z'],
                        spin2x      = binaries['chi2x'],
                        spin2y      = binaries['chi2y'],
                        spin2z      = binaries['chi2z'],
                        inclination = binaries['iota'],
                        coa_phase   = binaries['phi_c'],
                        delta_f     = binaries['delta_f'],
                        f_lower     = binaries['flow'],
                        distance    = binaries['lumdist'],
                        )

    # Compute SNR for each specified detector
    snrs = []
    for det,psd in zip(dets,psds):
        f_plus, f_cross = det.antenna_pattern(binaries['ra'],binaries['dec'],binaries['psi'],binaries['geocent_time'])
        template = f_plus * hp + f_cross * hc
        dt = det.time_delay_from_earth_center(binaries['ra'],binaries['dec'],binaries['geocent_time'])
        template = template.cyclic_time_shift(dt)
        template.resize(len(hp) // 2 + 1)
        snr_opt = pycbc.filter.matched_filter(template, template,
                psd = psd,
                low_frequency_cutoff  = binaries['flow'],
                high_frequency_cutoff = binaries['fhigh'] - 0.5)
        maxsnr, _ = snr_opt.abs_max_loc()
        snrs.append(maxsnr)

    # Return the SNR in each detector and construct the network SNR later on
    return snrs

def evaluate_binaries(inbinaries, ifos=['H1','L1','V1'], ncore=1, approximant='IMRPhenomXPHM', noisecurve='design', SNRthreshold=12):
    '''
    Compute the SNRs of a set of binaries
    '''

    binaries = copy.deepcopy(inbinaries)
    
    # Prepare detectors and PSDs
    flow=20.0
    fhigh=2048.
    geocent_time=0.
    delta_f=1/64.
    flen=int(fhigh / delta_f) + 1
    phi_c=0.

    # Populate binaries with additional parameters
    # NOTE These hyperparameters should also be binary dependent!
    binaries['flow'] = np.ones(binaries['N'])*flow
    binaries['fhigh'] = np.ones(binaries['N'])*fhigh
    binaries['geocent_time'] = np.ones(binaries['N'])*geocent_time
    binaries['delta_f'] = np.ones(binaries['N'])*delta_f
    binaries['flen'] = np.ones(binaries['N'], dtype=int)*flen
    binaries['phi_c'] = np.ones(binaries['N'])*phi_c
    binaries['m1z'] = binaries['mtot']/(1+binaries['q'])
    binaries['m2z'] = binaries['q']*binaries['m1z']
    binaries['lumdist'] = astropy.cosmology.Planck15.luminosity_distance(binaries['z']).value # Mpc

    with MultiPool(ncore) as pool:
        output_snrs = list(
            tqdm(
                pool.imap(
                    calculate_snr,
                    [({k: binaries[k][i] for k in binaries.keys() if k != 'N'}, ifos, approximant, noisecurve) for i in range(binaries['N'])],
                ),
                total=binaries['N'],
            )
        )

    binaries['snr']=np.array(output_snrs)
    binaries['network_snr'] = np.linalg.norm(binaries['snr'], axis=1)

    # Detectability: 1 means "detected", 0 means "not detected"
    # NOTE Maybe implement "Quick recipes for gravitational-wave selection effects" here
    binaries['det']= np.where(binaries['network_snr']>SNRthreshold , 1,0 )

    return binaries

def store_binaries(filename, N, approximant='IMRPhenomXPHM', noisecurve='design', SNRthreshold=12):
    ''' Generate binaries, compute SNR, and store'''

    inbinaries = generate_binaries(N)
    outbinaries = evaluate_binaries(inbinaries, approximant, noisecurve, SNRthreshold)

    with open(filename, "wb") as f:
        pickle.dump(outbinaries, f)

    return filename

def readsample(filename):
    '''
    Read a validation sample that already exists
    '''
    with open(filename, "rb") as f:
        binaries = pickle.load(f)

    return binaries

def splittwo(binaries):
    '''
    Split sample into two subsamples of (almost) equal size
    '''

    one={}
    two={}
    for k in binaries.keys():
        if k == 'N':
            continue
        one[k],two[k] = np.array_split(binaries[k],2)
    one['N'],two['N']= len(one['mtot']),len(two['mtot'])

    return one,two


def rescale(x,var):
    '''
    Rescale variable sample x of variable var between -1 and 1
    '''

    limits=lookup_limits()
    if var not in limits:
        raise ValueError

    return 1-2*(np.array(x)-min(limits[var]))/(max(limits[var])-min(limits[var]))


def nnet_in(binaries):
    '''
    Prepare neural network inputs.
    '''

    return np.array([rescale(binaries[k],k) for k in train_variables]).T

def nnet_out(binaries, which='detnetwork'):
    '''
    Prepare neural network outputs.
    '''

    return binaries['det']

class LabeledDataset(data.Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels
        self.length = len(inputs)
 
    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]
    
    def __len__(self):
        return self.length

def loadnetwork(filename,verbose=False):
    '''
    Load a trained neural network
    '''

    model = torch.load(filename, weights_only=False)
    if verbose:
        print(model.__str__())

    return model

def trainnetwork_with_torch(train_binaries, test_binaries, filename='trained_model.pt', lr=1e-2, batch_size=1024, nepoch=50):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # Set default dtype
    torch.set_default_dtype(torch.float32)

    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)

    if not os.path.isfile(filename):
        train_in = nnet_in(train_binaries) # binary parameters
        train_out = nnet_out(train_binaries) # detectable or not
        test_in = nnet_in(test_binaries)
        test_out = nnet_out(test_binaries)

        nfeatures = np.shape(train_in)[1] # Number of binary parameters
        # Define neural network architecture
        model = nn.Sequential(
            nn.Linear(nfeatures, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

        # Initialize the weights for the linear layers
        model.apply(init_weights)
        # Move model to device (GPU or CPU)
        model.to(device)

        # Prepare the data and move to device
        train_in = torch.from_numpy(train_in.astype(np.float32)).to(device)
        train_out = torch.from_numpy(train_out.astype(np.float32)).to(device) # Either 0 or 1
        test_in = torch.from_numpy(test_in.astype(np.float32)).to(device)
        test_out = torch.from_numpy(test_out.astype(np.float32)).to(device)
        train_dataset = LabeledDataset(train_in, train_out)
        test_dataset = LabeledDataset(test_in, test_out)

        train_batch_size = batch_size
        validate_batch_size = batch_size # NOTE Does not have to be the same as train_batch_size
        train_dataloader = data.DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
        )
        test_dataloader = data.DataLoader(
            test_dataset,
            batch_size=validate_batch_size,
            shuffle=True,
        )

        # Loss function: binary cross entropy
        loss_fn = nn.BCELoss()

        optimizer = optim.Adam(model.parameters(), lr=lr)
        # Trying to reproduce Davide's learning rate schedule with torch
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                optim.lr_scheduler.ConstantLR(optimizer, factor=1, total_iters=10),
                optim.lr_scheduler.ExponentialLR(optimizer, gamma=np.exp(-0.05)),
            ],
            milestones=[10],
        )
        tbar = tqdm(range(nepoch))

        # Training loop
        best_score = -np.inf
        for step in tbar:
            # In each epoch
            model.train()

            for _, batch_data in enumerate(train_dataloader):
                batch_in, batch_label = batch_data
                optimizer.zero_grad()
                output = model(batch_in).squeeze()
                loss = loss_fn(output, batch_label)
                loss.backward()
                optimizer.step()

            # Check for accuracy on the validation set
            model.eval()
            with torch.no_grad():
                test_accuracy = []
                for _, batch_data in enumerate(test_dataloader):
                    batch_in, batch_label = batch_data
                    test_output = model(batch_in).squeeze()
                    test_accuracy.append(((test_output > 0.5) == batch_label).float().mean().item())
                test_accuracy = np.mean(test_accuracy)
                print("Current test accuracy: ", test_accuracy)
                print("Previous best score: ", best_score)
                if test_accuracy > best_score:
                    best_score = test_accuracy
                    torch.save(model, filename)

            scheduler.step()
    else:
        model = loadnetwork(filename, verbose=False)

    return model

def predictnetwork(model, binaries):
    '''
    Use a network to predict the detectability of a set of binaries.
    '''
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    inputs = torch.from_numpy(nnet_in(binaries).astype(np.float32)).to(device)

    model.eval()
    with torch.no_grad():
        pass
    # Return the class (0 or 1) that is preferred
    predictions = np.squeeze((model(inputs) > 0.5).detach().cpu().numpy().astype("int32"))
    return predictions

def keep_splitting(binaries, target_N):
    '''
    Split a sample until the number of binaries in each split is less than or equal to target_N
    '''

    if binaries['N']<=target_N:
        return [binaries]

    split = splittwo(binaries)
    return keep_splitting(split[0], target_N) + keep_splitting(split[1], target_N)

def _pdet(model,binaries, Nmc = 10000):
    '''
    Numerical marginalization over the extrinsic parameters. Nmc is the nubmer of Monte Carlo samples used to estimate the integral.
    '''

    limits = lookup_limits()
    # Number of binaries
    N=binaries['N']

    # Resample the extrinsic variables from isotropic distribution
    extrinsic={}
    extrinsic['iota'] =  np.arccos(np.random.uniform(-1,1,Nmc))
    extrinsic['ra']   =  np.pi*np.random.uniform(-1,1,Nmc)
    extrinsic['psi']  =  np.pi*np.random.uniform(-1,1,Nmc)
    extrinsic['dec']  =  np.arccos(np.random.uniform(-1,1,Nmc))- np.pi/2

    # Inflate the array with the intrisinc variables
    intersection  = [value for value in intrvar if value in binaries]
    intr = np.repeat([rescale(binaries[k],k) for k in intersection], Nmc, axis=1)
    # Inflate the array with the extrinsic variables
    extr = np.reshape(np.repeat([rescale(extrinsic[k],k) for k in extrvar],N,axis=0), (len(extrvar),N*Nmc))
    # Pair
    both = np.concatenate((intr,extr)).T

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    both = torch.from_numpy(both.astype(np.float32)).to(device)

    model.eval()
    with torch.no_grad():
        # Apply network
        predictions =  np.reshape( np.squeeze(( model(both)> 0.5).detach().cpu().numpy().astype("int32")), (N,Nmc) )

    # Approximante integral with monte carlo sum
    pdet_mc = np.sum(predictions,axis=1)/Nmc

    del both
    torch.cuda.empty_cache() # Clean up cache

    return pdet_mc

def pdet(model, binaries, Nmc=10000, nperbatch=1000):
    # Depending on how much the GPU can handle,
    # we can split the binaries into *bigger* chunks
    splitted_binaries = keep_splitting(binaries, int(nperbatch))

    predictions = np.array([])
    for binaries in tqdm(splitted_binaries):
        predictions = np.append(predictions, _pdet(model, binaries, Nmc=Nmc))

    return predictions