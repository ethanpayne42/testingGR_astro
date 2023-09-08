import h5py
import sys
import os
import numpy as np
import pandas as pd

import sys

from tqdm import tqdm
from glob import glob

from copy import deepcopy

from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

from astropy.cosmology import Planck15, z_at_value
from astropy import units

import lal
from lalinference.bayespputils import lambda_a
HPLANCK = 4.135667662E-15

def euclidean_distance_prior(samples):
    redshift = samples["redshift"]
    luminosity_distance = Planck15.luminosity_distance(redshift).to(units.Gpc).value
    return luminosity_distance**2 * (
        luminosity_distance / (1 + redshift)
        + (1 + redshift)
        * Planck15.hubble_distance.to(units.Gpc).value
        / Planck15.efunc(redshift)
    )
    
zs_ = np.linspace(0, 2.01, 1000)
p_z = euclidean_distance_prior(dict(redshift=zs_))
p_z /= np.trapz(p_z, zs_)
interpolated_p_z = interp1d(zs_, p_z)
    
def jacobian_mass(loglA, loglAmin, loglAmax):
    return np.log(10)*(np.power(10.0, loglAmax) - np.power(10.0, loglAmin))/\
           (np.power(10.0, loglA + loglAmax + loglAmin))

def energy_scale(lambda_A):
    return HPLANCK*lal.C_SI/lambda_A

def lambda_A_of_eff(leffedata, z, dist):
    return lambda_a(z, 0, leffedata, dist)

def mass_from_logl_eff(log_lambda_eff, redshift, luminosity_distance):
    
    # Returns mass in eV/c^2
    lambda_eff = np.power(10, log_lambda_eff)
    lambda_0 = lambda_A_of_eff(lambda_eff, redshift, luminosity_distance)
    
    return np.log10(energy_scale(lambda_0))


interested_keys = ['mass_1_source', 'mass_2_source', 'mass_ratio', 'cos_tilt_1', 'cos_tilt_2', 'a_1', 'a_2', 'redshift', 'log_mg', 'prior']
#interested_keys_o3b = ['mass_1', 'mass_2', 'mass_ratio', 'cos_tilt_1', 'cos_tilt_2', 'a_1', 'a_2', 'redshift', param, 'prior']
interested_keys2 = ['mass_1_source', 'mass_2_source', 'mass_ratio', 'cos_tilt_1', 'cos_tilt_2', 'a_1', 'a_2', 'redshift', 'dphi', 'prior']


outdir = sys.argv[1]
o3a_data_dir = sys.argv[2]
o3b_data_dir = sys.argv[3]

O3a_files = glob(f'{o3a_data_dir}/*Aplus_alpha0.h5')
O3b_files = glob(f'{o3b_data_dir}/*Aplus_alpha0.h5')

print(len(O3a_files)+len(O3b_files))

if not os.path.exists(f'{outdir}/O3a/'):
    os.makedirs(f'{outdir}/O3a/')

if not os.path.exists(f'{outdir}/O3b/'):
    os.makedirs(f'{outdir}/O3b/')

for i, file in tqdm(enumerate(O3a_files)):
    filename = file.split('/')[-1].split('.')[0]
    print(filename)
    
    sample_dict = {
        'tgr':{'posterior_samples':{}}
    }
    
    # Read in the data
    data = h5py.File(file)
    print(data.keys())
    try:
        try:
            arr_df = pd.DataFrame(np.array(data['aplus_alpha0']['posterior_samples']))
            
        except:
            try:
                arr_df = pd.DataFrame(np.array(data['Aplus_alpha0']['posterior_samples']))
            except:
                arr_df = pd.DataFrame(np.array(data['Aplus_alpha0p0']['posterior_samples']))
        
    except:
        print(data['posterior_samples'].keys())
        
        try:
            data_name = 'aplus_alpha0'
            names = [n.decode() for n in data['posterior_samples']['aplus_alpha0']['parameter_names'][()]]
        except:
            try:
                data_name = 'Aplus_alpha0'
                names = [n.decode() for n in data['posterior_samples']['Aplus_alpha0']['parameter_names'][()]]
            except:
                data_name = 'Aplus_alpha0p0'
                names = [n.decode() for n in data['posterior_samples']['Aplus_alpha0p0']['parameter_names'][()]]
        print(names)
        arr = data['posterior_samples'][data_name]['samples'][()]
        
        df_dict = {}
        arr_samples = []
        for key in names:
            matched_indexes = []
            k=0
            while k < len(names):
                if key == names[k]:
                    matched_indexes.append(k)
                k += 1
            
            df_dict[key] = arr[:,matched_indexes[0]]

        arr_df = pd.DataFrame.from_dict(df_dict)
    
    # Convert the samples
    arr_df['log_mg'] = mass_from_logl_eff(arr_df['log10lambda_eff'], arr_df['redshift'], arr_df['luminosity_distance'])
    mask = arr_df['log_mg'] > -30
    
    # Sampling prior
    arr_df['prior'] = 1/4
    arr_df['prior'] *= interpolated_p_z(arr_df["redshift"])
    arr_df['prior'] *= arr_df["mass_1"]
    
    new_arr = arr_df.sample(n=10000, replace=True, weights=mask).reset_index(drop=True)[interested_keys]
    ds_dt = np.dtype({'names':interested_keys,'formats':[(float)]*len(interested_keys)}) 
    posterior_samples = np.rec.fromarrays([np.array(new_arr)[:,i] for i in range(len(interested_keys))], dtype=ds_dt)
    posterior_samples.dtype.names = interested_keys2
    
    sample_dict['tgr']['posterior_samples'] = posterior_samples
    
    with h5py.File(f'{outdir}/O3a/{filename}_noAstroWeight.h5', 'w') as h5file:
        h5file.create_dataset('tgr/posterior_samples', data=posterior_samples, dtype=posterior_samples.dtype)
        
    # Calculation for the astro reweighted samples to significantly increase efficiency
    astro_prior = arr_df['mass_1_source']**(-4)
    astro_prior *= np.exp(-(arr_df['a_1'] - 0.3)**2/(2*0.3**2))
    astro_prior *= np.exp(-(arr_df['a_2'] - 0.3)**2/(2*0.3**2))
    astro_prior *= interpolated_p_z(arr_df["redshift"]) * (1+arr_df["redshift"])**3
    
    astro_weight = astro_prior/arr_df['prior']
    arr_df['prior'] = astro_prior
    
    new_arr2 = arr_df.sample(n=10000, replace=True, weights=mask*astro_weight).reset_index(drop=True)[interested_keys]
    
    ds_dt = np.dtype({'names':interested_keys,'formats':[(float)]*len(interested_keys)}) 
    posterior_samples = np.rec.fromarrays([np.array(new_arr2)[:,i] for i in range(len(interested_keys))], dtype=ds_dt)
    posterior_samples.dtype.names = interested_keys2
    
    sample_dict['tgr']['posterior_samples'] = posterior_samples
    
    with h5py.File(f'{outdir}/O3a/{filename}_weights.h5', 'w') as h5file:
        h5file.create_dataset('tgr/posterior_samples', data=posterior_samples, dtype=posterior_samples.dtype)
    
for i, file in tqdm(enumerate(O3b_files)):
    filename = file.split('/')[-1].split('.')[0]
    
    print(filename)
    
    sample_dict = {
        'tgr':{'posterior_samples':{}}
    }
    
    # Read in the data
    data = h5py.File(file)
    
    print(data.keys())
    try:
        try:
            try:
                arr_df = pd.DataFrame(np.array(data[f"liv_{filename.split('_')[1]}_Aplus_dalpha0"]['posterior_samples']))
                
            except:
                arr_df = pd.DataFrame(np.array(data[f"liv_{filename.split('_')[1]}_set1-dalpha0"]['posterior_samples']))
            
        except:
            try:
                arr_df = pd.DataFrame(np.array(data['LIV_Aplus_a0_pesummary']['posterior_samples']))
            except:
                arr_df = pd.DataFrame(np.array(data['Aplus_alpha0p0']['posterior_samples']))
        
    except:
        print(data['posterior_samples'].keys())
        
        try:
            data_name = 'aplus_alpha0'
            names = [n.decode() for n in data['posterior_samples']['aplus_alpha0']['parameter_names'][()]]
        except:
            try:
                data_name = 'Aplus_alpha0'
                names = [n.decode() for n in data['posterior_samples']['Aplus_alpha0']['parameter_names'][()]]
            except:
                data_name = 'Aplus_alpha0p0'
                names = [n.decode() for n in data['posterior_samples']['Aplus_alpha0p0']['parameter_names'][()]]
        print(names)
        arr = data['posterior_samples'][data_name]['samples'][()]
        
        df_dict = {}
        arr_samples = []
        for key in names:
            matched_indexes = []
            k=0
            while k < len(names):
                if key == names[k]:
                    matched_indexes.append(k)
                k += 1
            
            df_dict[key] = arr[:,matched_indexes[0]]

        arr_df = pd.DataFrame.from_dict(df_dict)
    
    # Convert the samples
    arr_df['log_mg'] = mass_from_logl_eff(arr_df['log10lambda_eff'], arr_df['redshift'], arr_df['luminosity_distance'])
    mask = arr_df['log_mg'] > -30
    
    # Sampling prior
    arr_df['prior'] = 1/4
    arr_df['prior'] *= interpolated_p_z(arr_df["redshift"])
    arr_df['prior'] *= arr_df["mass_1"]
    
    new_arr = arr_df.sample(n=10000, weights=mask, replace=True).reset_index(drop=True)[interested_keys]
    ds_dt = np.dtype({'names':interested_keys,'formats':[(float)]*len(interested_keys)}) 
    posterior_samples = np.rec.fromarrays([np.array(new_arr)[:,i] for i in range(len(interested_keys))], dtype=ds_dt)
    posterior_samples.dtype.names = interested_keys2
    
    sample_dict['tgr']['posterior_samples'] = posterior_samples
    
    with h5py.File(f'{outdir}/O3b/{filename}_noAstroWeight.h5', 'w') as h5file:
        h5file.create_dataset('tgr/posterior_samples', data=posterior_samples, dtype=posterior_samples.dtype)
        
    # Calculation for the astro reweighted samples to significantly increase efficiency
    astro_prior = arr_df['mass_1_source']**(-4)
    astro_prior *= np.exp(-(arr_df['a_1'] - 0.3)**2/(2*0.3**2))
    astro_prior *= np.exp(-(arr_df['a_2'] - 0.3)**2/(2*0.3**2))
    astro_prior *= interpolated_p_z(arr_df["redshift"]) * (1+arr_df["redshift"])**3
    
    astro_weight = astro_prior/arr_df['prior']
    arr_df['prior'] = astro_prior
    
    new_arr2 = arr_df.sample(n=10000, replace=True, weights=mask*astro_weight).reset_index(drop=True)[interested_keys]
    
    ds_dt = np.dtype({'names':interested_keys,'formats':[(float)]*len(interested_keys)}) 
    posterior_samples = np.rec.fromarrays([np.array(new_arr2)[:,i] for i in range(len(interested_keys))], dtype=ds_dt)
    posterior_samples.dtype.names = interested_keys2
    
    sample_dict['tgr']['posterior_samples'] = posterior_samples
    
    
    with h5py.File(f'{outdir}/O3b/{filename}_weights.h5', 'w') as h5file:
        h5file.create_dataset('tgr/posterior_samples', data=posterior_samples, dtype=posterior_samples.dtype)