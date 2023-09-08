import h5py
import sys
import os
import bilby
import numpy as np
import pandas as pd

from astropy.cosmology import Planck15

from tqdm import tqdm
from glob import glob

from copy import deepcopy

from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

from astropy.cosmology import Planck15, z_at_value
from astropy import units

def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a,idx,axis=axis)

import numpy as np
import argparse
import re

def euclidean_distance_prior(samples):
    redshift = samples["redshift"]
    luminosity_distance = Planck15.luminosity_distance(redshift).to(units.Gpc).value
    return luminosity_distance**2 * (
        luminosity_distance / (1 + redshift)
        + (1 + redshift)
        * Planck15.hubble_distance.to(units.Gpc).value
        / Planck15.efunc(redshift)
    )

lal_gamma = 0.577215664901532860606512090082402431

#The functions phi${N} return the coefficient of the N/2-PN term in the inspiral (as in Eq. A4 of https://arxiv.org/abs/1005.3306)
def phi0(m1, m2, a1L, a2L, a1sq, a2sq, a1dota2):
  return np.ones(len(m1))

def phi1(m1, m2, a1L, a2L, a1sq, a2sq, a1dota2):
  return np.ones(len(m1))

def phi2(m1, m2, a1L, a2L, a1sq, a2sq, a1dota2):
  eta = (m1*m2)/(m1+m2)**2.
  return 5.*(743./84. + 11.*eta)/9.

def phi3(m1, m2, a1L, a2L, a1sq, a2sq, a1dota2):
  m1M = m1/(m1+m2)
  m2M = m2/(m1+m2)
  d = (m1-m2)/(m1+m2)
  SL = m1M * m1M * a1L + m2M * m2M * a2L
  dSigmaL = d * (m2M * a2L - m1M * a1L)
  return -16.* np.pi + 188.*SL/3. + 25.*dSigmaL

def phi4(m1, m2, a1L, a2L, a1sq, a2sq, a1dota2):
#qm_def are the spin susceptibailities of the objects, which we take as the black hole value of 1.
  qm_def1 = 1
  qm_def2 = 1
  m1M = m1/(m1+m2)
  m2M = m2/(m1+m2)
  eta = (m1*m2)/(m1+m2)**2.
  pnsigma = eta * (721./48. * a1L * a2L - 247./48. * a1dota2) + (720.*(qm_def1) - 1.)/96.0* m1M* m1M * a1L * a1L + (720. *(qm_def2) - 1.)/96.0 * m2M * m2M * a2L * a2L - (240.*(qm_def1) - 7.)/96.0 * m1M * m1M * a1sq - (240.*(qm_def2) - 7.)/96.0 * m2M * m2M * a2sq

  return 5.*(3058.673/7.056 + 5429./7.*eta + 617.*eta*eta)/72. - 10.*pnsigma

def phi5l(m1, m2, a1L, a2L, a1sq, a2sq, a1dota2):
  m1M = m1/(m1+m2)
  m2M = m2/(m1+m2)
  d = (m1-m2)/(m1+m2)
  eta = (m1*m2)/(m1+m2)**2.
  SL = m1M * m1M * a1L + m2M * m2M * a2L
  dSigmaL = d * (m2M * a2L - m1M * a1L)
  pngamma = (554345./1134. + 110.*eta/9.)*SL + (13915./84. - 10.*eta/3.)*dSigmaL
  return 5./3. * (7729./84. - 13. * eta) * np.pi - 3. * pngamma

def phi6(m1, m2, a1L, a2L, a1sq, a2sq, a1dota2):
  #qm_def are the spin susceptibailities of the objects, which we take as the black hole value of 1.
  qm_def1 = 1
  qm_def2 = 1
  m1M = m1/(m1+m2)
  m2M = m2/(m1+m2)
  d = (m1-m2)/(m1+m2)
  eta = (m1*m2)/(m1+m2)**2.
  SL = m1M * m1M * a1L + m2M * m2M * a2L
  dSigmaL = d * (m2M * a2L - m1M * a1L)
  pnss3 = (326.75/1.12 + 557.5/1.8*eta) * eta * a1L * a2L + ((4703.5/8.4 + 2935./6. * m1M - 120. * m1M * m1M)*(qm_def1) + (-4108.25/6.72 - 108.5/1.2*m1M + 125.5/3.6*m1M*m1M))*m1M*m1M* a1sq + ((4703.5/8.4 + 2935./6. * m2M - 120. * m2M * m2M)*(qm_def2) + (-4108.25/6.72 - 108.5/1.2*m2M + 125.5/3.6*m2M*m2M))*m2M*m2M* a2sq
  return (11583.231236531/4.694215680 - 640./3. * np.pi * np.pi - 6848./21.*lal_gamma) + eta*(-15737.765635/3.048192 + 2255./12.*np.pi*np.pi) + eta*eta*76055./1728. - eta*eta*eta*127825./1296. + (-6848./21.)*np.log(4.) + np.pi*(3760.*SL + 1490*dSigmaL)/3. + pnss3

def phi6l(m1, m2, a1L, a2L, a1sq, a2sq, a1dota2):
  return -6848./21. * np.ones(len(m1))

def phi7(m1, m2, a1L, a2L, a1sq, a2sq, a1dota2):
  m1M = m1/(m1+m2)
  m2M = m2/(m1+m2)
  d = (m1-m2)/(m1+m2)
  eta = (m1*m2)/(m1+m2)**2.
  SL = m1M * m1M * a1L + m2M * m2M * a2L
  dSigmaL = d * (m2M * a2L - m1M * a1L)
  return np.pi*(77096675./254016. + 378515./1512.*eta - 74045./756.*eta*eta) + (-8980424995./762048. + 6586595.*eta/756. - 305.*eta*eta/36.)* SL - (170978035./48384. - 2876425.*eta/672. - 4735.*eta*eta/144.)* dSigmaL

def phiMinus2(m1, m2, a1L, a2L, a1sq, a2sq, a1dota2):
  return np.ones(len(m1))

def phi3PNSS(m1, m2, a1L, a2L, a1sq, a2sq, a1dota2):
  qm_def1 = 1
  qm_def2 = 1
  m1M     = m1/(m1+m2)
  m2M     = m2/(m1+m2)
  d       = (m1-m2)/(m1+m2)
  eta     = (m1*m2)/(m1+m2)**2.
  SL      = m1M * m1M * a1L + m2M * m2M * a2L
  dSigmaL = d * (m2M * a2L - m1M * a1L)
  pnss3 = (326.75/1.12 + 557.5/1.8*eta) * eta * a1L * a2L + ((4703.5/8.4 + 2935./6. * m1M - 120. * m1M * m1M)*(qm_def1) + (-4108.25/6.72 - 108.5/1.2*m1M + 125.5/3.6*m1M*m1M))*m1M*m1M* a1sq + ((4703.5/8.4 + 2935./6. * m2M - 120. * m2M * m2M)*(qm_def2) + (-4108.25/6.72 - 108.5/1.2*m2M + 125.5/3.6*m2M*m2M))*m2M*m2M* a2sq
  return pnss3


#The functions phi${N}NS return the spin-independent component of the coefficient of the N/2-PN term in the inspiral

def phi0NS(m1,m2):
  return phi0(m1,m2,np.zeros(len(m1)),np.zeros(len(m1)),np.zeros(len(m1)),np.zeros(len(m1)),np.zeros(len(m1)))

def phi1NS(m1,m2):
  return phi1(m1,m2,np.zeros(len(m1)),np.zeros(len(m1)),np.zeros(len(m1)),np.zeros(len(m1)),np.zeros(len(m1)))

def phi2NS(m1,m2):
  return phi2(m1,m2,np.zeros(len(m1)),np.zeros(len(m1)),np.zeros(len(m1)),np.zeros(len(m1)),np.zeros(len(m1)))

def phi3NS(m1,m2):
  return phi3(m1,m2,np.zeros(len(m1)),np.zeros(len(m1)),np.zeros(len(m1)),np.zeros(len(m1)),np.zeros(len(m1)))

def phi4NS(m1,m2):
  return phi4(m1,m2,np.zeros(len(m1)),np.zeros(len(m1)),np.zeros(len(m1)),np.zeros(len(m1)),np.zeros(len(m1)))

def phi5lNS(m1,m2):
  return phi5l(m1,m2,np.zeros(len(m1)),np.zeros(len(m1)),np.zeros(len(m1)),np.zeros(len(m1)),np.zeros(len(m1)))

def phi6NS(m1,m2):
  return phi6(m1,m2,np.zeros(len(m1)),np.zeros(len(m1)),np.zeros(len(m1)),np.zeros(len(m1)),np.zeros(len(m1)))

def phi6lNS(m1,m2):
  return phi6l(m1,m2,np.zeros(len(m1)),np.zeros(len(m1)),np.zeros(len(m1)),np.zeros(len(m1)),np.zeros(len(m1)))

def phi7NS(m1,m2):
  return phi7(m1,m2,np.zeros(len(m1)),np.zeros(len(m1)),np.zeros(len(m1)),np.zeros(len(m1)),np.zeros(len(m1)))

def phiMinus2NS(m1,m2):
  return phiMinus2(m1,m2,np.zeros(len(m1)),np.zeros(len(m1)),np.zeros(len(m1)),np.zeros(len(m1)),np.zeros(len(m1)))

#The functions phi${N}S return the spin-dependent component of the coefficient of the N/2-PN term in the inspiral

def phi0S(m1, m2, a1L, a2L, a1sq, a2sq, a1dota2):
  return phi0(m1, m2, a1L, a2L, a1sq, a2sq, a1dota2) - phi0NS(m1, m2)

def phi1S(m1, m2, a1L, a2L, a1sq, a2sq, a1dota2):
  return phi1(m1, m2, a1L, a2L, a1sq, a2sq, a1dota2) - phi1NS(m1, m2)

def phi2S(m1, m2, a1L, a2L, a1sq, a2sq, a1dota2):
  return phi2(m1, m2, a1L, a2L, a1sq, a2sq, a1dota2) - phi2NS(m1, m2)

def phi3S(m1, m2, a1L, a2L, a1sq, a2sq, a1dota2):
  return phi3(m1, m2, a1L, a2L, a1sq, a2sq, a1dota2) - phi3NS(m1, m2)

def phi4S(m1, m2, a1L, a2L, a1sq, a2sq, a1dota2):
  return phi4(m1, m2, a1L, a2L, a1sq, a2sq, a1dota2) - phi4NS(m1, m2)

def phi5lS(m1, m2, a1L, a2L, a1sq, a2sq, a1dota2):
  return phi5l(m1, m2, a1L, a2L, a1sq, a2sq, a1dota2) - phi5lNS(m1, m2)

def phi6S(m1, m2, a1L, a2L, a1sq, a2sq, a1dota2):
  return phi6(m1, m2, a1L, a2L, a1sq, a2sq, a1dota2) - phi6NS(m1, m2)

def phi6lS(m1, m2, a1L, a2L, a1sq, a2sq, a1dota2):
  return phi6l(m1, m2, a1L, a2L, a1sq, a2sq, a1dota2) - phi6lNS(m1, m2)

def phi7S(m1, m2, a1L, a2L, a1sq, a2sq, a1dota2):
  return phi7(m1, m2, a1L, a2L, a1sq, a2sq, a1dota2) - phi7NS(m1, m2)

def phiMinus2S(m1, m2, a1L, a2L, a1sq, a2sq, a1dota2):
  return phiMinus2(m1, m2, a1L, a2L, a1sq, a2sq, a1dota2) - phiMinus2NS(m1, m2)

#Dictionaries that map the testing-GR parameter of each run to the corresponding function above
phiDict = {'dchi0':phi0, 'dchi1':phi1, 'dchi2':phi2, 'dchi3':phi3, 'dchi4':phi4, 'dchi5l':phi5l, 'dchi6':phi6, 'dchi6l':phi6l, 'dchi7':phi7, 'dchiMinus2':phiMinus2, 'dipolecoeff':phiMinus2}
phiNSDict = {'dchi0':phi0NS, 'dchi1':phi1NS, 'dchi2':phi2NS, 'dchi3':phi3NS, 'dchi4':phi4NS, 'dchi5l':phi5lNS, 'dchi6':phi6NS, 'dchi6l':phi6lNS, 'dchi7':phi7NS, 'dchiMinus2':phiMinus2NS, 'dipolecoeff':phiMinus2NS}
phiSDict = {'dchi0':phi0S, 'dchi1':phi1S, 'dchi2':phi2S, 'dchi3':phi3S, 'dchi4':phi4S, 'dchi5l':phi5lS, 'dchi6':phi6S, 'dchi6l':phi6lS, 'dchi7':phi7S, 'dchiMinus2':phiMinus2S, 'dipolecoeff':phiMinus2S}

zs_ = np.linspace(0, 2.01, 1000)
p_z = euclidean_distance_prior(dict(redshift=zs_))
p_z /= np.trapz(p_z, zs_)
interpolated_p_z = interp1d(zs_, p_z)

param = sys.argv[1]

outdir = sys.argv[2]
o3a_data_dir = sys.argv[3]
o3b_data_dir = sys.argv[4]

interested_keys = ['mass_1_source', 'mass_2_source', 'mass_ratio', 'cos_tilt_1', 'cos_tilt_2', 'a_1', 'a_2', 'redshift', param, 'prior']
interested_keys_o3b = ['mass_1', 'mass_2', 'mass_ratio', 'cos_tilt_1', 'cos_tilt_2', 'a_1', 'a_2', 'redshift', param, 'prior']
interested_keys2 = ['mass_1_source', 'mass_2_source', 'mass_ratio', 'cos_tilt_1', 'cos_tilt_2', 'a_1', 'a_2', 'redshift', "dphi", 'prior']


interested_keys_phenom = ['mass_1_source', 'mass_2_source', 'mass_ratio', 'cos_tilt_1', 'cos_tilt_2', 'a_1', 'a_2', 'redshift', param, 'log_prior']
interested_keys2_phenom = ['mass_1_source', 'mass_2_source', 'mass_ratio', 'cos_tilt_1', 'cos_tilt_2', 'a_1', 'a_2', 'redshift', "dphi", 'prior']

O3a_files = glob(f'{o3a_data_dir}/*seob_*{param}.h5')
O3b_files = glob(f'{o3b_data_dir}/*seob*{param}.h5')

print(len(O3a_files)+len(O3b_files))

if not os.path.exists(f'{outdir}/{param}/O3a/'):
    os.makedirs(f'{outdir}/{param}/O3a/')

if not os.path.exists(f'{outdir}/{param}/O3b/'):
    os.makedirs(f'{outdir}/{param}/O3b/')

fig, ax = plt.subplots(nrows=29, ncols=2, figsize=[4.1*2, 4.1*29], gridspec_kw = {'hspace':0.3})

align_spin_prior = bilby.gw.prior.AlignedSpin()

for i, file in tqdm(enumerate(O3a_files)):
    filename = file.split('/')[-1].split('.')[0]
    
    sample_dict = {
        'tgr':{'posterior_samples':{}}
    }
    
    data = h5py.File(file)
    
    data_name = None
    for key in data.keys():
        if param in key:
            data_name = key 
            break
    
    arr_df = pd.DataFrame(np.array(data[data_name]['posterior_samples']))
    m1 = np.array(arr_df[interested_keys[0]].values); m2 = np.array(arr_df[interested_keys[1]].values)
    a1z = np.array(arr_df[interested_keys[3]].values * arr_df[interested_keys[5]].values); a1sq = a1z**2
    a2z = np.array(arr_df[interested_keys[4]].values * arr_df[interested_keys[6]].values); a2sq = a2z**2
    a1dota2 = a1z * a2z
    
    factor = (1. + phiSDict[param](m1,m2,a1z,a2z,a1sq,a2sq,a1dota2)/phiNSDict[param](m1,m2))
    
    arr_df[param] = arr_df[param] * factor
    arr_df['weight'] = np.abs(factor)/np.sum(np.abs(factor))
    
    # Sampling prior
    arr_df['prior'] = align_spin_prior.prob(arr_df['a_1'])*align_spin_prior.prob(arr_df['a_2'])
    arr_df['prior'] *= interpolated_p_z(arr_df["redshift"])
    arr_df['prior'] *= arr_df["mass_1"]
    
    new_arr = arr_df.sample(n=10000, replace=True, weights=arr_df['weight']).reset_index(drop=True)[interested_keys]
    ds_dt = np.dtype({'names':interested_keys,'formats':[(float)]*len(interested_keys)}) 
    posterior_samples = np.rec.fromarrays([np.array(new_arr)[:,i] for i in range(len(interested_keys))], dtype=ds_dt)
    posterior_samples.dtype.names = interested_keys2
    
    sample_dict['tgr']['posterior_samples'] = posterior_samples
    
    with h5py.File(f'{outdir}/{param}/O3a/{filename}_noAstroWeight.h5', 'w') as h5file:
        h5file.create_dataset('tgr/posterior_samples', data=posterior_samples, dtype=posterior_samples.dtype)
        
    # Calculation for the astro reweighted samples to significantly increase efficiency
    astro_prior = arr_df['mass_1_source']**(-4)
    astro_prior *= np.exp(-(arr_df['a_1'] - 0.3)**2/(2*0.3**2))
    astro_prior *= np.exp(-(arr_df['a_2'] - 0.3)**2/(2*0.3**2))
    astro_prior *= interpolated_p_z(arr_df["redshift"]) * (1+arr_df["redshift"])**3
    
    astro_weight = astro_prior/arr_df['prior']
    arr_df['prior'] = astro_prior
    
    new_arr2 = arr_df.sample(n=10000, replace=True, weights=arr_df['weight']*astro_weight).reset_index(drop=True)[interested_keys]
    
    ds_dt = np.dtype({'names':interested_keys,'formats':[(float)]*len(interested_keys)}) 
    posterior_samples = np.rec.fromarrays([np.array(new_arr2)[:,i] for i in range(len(interested_keys))], dtype=ds_dt)
    posterior_samples.dtype.names = interested_keys2
    
    sample_dict['tgr']['posterior_samples'] = posterior_samples
    
    with h5py.File(f'{outdir}/{param}/O3a/{filename}_weights.h5', 'w') as h5file:
        h5file.create_dataset('tgr/posterior_samples', data=posterior_samples, dtype=posterior_samples.dtype)
    
    fig3, axe = plt.subplots(nrows=1, ncols=1)
    
    axe.hist(new_arr[param], bins=50, histtype='step', label='No astro weight')
    axe.hist(new_arr2[param], bins=50, histtype='step', label='Approx astro weight')
    axe.legend()
    
    fig3.savefig(f'{outdir}/{param}/O3a/{filename}_dphi.pdf', bbox_inches='tight')
    
    
    ax[i][0].hist2d(posterior_samples['mass_1_source']+posterior_samples['mass_2_source'], posterior_samples['dphi'], bins=50)
    ax[i][0].set_ylabel(param)
    ax[i][0].set_xlabel('total mass')
    ax[i][0].set_title(filename)
    ax[i][1].hist2d(posterior_samples['mass_ratio'], posterior_samples['dphi'], bins=50)
    ax[i][1].set_xlabel('q')
    
redshift_interp1d = interp1d(Planck15.luminosity_distance(np.linspace(0,2,200)).value, np.linspace(0,2,200))

for i, file in tqdm(enumerate(O3b_files)):
    filename = file.split('/')[-1].split('.')[0]
    
    sample_dict = {
        'tgr':{'posterior_samples':{}}
    }
    
    data = h5py.File(file)
    
    data_name = None
    for key in data.keys():
        if param in key:
            data_name = key 
            break
            
    arr_df = pd.DataFrame(np.array(data[data_name]['posterior_samples']))
    m1 = np.array(arr_df[interested_keys_o3b[0]].values); m2 = np.array(arr_df[interested_keys_o3b[1]].values)
    a1z = np.array(arr_df[interested_keys_o3b[3]].values * arr_df[interested_keys_o3b[5]].values); a1sq = a1z**2
    a2z = np.array(arr_df[interested_keys_o3b[4]].values * arr_df[interested_keys_o3b[6]].values); a2sq = a2z**2
    a1dota2 = a1z * a2z
    
    factor = (1. + phiSDict[param](m1,m2,a1z,a2z,a1sq,a2sq,a1dota2)/phiNSDict[param](m1,m2))
    
    arr_df[param] = arr_df[param] * factor
    arr_df['weight'] = np.abs(factor)/np.sum(np.abs(factor))
    
    redshift = redshift_interp1d(arr_df['marginalized_distance'])
    mass_1_source = arr_df['mass_1']/(1+redshift)
    mass_2_source = arr_df['mass_2']/(1+redshift)
    
    arr_df['redshift'] = redshift
    arr_df['mass_1_source'] = mass_1_source
    arr_df['mass_2_source'] = mass_2_source
    
    # Sampling prior
    arr_df['prior'] = align_spin_prior.prob(arr_df['a_1'])*align_spin_prior.prob(arr_df['a_2'])
    arr_df['prior'] *= interpolated_p_z(arr_df["redshift"])
    arr_df['prior'] *= arr_df["mass_1"]
    
    new_arr = arr_df.sample(n=10000, replace=True, weights=arr_df['weight']).reset_index(drop=True)[interested_keys]
    ds_dt = np.dtype({'names':interested_keys,'formats':[(float)]*len(interested_keys)}) 
    posterior_samples = np.rec.fromarrays([np.array(new_arr)[:,i] for i in range(len(interested_keys))], dtype=ds_dt)
    posterior_samples.dtype.names = interested_keys2
    
    sample_dict['tgr']['posterior_samples'] = posterior_samples
    
    with h5py.File(f'{outdir}/{param}/O3b/{filename}_noAstroWeight.h5', 'w') as h5file:
        h5file.create_dataset('tgr/posterior_samples', data=posterior_samples, dtype=posterior_samples.dtype)
        
    # Calculation for the astro reweighted samples to significantly increase efficiency
    astro_prior = arr_df['mass_1_source']**(-4)
    astro_prior *= np.exp(-(arr_df['a_1'] - 0.3)**2/(2*0.3**2))
    astro_prior *= np.exp(-(arr_df['a_2'] - 0.3)**2/(2*0.3**2))
    astro_prior *= interpolated_p_z(arr_df["redshift"]) * (1+arr_df["redshift"])**3
    
    astro_weight = astro_prior/arr_df['prior']
    arr_df['prior'] = astro_prior
    
    new_arr2 = arr_df.sample(n=10000, replace=True, weights=arr_df['weight']*astro_weight).reset_index(drop=True)[interested_keys]
    
    ds_dt = np.dtype({'names':interested_keys,'formats':[(float)]*len(interested_keys)}) 
    posterior_samples = np.rec.fromarrays([np.array(new_arr2)[:,i] for i in range(len(interested_keys))], dtype=ds_dt)
    posterior_samples.dtype.names = interested_keys2
    
    sample_dict['tgr']['posterior_samples'] = posterior_samples
    
    with h5py.File(f'{outdir}/{param}/O3b/{filename}_weights.h5', 'w') as h5file:
        h5file.create_dataset('tgr/posterior_samples', data=posterior_samples, dtype=posterior_samples.dtype)
        
    fig3, axe = plt.subplots(nrows=1, ncols=1)
    
    axe.hist(new_arr[param], bins=50, histtype='step', label='No astro weight')
    axe.hist(new_arr2[param], bins=50, histtype='step', label='Approx astro weight')
    axe.legend()
    
    fig3.savefig(f'{outdir}/{param}/O3a/{filename}_dphi.pdf', bbox_inches='tight')
        
    ax[i+20][0].hist2d(posterior_samples['mass_1_source']+posterior_samples['mass_2_source'], posterior_samples['dphi'], bins=50)
    ax[i+20][0].set_ylabel(param)
    ax[i+20][0].set_xlabel('total mass')
    ax[i+20][0].set_title(filename)
    ax[i+20][1].hist2d(posterior_samples['mass_ratio'], posterior_samples['dphi'], bins=50)
    ax[i+20][1].set_xlabel('q')
    
fig.savefig(f'{outdir}/{param}/hist2d.pdf', bbox_inches='tight')
    

O3a_files = glob(f'{o3a_data_dir}/*phenom_*{param}.h5')

allowed_events = ['par_S190728q', 'par_S190910s', 'par_S190503bf', 'par_S190408an', 'par_S190828l', 'par_S190512at', 'par_S190915ak', 'par_S190720a', 'par_S190602aq', 'par_S190517h', 'par_S190707q', 'par_S190521r', 'par_S190708ap', 'par_S190828j', 'par_S190924h', 'par_S190630ag', 'par_S190412m', 'par_S190513bm', 'par_S190727h', 'par_S191204r', 'par_S200129m', 'par_S200202ac', 'par_S200311bg', 'par_S200225q', 'par_S191216ap', 'par_S191129u', 'par_S200316bj']

O3a_files_true = []
for file in O3a_files:
    event_name_ls = file.split('/')[-1].split('.')[0].split('_')[:2]
    event_name = f'{event_name_ls[0]}_{event_name_ls[1]}'
    if event_name in allowed_events:
        O3a_files_true.append(file)
O3a_files = O3a_files_true

if not os.path.exists(f'{outdir}/{param}/O3a_phenom/'):
    os.makedirs(f'{outdir}/{param}/O3a_phenom/')

fig, ax = plt.subplots(nrows=len(O3a_files), ncols=3, figsize=[4.1*3, 4.1*len(O3a_files)], gridspec_kw = {'hspace':0.3})

if param == 'dchiMinus2':
    interested_keys_phenom = ['mass_1_source', 'mass_2_source', 'mass_ratio', 'cos_tilt_1', 'cos_tilt_2', 'a_1', 'a_2', 'redshift', 'dchimin2', 'log_prior']

for i, file in tqdm(enumerate(O3a_files)):
    filename = file.split('/')[-1].split('.')[0]
    
    sample_dict = {
        'tgr':{'posterior_samples':{}}
    }
    
    data = h5py.File(file)
    
    data_name = None
    for key in data.keys():
        if param in key:
            data_name = key 
            break
    
    try:
        arr_df = pd.DataFrame(np.array(data[data_name]['posterior_samples']))
        
        arr_df['log_prior'] = 1/4
        arr_df['log_prior'] *= interpolated_p_z(arr_df["redshift"])
        arr_df['log_prior'] *= arr_df["mass_1"]
        
        astro_prior = arr_df['mass_1_source']**(-4)
        astro_prior *= np.exp(-(arr_df['a_1'] - 0.3)**2/(2*0.3**2))
        astro_prior *= np.exp(-(arr_df['a_2'] - 0.3)**2/(2*0.3**2))
        astro_prior *= interpolated_p_z(arr_df["redshift"]) * (1+arr_df["redshift"])**3

        astro_weight = astro_prior/arr_df['log_prior']
        
        new_arr = arr_df.sample(10000).reset_index(drop=True)
        
        ds_dt = np.dtype({'names':interested_keys2_phenom,'formats':[(float)]*len(interested_keys_phenom)}) 
        posterior_samples = np.rec.fromarrays([np.array(new_arr[key]) for key in interested_keys_phenom], dtype=ds_dt)
        sample_dict['tgr']['posterior_samples'] = posterior_samples

        with h5py.File(f'{outdir}/{param}/O3a_phenom/{filename}_noAstroWeight.h5', 'w') as h5file:
           h5file.create_dataset('tgr/posterior_samples', data=np.asarray(sample_dict['tgr']['posterior_samples']))
        
        arr_df['log_prior'] = astro_prior
        
        new_arr = arr_df.sample(n=10000, replace=True, weights=astro_weight).reset_index(drop=True)[interested_keys_phenom]
        
        ds_dt = np.dtype({'names':interested_keys2_phenom,'formats':[(float)]*len(interested_keys_phenom)}) 
        posterior_samples = np.rec.fromarrays([np.array(new_arr[key]) for key in interested_keys_phenom], dtype=ds_dt)
        sample_dict['tgr']['posterior_samples'] = posterior_samples

        with h5py.File(f'{outdir}/{param}/O3a_phenom/{filename}_weights.h5', 'w') as h5file:
            h5file.create_dataset('tgr/posterior_samples', data=np.asarray(sample_dict['tgr']['posterior_samples']))

        ax[i][0].hist2d(posterior_samples['mass_1_source']+posterior_samples['mass_2_source'], posterior_samples['dphi'], bins=50)
        ax[i][0].set_ylabel(param)
        ax[i][0].set_xlabel('total mass')
        ax[i][0].set_title(filename)
        ax[i][1].hist2d(posterior_samples['mass_ratio'], posterior_samples['dphi'], bins=50)
        ax[i][1].set_xlabel('q')
        ax[i][2].hist2d(posterior_samples['cos_tilt_1'], posterior_samples['dphi'], bins=50)
        ax[i][2].set_xlabel(r'$\cos\theta_1$')

    except:
        data_name = None
        for key in data['posterior_samples'].keys():
            if param in key:
                data_name = key 
                break
        
        
        names = [n.decode() for n in data['posterior_samples'][data_name]['parameter_names'][()]]
        print(names)
        arr = data['posterior_samples'][data_name]['samples'][()]
        
        df_dict = {}
        arr_samples = []
        for key in interested_keys_phenom:
            print(key)
            if key != 'dphi':
                matched_indexes = []
                k=0
                while k < len(names):
                    if key == names[k]:
                        matched_indexes.append(k)
                    k += 1
                
                df_dict[key] = arr[:,matched_indexes[0]]
            else:
                matched_indexes = []
                k=0
                while k < len(names):
                    if param == names[k]:
                        matched_indexes.append(k)
                        break
                    k += 1
                
                df_dict[key] = arr[:,matched_indexes[0]]
        
        
        pd_df = pd.DataFrame.from_dict(df_dict)
        pd_df['prior'] = pd_df['log_prior']
        pd_df['dphi'] = pd_df[param]
        pd_df.pop(param)
                                 
        pd_df['prior'] = 1/4
        pd_df['prior'] *= interpolated_p_z(pd_df["redshift"])
        pd_df['prior'] *= pd_df["mass_1_source"] * (1+pd_df['redshift'])**2
        
        astro_prior = pd_df['mass_1_source']**(-4)
        astro_prior *= np.exp(-(arr_df['a_1'] - 0.3)**2/(2*0.3**2))
        astro_prior *= np.exp(-(arr_df['a_2'] - 0.3)**2/(2*0.3**2))
        astro_prior *= interpolated_p_z(pd_df["redshift"]) * (1+pd_df["redshift"])**3

        astro_weight = astro_prior/pd_df['prior']
        
        arr_df = pd_df.sample(10000)[interested_keys2_phenom].reset_index(drop=True)
        
        ds_dt = np.dtype({'names':interested_keys2_phenom,'formats':[(float)]*len(interested_keys_phenom)}) 
        posterior_samples = np.rec.fromarrays([np.array(arr_df[key]) for key in interested_keys2_phenom], dtype=ds_dt)
        sample_dict['tgr']['posterior_samples'] = posterior_samples

        with h5py.File(f'{outdir}/{param}/O3a_phenom/{filename}_noAstroWeight.h5', 'w') as h5file:
            h5file.create_dataset('tgr/posterior_samples', data=np.asarray(sample_dict['tgr']['posterior_samples']))
                                 
        pd_df['prior'] = astro_prior
        
        new_arr_df = pd_df.sample(n=10000, replace=True, weights=astro_weight).reset_index(drop=True)[interested_keys2_phenom]
        
        ds_dt = np.dtype({'names':interested_keys2_phenom,'formats':[(float)]*len(interested_keys_phenom)}) 
        posterior_samples = np.rec.fromarrays([np.array(new_arr_df[key]) for key in interested_keys2_phenom], dtype=ds_dt)
        sample_dict['tgr']['posterior_samples'] = posterior_samples
        print(np.asarray(sample_dict['tgr']['posterior_samples']))
        print(np.asarray(sample_dict['tgr']['posterior_samples']).dtype)

        with h5py.File(f'{outdir}/{param}/O3a_phenom/{filename}_weights.h5', 'w') as h5file:
            h5file.create_dataset('tgr/posterior_samples', data=np.asarray(sample_dict['tgr']['posterior_samples']))
            posterior_samples = h5file['tgr']['posterior_samples']
            
            ax[i][0].hist2d(posterior_samples['mass_1_source']+posterior_samples['mass_2_source'], posterior_samples['dphi'], bins=50)
            ax[i][0].set_ylabel(param)
            ax[i][0].set_xlabel('total mass')
            ax[i][0].set_title(filename)
            ax[i][1].hist2d(posterior_samples['mass_ratio'], posterior_samples['dphi'], bins=50)
            ax[i][1].set_xlabel('q')
            ax[i][2].hist2d(posterior_samples['cos_tilt_1'], posterior_samples['dphi'], bins=50)
            ax[i][2].set_xlabel(r'$\cos\theta_1$')

fig.savefig(f'{outdir}/{param}/hist2d_phenom.pdf', bbox_inches='tight')