. import models, utils

from glob import glob
import h5py
import pandas as pd
import numpy as np

import jax
import numpyro
from numpyro.infer import MCMC, NUTS, init_to_median
import arviz as az

import matplotlib.pyplot as plt

import sys
import os

numpyro.enable_x64()

# run settings
n_warmup = 1000
n_sample = 2000
n_chains = 6


param = sys.argv[1]
waveform = sys.argv[2]

outdir = sys.argv[3]
data_dir = sys.argv[4]
inj_dir = sys.argv[5]

allowed_events = ['S190728q', 'S190408an', 'S190828l', 'S190512at', 'S190720a', 
                  'S190707q', 'S190521r', 'S190708ap', 'S190828j', 'S190924h', 
                  'S190630ag', 'S190412m', 'S191204r', 'S200129m', 'S200202ac', 
                  'S200311bg', 'S200225q', 'S191216ap', 'S191129u', 'S200316bj']

# Generate the output directory
outdir = f'{outdir}/{waveform}/{param}/'
if not os.path.exists(outdir):
    os.makedirs(outdir)
    
if waveform == 'seob':
    injection_file = f'{inj_dir}/o3b_bbh_injs_inspiral_only.hdf5'
    event_files = glob(f'{data_dir}/{param}/O3a/*_weights.h5') + glob(f'{data_dir}/{param}/O3b/*_weights.h5')
    event_files_tgr = glob(f'{data_dir}/{param}/O3a/*_noAstroWeight.h5') + glob(f'{data_dir}/{param}/O3b/*_noAstroWeight.h5')
    
    use_tilts = False

elif waveform == 'phenom':
    injection_file = f'{inj_dir}/o3a_bbh_injs_inspiral_only.hdf5'
    event_files = glob(f'{data_dir}/{param}/O3a_phenom/*_weights.h5')
    event_files_tgr = glob(f'{data_dir}/{param}/O3a_phenom/*_noAstroWeight.h5')
    
    use_tilts = True
else:
    raise ValueError('No Samples for selected waveform')

event_posteriors = []
for filename in event_files:
    
    for event in allowed_events:
        if event in filename:
            event_posteriors.append(pd.read_hdf(filename, 'tgr/posterior_samples'))
    
event_posteriors_tgr = []
for filename in event_files_tgr:
    
    for i, event in enumerate(allowed_events):
        if event in filename:
            data = pd.read_hdf(filename, 'tgr/posterior_samples')
            event_posteriors_tgr.append(data)

# TGR ONLY MODEL
dphis, bws_tgr, Nobs = models.generate_tgr_only_data(event_posteriors_tgr)

kernel = NUTS(models.make_tgr_only_model, find_heuristic_step_size=True, target_accept_prob=0.8, regularize_mass_matrix=False)
mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_sample, num_chains=n_chains)
mcmc.run(jax.random.PRNGKey(np.random.randint(1<<32)), dphis, bws_tgr, Nobs)

fit = az.from_numpyro(mcmc)

fname = f'{outdir}/result_tgr.nc'
fit.to_netcdf(fname)
print(f"Saved: {fname}")

# JOINT MODEL
event_data_array, injection_data_array, BW_matrices, BW_matrices_sel, Nobs, Ndraw = \
    models.generate_data(event_posteriors, injection_file, use_tilts=use_tilts, use_tgr=True)

kernel = NUTS(models.make_joint_model, find_heuristic_step_size=True, target_accept_prob=0.8, regularize_mass_matrix=False)
mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_sample, num_chains=n_chains)
mcmc.run(jax.random.PRNGKey(np.random.randint(1<<32)), event_data_array, injection_data_array, BW_matrices, BW_matrices_sel, Nobs, Ndraw, use_tilts, True)

fit = az.from_numpyro(mcmc)
    
fname = f'{outdir}/result_joint.nc'
fit.to_netcdf(fname)
print(f"Saved: {fname}")

# ASTRO ONLY MODEL
event_data_array, injection_data_array, BW_matrices, BW_matrices_sel, Nobs, Ndraw = \
    models.generate_data(event_posteriors, injection_file, use_tilts=use_tilts, use_tgr=False)

kernel = NUTS(models.make_joint_model, find_heuristic_step_size=True, target_accept_prob=0.8, regularize_mass_matrix=False)
mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_sample, num_chains=n_chains)
mcmc.run(jax.random.PRNGKey(np.random.randint(1<<32)), event_data_array, injection_data_array, BW_matrices, BW_matrices_sel, Nobs, Ndraw, use_tilts, False)

fit = az.from_numpyro(mcmc)
    
fname = f'{outdir}/result_astro.nc'
fit.to_netcdf(fname)
print(f"Saved: {fname}")

# GR ONLY MODEL
event_data_array, injection_data_array, BW_matrices, BW_matrices_sel, Nobs, Ndraw = \
    models.generate_data(event_posteriors, injection_file, use_tilts=use_tilts, use_tgr=True)

kernel = NUTS(models.make_gr_astro_model, find_heuristic_step_size=True, target_accept_prob=0.8, regularize_mass_matrix=False)
mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_sample, num_chains=n_chains)
mcmc.run(jax.random.PRNGKey(np.random.randint(1<<32)), event_data_array, injection_data_array, BW_matrices, BW_matrices_sel, Nobs, Ndraw, use_tilts)

fit = az.from_numpyro(mcmc)
    
fname = f'{outdir}/result_gr.nc'
fit.to_netcdf(fname)
print(f"Saved: {fname}")
