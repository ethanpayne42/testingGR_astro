from . import models, utils

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

outdir = sys.argv[1]
data_dir = sys.argv[2]
inj_dir = sys.argv[3]

# run settings
n_warmup = 1000
n_sample = 2000
n_chains = 6


allowed_events = ['S190408an', 'S190412m', 'S190421ar', 'S190503bf', 'S190512at', 'S190513bm',
                  'S190517h', 'S190519bj', 'S190521r', 'S190602aq', 'S190630ag', 'S190706ai',
                  'S190707q', 'S190708ap', 'S190720a', 'S190727h', 'S190728q', 'S190828j', 
                  'S190828l', 'S190910s', 'S190915ak', 'S190924h', # Now time for the O3b events
                  'S191129u', 'S191204r', 'S191215w', 'S191216ap', 'S191222n', 'S200129m', 'S200202ac',
                  'S200208q', 'S200219ac', 'S200224ca', 'S200225q', 'S200311bg']

# Generate the output directory
if not os.path.exists(outdir):
    os.makedirs(outdir)

injection_file = f'{inj_dir}/o3_bbh_injs.hdf5'
event_files = glob(f'{data_dir}/O3a/*_weights.h5') + glob(f'{data_dir}/O3b/*_weights.h5')
event_files_tgr = glob(f'{data_dir}/O3a/*_noAstroWeight.h5') + glob(f'{data_dir}/O3b/*_noAstroWeight.h5')

use_tilts = True

event_posteriors = []
for filename in event_files:
    
    for event in allowed_events:
        if event in filename:
            event_posteriors.append(pd.read_hdf(filename, 'tgr/posterior_samples'))
    
event_posteriors_tgr = []
for filename in event_files_tgr:
    
    for i, event in enumerate(allowed_events):
        if event in filename:
            #print(filename)
            data = pd.read_hdf(filename, 'tgr/posterior_samples')
            event_posteriors_tgr.append(data)

# TGR ONLY MODEL
dphis, bws_tgr, Nobs = models.generate_tgr_only_data(event_posteriors_tgr)

print(dphis)
print(bws_tgr)

kernel = NUTS(models.make_tgr_only_graviton_model, find_heuristic_step_size=True, target_accept_prob=0.8, regularize_mass_matrix=False)
mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_sample, num_chains=n_chains)
mcmc.run(jax.random.PRNGKey(np.random.randint(1<<32)), dphis, bws_tgr, Nobs, True)

fit = az.from_numpyro(mcmc)

fname = f'{outdir}/result_tgr.nc'
fit.to_netcdf(fname)
print(f"Saved: {fname}")

kernel = NUTS(models.make_tgr_only_graviton_model, find_heuristic_step_size=True, target_accept_prob=0.8, regularize_mass_matrix=False)
mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_sample, num_chains=n_chains)
mcmc.run(jax.random.PRNGKey(np.random.randint(1<<32)), dphis, bws_tgr, Nobs, False)

fit = az.from_numpyro(mcmc)

fname = f'{outdir}/result_tgr_nosigma.nc'
fit.to_netcdf(fname)
print(f"Saved: {fname}")

# JOINT MODEL
event_data_array, injection_data_array, BW_matrices, BW_matrices_sel, Nobs, Ndraw = \
    models.generate_data(event_posteriors, injection_file, use_tilts=use_tilts, use_tgr=True)

print(BW_matrices)

kernel = NUTS(models.make_joint_graviton_model, find_heuristic_step_size=True, target_accept_prob=0.8, regularize_mass_matrix=False)
mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_sample, num_chains=n_chains)
mcmc.run(jax.random.PRNGKey(np.random.randint(1<<32)), event_data_array, injection_data_array, BW_matrices, BW_matrices_sel, Nobs, Ndraw, use_tilts, True, False)

fit = az.from_numpyro(mcmc)
    
fname = f'{outdir}/result_joint_nosigma.nc'
fit.to_netcdf(fname)
print(f"Saved: {fname}")

kernel = NUTS(models.make_joint_graviton_model, find_heuristic_step_size=True, target_accept_prob=0.8, regularize_mass_matrix=False)
mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_sample, num_chains=n_chains)
mcmc.run(jax.random.PRNGKey(np.random.randint(1<<32)), event_data_array, injection_data_array, BW_matrices, BW_matrices_sel, Nobs, Ndraw, use_tilts, True, True)

fit = az.from_numpyro(mcmc)
    
fname = f'{outdir}/result_joint.nc'
fit.to_netcdf(fname)
print(f"Saved: {fname}")

# ASTRO ONLY MODEL
event_data_array, injection_data_array, BW_matrices, BW_matrices_sel, Nobs, Ndraw = \
    models.generate_data(event_posteriors, injection_file, use_tilts=use_tilts, use_tgr=False)

kernel = NUTS(models.make_joint_graviton_model, find_heuristic_step_size=True, target_accept_prob=0.8, regularize_mass_matrix=False)
mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_sample, num_chains=n_chains)
mcmc.run(jax.random.PRNGKey(np.random.randint(1<<32)), event_data_array, injection_data_array, BW_matrices, BW_matrices_sel, Nobs, Ndraw, use_tilts, False, False)

fit = az.from_numpyro(mcmc)
    
fname = f'{outdir}/result_astro.nc'
fit.to_netcdf(fname)
print(f"Saved: {fname}")
