import h5py
import numpy as np

import bilby
import pandas as pd

align_spin_prior = bilby.gw.prior.AlignedSpin()

import pytensor.tensor as pt

def read_injection_file(vt_file, ifar_threshold=1000, use_tilts=False):
    with h5py.File(vt_file, "r") as ff:
        data = ff["injections"]
        found = np.zeros_like(data["mass1_source"][()], dtype=bool)
        for key in data:
            if "ifar" in key.lower():
                found = found | (data[key][()] > ifar_threshold)
                
        n_found = np.sum(found)
        gwpop_data = dict(
            mass_1=np.asarray(data["mass1_source"][()][found]),
            mass_ratio=np.asarray(
                data["mass2_source"][()][found] / data["mass1_source"][()][found]
            ),
            redshift=np.asarray(data["redshift"][()][found]),
            total_generated=int(data.attrs["total_generated"][()]),
            analysis_time=data.attrs["analysis_time_s"][()] / 365.25 / 24 / 60 / 60,
        )
        for ii in [1, 2]:
            gwpop_data[f"a_{ii}"] = (
                np.asarray(
                    data.get(f"spin{ii}x", np.zeros(n_found))[()][found] ** 2
                    + data.get(f"spin{ii}y", np.zeros(n_found))[()][found] ** 2
                    + data[f"spin{ii}z"][()][found] ** 2
                )
                ** 0.5
            )
            
            gwpop_data[f"cos_tilt_{ii}"] = (
                np.asarray(data[f"spin{ii}z"][()][found]) / gwpop_data[f"a_{ii}"]
            )
            
            gwpop_data[f"a_{ii}z"] = np.abs(np.asarray(data[f"spin{ii}z"][()][found]))

        if use_tilts:
            gwpop_data["prior"] = (
                np.asarray(data["sampling_pdf"][()][found])
                * np.asarray(data["mass1_source"][()][found])
                * (2 * np.pi * gwpop_data["a_1"] ** 2) #* alignedspinz1
                * (2 * np.pi * gwpop_data["a_2"] ** 2) #* alignedspinz2 # TODO see jacob's messages
            )
        else:
            gwpop_data["prior"] = (
                np.asarray(data["sampling_pdf"][()][found])
                * np.asarray(data["mass1_source"][()][found])
                / np.asarray(data["spin1x_spin1y_spin1z_sampling_pdf"][()][found])
                / np.asarray(data["spin2x_spin2y_spin2z_sampling_pdf"][()][found])
                * align_spin_prior.prob(data["spin1z"][()][found])
                * align_spin_prior.prob(data["spin2z"][()][found])
            )
        
    return gwpop_data

def read_deviation_injection_file(file):
    gwpop_data = pd.read_csv(file, sep=' ')
    return gwpop_data

def pt_interp(x, xs, ys):
    """Linear interpolation: ``f(x; xs, ys)``"""
    x = pt.as_tensor(x)
    xs = pt.as_tensor(xs)
    ys = pt.as_tensor(ys)

    n = xs.shape[0]
    
    inds = pt.searchsorted(xs, x)
    inds = pt.where(inds <= 0, 1, inds)
    inds = pt.where(inds > n-1, n-1, inds)
    
    r = (x - xs[inds-1]) / (xs[inds] - xs[inds-1])

    return r*ys[inds] + (1-r)*ys[inds-1]