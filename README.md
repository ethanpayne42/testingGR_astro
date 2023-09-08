# Testing GR while additionally modeling the astrophysical population 

Code release for "Fortifying gravitational-wave tests of general relativity against astrophysical assumptions" - Ethan Payne, Max Isi, Katerina Chatziioannou, Will Farr

The code base is simply a series of scripts which allow for the joint hierarchicla inference of the astrophysical parameters in additional to parameters governing the deviations from GR. 

# Usage

This code contains simply two scripts to correctly format the LVK data for the analysis, and two to run all the different variations of the analysis. 

## Data collection and curation

First, the user must download the publicly available **full** posterior sample set for all the events desired. These samples must include the individual masses, spins, and redshift as well as the deviation parameters. Once the user has this (and installed the relevant libraries), the data formatted and saved in the correct format via
```
$python3 collect_PN_event_data.py <param[dchi0 etc]> <output directory> <o3a LVK posterior directory> <o3b LVK posterior directory>
```
and,
```
$python3 collect_graviton_event_data.py <data output directory> <o3a LVK posterior directory> <o3b LVK posterior directory>
```

## Running the analysis

Once the data are correctly formatted, the analysis can be run with the following simple commands: 
```
$python3 run_PN_analysis.py <param[dchi0 etc]> <waveform[seob or phenom]> <result output directory> <data output directory> <injection file directory>
```
for PN analyses, and for the massive graviton:
```
$python3 run_graviton_analysis.py <result output directory> <data output directory> <injection file directory>
```

# Citation 

Feel free to use any/all of this code, but please cite the corresponding paper upon its use: 

```
TODO
```
