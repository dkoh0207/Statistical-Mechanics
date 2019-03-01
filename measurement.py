import numpy as np
import matplotlib.pyplot as plt
import scipy
import pandas as pd
import os
import re
from collections import defaultdict, OrderedDict


def sample_events(df, system_size, beta, sampling_block=50):
    '''
    Sample one measurement from block sample data.
    '''
    c, susc = 0, 0
    sample = df.sample(sampling_block)
    sample = dict(sample.mean())
    # Heat Capacity
    if 'E' in sample and 'E2' in sample:
        c = sample['E2'] - sample['E']**2
        sample['C'] = beta**2 * system_size * c
    # Susceptibility
    if 'M' in sample and 'M2' in sample:
        susc = sample['M2'] - sample['M']**2
        sample['X'] = beta * system_size * susc

    return sample


def bootstrap(filename, system_size, beta, sampling_block=50,
              repeat=200, thermalization=1000):

    estimates = {}
    observables = defaultdict(list)
    df = pd.read_csv(filename, skipinitialspace=True)
    # Drop data taken before sufficient thermalization.
    df = df.iloc[thermalization:]
    for i in range(repeat):
        d = sample_events(df, system_size, beta, sampling_block)
        for name, val in d.items():
            observables[name].append(val)
    for name, vals in observables.items():
        estimates[name] = np.mean(vals)
        err = name + 'err'
        estimates[err] = np.std(vals)

    return estimates


def process_data_for_plotting(path, system_size, sampling_block=50,
                              offset=0, write=False, repeat=200, thermalization=2000):
    '''
    Run over all CSV datafiles inside directory and preprocess
    data for use in plotting.

    Inputs:
        path: path to CSV datafiles

        system_size: Size of the system (L^3 for 3D Ising, L^3 * 3 for 3D LGT)

        offset: Add an offset if the name of generated CSV files
        do not match actual K value of the Monte Carlo run.
    '''
    rows_list = []
    files = [f for f in os.listdir(path) if re.match(r'[0-9]+.*\.csv', f)]
    for fname in files:
        data_dict = {}
        index = re.findall(r"[0-9]+", fname)
        if not index:
            print("Bad Filename?")
            print(fname)
            raise NameError
        index = int(index[0])
        k = 0.01 * index + offset
        beta = 1.0 / k
        fname = path + str(fname)
        estimates = bootstrap(fname, system_size, beta, repeat, thermalization)
        data_dict['K'] = k
        data_dict.update(estimates)
        rows_list.append(data_dict)
    df = pd.DataFrame(rows_list)

    return df

def plot_observable(df, system_size, sweep, obs_name, fmt='ks'):
    '''
    Plot an observable (name) from trimmed data dictionary.
    '''
    fig, ax = plt.subplots()
    label = "L = {0}".format(system_size)
    ax.errorbar(df[sweep], df[obs_name], yerr=df[obs_name + 'err'],
        fmt=fmt, label=label, elinewidth=1, markersize=4, capsize=2)
    ax.grid(linestyle='--')
    ax.set_xlabel("${0}$".format(sweep), fontsize=14)
    ax.set_ylabel("${0}$".format(obs_name), fontsize=14)
    return fig

if __name__ == "__main__":
