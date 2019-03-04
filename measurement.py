import numpy as np
import matplotlib.pyplot as plt
import scipy
import pandas as pd
import os
import re
from collections import defaultdict, OrderedDict


class MCDataFrame:

    def __init__(self, path, size_multiplier=1, sampling_block=50,
                write=False, repeat=200, thermalization=1000):
        self.__path = path
        self.df, self.__mcs, self.__size = process_data_for_plotting(path,
        size_multiplier, sampling_block, write, repeat, thermalization)

    @property
    def size(self):
        return self.__size

    @property
    def mcs(self):
        return self.__mcs

    def __repr__(self):
        return "MCData at '{0}', MCS = {1}, SIZE = {2}".format(
            self.__path, self.__mcs, self.__size)


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


def process_data_for_plotting(path, size_multiplier=1, sampling_block=50,
                              write=False, repeat=200, thermalization=2000):
    '''
    Run over all CSV datafiles inside directory and preprocess
    data for use in plotting.

    Inputs:
        path: path to CSV datafiles

        system_size: Size of the system (L^3 for 3D Ising, L^3 * 3 for 3D LGT)

    '''
    mcs, lsize, sweep = process_meta_info(path)
    system_size = lsize * lsize * lsize * size_multiplier
    k, beta = 0, 0
    rows_list = []
    files = [f for f in os.listdir(path) if re.match(r'cold.*\.csv', f)]
    for fname in files:
        data_dict = {}
        index = re.findall(r"[0-9]+", fname)
        if not index:
            print("Bad Filename?")
            print(fname)
            raise NameError
        index = int(index[0])
        fname = path + '/' + str(fname)
        beta = sweep.at[index, 'K']
        data_dict['K'] = beta
        data_dict['T'] = 1/beta  # We set k_B = 1
        estimates = bootstrap(fname, system_size, beta, repeat, thermalization)
        data_dict.update(estimates)
        rows_list.append(data_dict)
    df = pd.DataFrame(rows_list)
    df = df.sort_values('T')

    return df, mcs, system_size


def process_meta_info(path):

    mcs, lsize = 0, 0
    k, t, l = [], [], []
    readme = path + "/readme.txt"
    with open(readme, 'r') as f:
        text = f.readline().strip()
        l = re.findall('Number of MCS: ([0-9]+)', text)
        text = f.readline().strip()
        l += re.findall('Lattice Size: ([0-9]+)', text)
    assert len(l) == 2
    mcs, lsize = int(l[0]), int(l[1])
    df = pd.read_csv(readme, skiprows=2)

    return mcs, lsize, df


def autocorrelation(name, t, df_raw):
    '''
    Calculate autocorrelation function of a variable "name".
    '''
    df_raw = df_raw.reset_index(drop=True)
    tmax = len(df_raw)
    assert t < tmax
    chi = 0
    sum1, sum2, sum3 = 0, 0, 0
    for i in range(0, tmax - t):
        sum1 += df_raw[name][i] * df_raw[name][i + t]
        sum2 += df_raw[name][i]
        sum3 += df_raw[name][i + t]
    norm = 1.0 / (tmax - t)
    chi = norm * sum1 - (norm**2) * sum2 * sum3

    return chi

def autocorrelation_fft(name, df_raw):
    '''
    Computes the autocorrelation function for all times t
    using the Discrete Fourier Transform.
    '''
    n = len(df_raw)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
    f = np.asarray(df_raw[name])
    fvi = np.fft.fft(f, n=2*n)
    acf = np.real(np.fft.ifft(np.multiply(fvi, np.conjugate(fvi)))[:n])
    acf = acf / n
    return acf

def compute_integrated_autocorrelation(df_raw, name):
    '''
    Function for computing the integrated autocorrelation
    for a given time series (monte carlo run)
    '''
    const = 5
    df = df_raw.reset_index(drop=True)
    chi, tau = 0, 1
    m = 0
    var = np.var(df[name])
    for i in range(len(df)):
        chi = autocorrelation(name, i, df)
        if i == 0:
            print(chi)
            print(np.var(df[name]))
        tau += 2 * chi / var
        if i >= const * tau:
            m = i
            break
    print("Tau = {0}, M = {1}, X(0) = {2}".format(tau, m, var))
    return tau
    

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

    return df[sweep], df[obs_name], df[obs_name + 'err']

def generate_equilibration_plots(path, name):
    '''
    Generate time series plots for each temperature run for
    determinations of equilibriation. 
    '''
    cold_files = sorted([f for f in os.listdir(path) if re.match(r'cold.*\.csv', f)])
    hot_files = sorted([f for f in os.listdir(path) if re.match(r'hot.*\.csv', f)])
    print(cold_files)
    print(hot_files)
    for c, h in zip(cold_files, hot_files):
        path_c = path + "/" + c
        path_h = path + "/" + h
        df_cold = pd.read_csv(path_c, skipinitialspace=True)
        df_hot = pd.read_csv(path_h, skipinitialspace=True)
        fig, ax = plt.subplots()
        ax.errorbar(df_hot.index, df_hot[name], fmt='r-')
        ax.errorbar(df_cold.index, df_cold[name], fmt='b-')
        ax.grid(linestyle='--')
        ax.set_xlabel('Time (Monte Carlo step per site)', fontsize=15)
        ax.set_ylabel(name, fontsize=15)
        num = re.findall(r'.*_([0-9]+).csv', c)[0]
        print(num)
        fig.savefig(path + "/" + num + ".pdf")
        plt.close()