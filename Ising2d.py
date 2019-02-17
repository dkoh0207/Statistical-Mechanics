import numpy as np
import matplotlib.pyplot as plt
import random
import queue

import numpy as np
import matplotlib.pyplot as plt
import random
import queue

class Lattice:

    def __init__(self, lsize, K=1.0):

        self.lattice = np.random.rand(lsize, lsize)
        self.lattice = np.round(self.lattice)
        self.lattice[self.lattice == 0] = 1
        self.idx = 1
        self.idy = 1
        self.K = K # Dimensionless Temperature
        self.lsize = lsize # Lattice Size
        self.prob = 1 - np.exp(-1 * self.K) # Acceptance probability for Wolff
        self.w = {key:np.exp(0) for key in [-8, -4, 0, 4, 8]}
        # This ensures that all flips towards negative
        # energy configuration have flip probability = 1.
        self.w[-8] = np.exp(-K * 8.0) # Adjust 
        self.w[-4] = np.exp(-K*4.0)
        self.M = 0
        self.M2 = 0
        self.absM = 0
        self.E = 0
        self.E2 = 0


    def reset(self):
        self.M = 0
        self.M2 = 0
        self.absM = 0
        self.E = 0
        self.E2 = 0
        

    def get_index(self, idx, idy):
        idx = (idx + self.lsize) % self.lsize  
        idy = (idy + self.lsize) % self.lsize
        return idx, idy


    def at(self, idx, idy):
        return self.lattice[self.get_index(idx, idy)]
    

    def get_neighbor_indices(self, site):
        return [self.get_index(site[0], site[1]+1), self.get_index(site[0], site[1]-1),
           self.get_index(site[0]+1, site[1]), self.get_index(site[0]-1, site[1])]


    def metropolis(self):
        '''
        Implements single metropolis pass.
        '''
        site_id = np.random.randint(0, self.lsize), np.random.randint(0, self.lsize)
        deltaE = 0
        for n in self.get_neighbor_indices(site_id):
            deltaE += self.at(site_id[0], site_id[1]) * self.at(n[0], n[1])
        deltaE = int(-2 * deltaE)
        if random.random() < self.w[deltaE]:
            self.lattice[site_id[0], site_id[1]] *= -1
    

    def metropolis_pass(self):
        '''
        Implements metropolis pass.
        '''
        mcs = self.lsize ** 2 * 100
        avg_M = []
        avg_M2 = []
        m = 0
        m2 = 0
        for i in range(mcs):
            self.metropolis()
            avg_M.append(self.M)
            avg_M2.append(self.M2)
        m = np.mean(avg_M)
        m2 = np.mean(avg_M2) - (np.mean(avg_M))**2.0
        print("M = {0}".format(m))
        print("Sigma(M) = {0}".format(m2))
        return m, m2


    def find_clusters(self):
        cluster = set([])
        unprocessed_sites = queue.Queue()
        site_id = np.random.randint(0, self.lsize), np.random.randint(0, self.lsize)
        unprocessed_sites.put(site_id)
        while not unprocessed_sites.empty():
            site_id = unprocessed_sites.get()
            neighbors = self.get_neighbor_indices(site_id)
            for n in neighbors:
                site_n = self.at(n[0], n[1])
                site_center = self.at(site_id[0], site_id[1])
                prob = 1 - np.exp(-self.K * site_n * site_center)
                if random.random() < prob:
                    if not n in cluster:
                        unprocessed_sites.put(n)
                    cluster.add(n)
        return cluster
    

    def wolff_pass(self):
        
        mcs = self.lsize ** 2

        for i in range(mcs):
            clusters = self.find_clusters()
            for c in clusters:
                self.lattice[c[0], c[1]] *= -1
        print("M = ", self.M)
        print("Sigma(M) = ", self.M2 - (self.M) ** 2.0)



    @property
    def M(self):
        return np.sum(self.lattice) / (self.lsize * self.lsize)
    
    @property
    def M2(self):
        return np.sum(np.power(self.lattice, 2)) / (self.lsize * self.lsize)

