__author__ = 'milad'

import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['text.usetex'] = True
class Traffic(object):
    def __init__(self,filename,maxlen=None):
        self.noise_cache={}
        self.base_traffic={}
        self.maxlen=maxlen
        self.parse(filename)
        self.currentSamples=[]
        self.jitter_std={}

    def parse(self,filename):
        f = open('/home/milad/Research/Watermarking/Code/streamingTest/data/caida_parsed/'+filename)
        ind=0
        for l in f :
            ipd=[float(t) for t in l.split(',')][:self.maxlen]
            self.base_traffic[ind]=ipd
            ind+=1


    def makeNoisy(self,jitter,drop):
        if self.noise_cache.has_key((jitter,drop)):
            return self.noise_cache[(jitter,drop)]
        ipdafter={}
        noises=[]
        for ipd in self.currentSamples:
            cum=0.0
            for f in xrange(len(self.base_traffic[ipd])):

                if np.random.rand() < drop:
                    cum+=self.base_traffic[ipd][f]
                else:
                    noise=np.random.laplace(0,jitter)
                    noises.append(noise)
                    if cum+self.base_traffic[ipd][f]+noise >0.0:

                        ipdafter.setdefault(ipd,[]).append(cum+self.base_traffic[ipd][f]+noise)
                    else:

                        ipdafter.setdefault(ipd,[]).append(0.000001)
                    cum=0.0


        self.noise_cache[(jitter,drop)]=ipdafter
        self.jitter_std[(jitter,drop)]=np.std(noises)
        return ipdafter

    def __len__(self):
        return len(self.base_traffic)


    def clearCaches(self):
        self.noise_cache={}
        self.jitter_std={}

    def selectNewSamples(self,num):
        keys=self.base_traffic.keys()
        self.currentSamples=random.sample(keys,num)
        self.clearCaches()
    def plotthem(self):
        graphs=[]
        for i in self.currentSamples:
            for f in self.base_traffic[i]:
                graphs.append(f)
        plt.hist(graphs,bins=np.arange(0,0.1,0.0001),color='k')
        plt.xlabel('IPD(s)')
        plt.show()



    def getCurrentSamples(self):
        ret={}
        for k in self.currentSamples:
            ret[k]=self.base_traffic[k]
        return ret