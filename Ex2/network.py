import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm

class HopfieldNetwork(object):      
    def train_weights(self, train_data):
        num_data =len(train_data)
        self.num_neuron = 100
        # initialize weights 100*100
        W = np.zeros((self.num_neuron, self.num_neuron))
        rho = np.sum([np.sum(t) for t in train_data]) / (num_data*self.num_neuron)
        # Hebb rule
        for i in tqdm(range(num_data)):
            t = train_data[i] - rho
            W += np.outer(t, t)
        
        # Make diagonal element of W into 0
        diagW = np.diag(np.diag(W))
        W = W - diagW
        W /= num_data
        self.W = W 
    
    def predict(self, data, num_iter=30, threshold=0, asyn=True):
        self.num_iter = num_iter
        self.threshold = threshold
        self.asyn = asyn
        copied_data = np.copy(data)
        predicted = []
        for i in tqdm(range(len(data))):
            #add to predict after update
            predicted.append(self._run(copied_data[i]))
        return predicted

    def energy(self, s):
        return -0.5 * s @ self.W @ s + np.sum(s * self.threshold)
    def _run(self, init_s):
            # Compute initial state energy
            s = init_s
            e = self.energy(s)
            for i in range(self.num_iter):
                for j in range(100):
                    # Select random neuron
                    idx = np.random.randint(0, self.num_neuron) 
                    # Update s with the neuron
                    s[idx] = np.sign(self.W[idx].T @ s - self.threshold)
                e_new = self.energy(s)
                # if s is converged
                if e == e_new:
                    return s
                e = e_new
            return s