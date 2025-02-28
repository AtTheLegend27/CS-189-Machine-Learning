#LDA Class

#Imports
import numpy as np


class LDA: 
    def __init__(self, t_labels_unique):
        self.t_labels_unique = t_labels_unique
        self.means = {}
        self.total_ep = None
    
    def lda_training(self, t_labels, t_data):
        ep_m = []
        
        for label in self.t_labels_unique:
            idx = (t_labels == label).flatten()
            class_data = t_data[idx]
            
            if class_data.ndim > 2:
                class_data = class_data.reshape(class_data.shape[0], -1)
                
            self.means[label] = np.mean(class_data, axis = 0)
            ep_m.append(np.cov(class_data, rowvar=False))
            
        epsilon = 1e-5
                
        self.total_ep = sum(ep_m) / len(self.t_labels_unique)
        self.total_ep += epsilon * np.eye(self.total_ep.shape[0])
    
    def lda_classify(self, tst_data):
        inv_ep = np.linalg.inv(self.total_ep)
        predictions = []
        
        for x in tst_data:
                        
            if x.ndim > 1: 
                x = x.flatten()
                
            scores = [
                -0.5 * np.dot(np.dot((x - self.means[label]).T, inv_ep), (x - self.means[label]))
                for label in self.t_labels_unique
                ]
            predictions.append(self.t_labels_unique[np.argmax(scores)])
            
        return np.array(predictions)