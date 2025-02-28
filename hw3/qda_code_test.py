#QDA Class

#Imports
import numpy as np


class QDA:
    def __init__(self, t_labels_unique):
        self.t_labels_unique = np.array(t_labels_unique)
        self.means = {}
        self.total_ep = {}
        self.priors = {}
        
        #Storing Standardization Paramters
        self.feature_means = None
        self.feature_stds = None
        
        
    def standardize_features(self, data):
        if self.feature_means is None or self.feature_stds is None:
            self.feature_means = np.mean(data, axis=0)
            self.feature_stds = np.std(data, axis=0) + 1e-8 
        return (data - self.feature_means) / self.feature_stds
        
        
    def qda_training(self, t_labels, t_data):
        if len(t_data.shape) > 2:
            t_data = t_data.reshape(t_data.shape[0], -1)
            
        t_data = self.standardize_features(t_data)
        total_samples = len(t_labels)
        
        for label in self.t_labels_unique:
            idx = (t_labels == label).flatten()
            self.priors[label] = np.sum(idx) / total_samples
            class_data = t_data[idx]
            
            # Mean vector
            self.means[label] = np.mean(class_data, axis=0)
            
            # Covariance matrix
            self.total_ep[label] = np.cov(class_data, rowvar=False)
            
            n_features = self.total_ep[label].shape[0]
            
            class_size = class_data.shape[0]
            epsilon = max(1e-4, 2.0 / class_size)
            
            self.total_ep[label] += epsilon * np.eye(n_features)
            
            if n_features > 100:
                eigenvalues, eigenvectors = np.linalg.eigh(self.total_ep[label])
                threshold = 1e-6 * np.max(eigenvalues)
                eigenvalues[eigenvalues < threshold] = threshold
                self.total_ep[label] = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
                
                
    def qda_classify(self, tst_data):
        if len(tst_data.shape) > 2:
            tst_data = tst_data.reshape(tst_data.shape[0], -1)
        tst_data = self.standardize_features(tst_data)
        num_samples = tst_data.shape[0]
        constant_terms = {}
        precision_matrices = {}
        
        for label in self.t_labels_unique:
            constant_terms[label] = np.log(self.priors[label])
            
            try:
                precision_matrices[label] = np.linalg.inv(self.total_ep[label])
            except np.linalg.LinAlgError:
                precision_matrices[label] = np.linalg.pinv(self.total_ep[label])
                
            sign, logdet = np.linalg.slogdet(self.total_ep[label])
            constant_terms[label] -= 0.5 * logdet
            
        predictions = np.zeros(num_samples, dtype=self.t_labels_unique.dtype)
        
        batch_size = min(1000, num_samples)
        
        for i in range(0, num_samples, batch_size):
            end_idx = min(i + batch_size, num_samples)
            
            batch = tst_data[i:end_idx]
            
            batch_scores = np.zeros((batch.shape[0], len(self.t_labels_unique)))
            
            for j, label in enumerate(self.t_labels_unique):
                centered_data = batch - self.means[label]
                
                batch_scores[:, j] = -0.5 * np.sum(
                    centered_data @ precision_matrices[label] * centered_data, 
                    axis=1
                )
                
                batch_scores[:, j] += constant_terms[label]
            
            max_indices = np.argmax(batch_scores, axis=1)
            predictions[i:end_idx] = self.t_labels_unique[max_indices]
            
        return predictions
    