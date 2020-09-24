import numpy as np 

class KNearestNeighbors:
    def __init__(self, k):
        self.k = k 

    def predict(self, X_test):
        
        predictions = []

        for i in range(len(X_test)):
            distances = []
            nearestneighbors = []
            for z in range(len(self.X_train)):
                dist = np.linalg.norm(np.array(self.X_train[z])-np.array(X_test[i]))
                distances.append([dist, z])
            distances.sort()
            distances = distances[0:self.k]
            for _, j in distances:
                nearestneighbors.append(self.y_train[j])
            result = max(nearestneighbors)
            predictions.append(result)
        return predictions

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return float(sum(y_pred == y_test)) / float(len(y_test)) 