import numpy as np

class LogisticRegression:
    def __init__(self,X,Y,Lrate):
        Feature = X.shape[1]
        self.Weights = np.random.rand(1,Feature)
        self.Biases = np.random.rand(1,1)
        self.Threshold = 0.5
        self.Lrate = Lrate
        self.Epochs =10
        
        self.Train(X,Y)

    def Sigmoid(self,Z):
        return 1/(1+np.e**-Z)
    
    def Train(self,X,Y):
        N = X.shape[0]

        for _ in range(self.Epochs):
            #Forward Propagation
            Z = np.dot(self.Weights,X.T) + self.Biases.copy()
            A = self.Sigmoid(Z)
            
            #Backward Propagation
            LOSS = Y.T-A
            LOSSX = np.dot(LOSS,X)
            
            #Updating Parameters
            self.Weights += (self.Lrate/N)*LOSSX
            self.Biases += (self.Lrate/N)*np.sum(LOSSX,axis=1)

    def Predict(self,X):
        Z = np.dot(self.Weights,X.T) + self.Biases.copy()
        A = self.Sigmoid(Z)

        # Add Threshold value

        return A

X = np.random.rand(1000,10)
Y = np.random.rand(1000,1)
LR = LogisticRegression(X,Y,0.01)
print(LR.Predict(np.random.rand(1,10)))