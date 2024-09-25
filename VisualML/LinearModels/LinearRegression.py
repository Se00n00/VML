import numpy as np

class LinearRegression:
    def __init__(self,X,Y,Lrate):
        Feature = X.shape[1]
        self.Weights = np.random.rand(1,Feature)
        self.Biases = np.random.rand(1,1)
        self.Lrate = Lrate
        self.Epochs =100
        print(self.Weights)
        self.Train(X,Y)
        print(self.Weights)

    def Train(self,X,Y):
        N = X.shape[0]

        for _ in range(self.Epochs):
            #Forward Propagation
            A = np.dot(self.Weights,X.T) + self.Biases.copy()
            
            #Backward Propagation
            LOSS = Y.T-A
            LOSSX = np.dot(LOSS,X)
            
            #Updating Parameters
            self.Weights -= (self.Lrate/N)*LOSSX
            self.Biases -= (self.Lrate/N)*np.sum(LOSSX,axis=1)

    def Predict(self,X):
        A = np.dot(self.Weights,X.T) + self.Biases.copy()

        return A

X = np.random.rand(1000,5)
Y = np.random.rand(1000,1)
LR = LinearRegression(X,Y,0.01)
print(LR.Predict(np.random.rand(5,5)))