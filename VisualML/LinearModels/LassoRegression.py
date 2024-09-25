import numpy as np

class LassoRegression:
    def __init__(self,X,Y,Lrate,lambd=0.1):
        Feature = X.shape[1]
        self.Weights = np.random.rand(1,Feature)
        self.Biases = np.random.rand(1,1)
        self.Lrate = Lrate
        self.Lambd = lambd
        self.Epochs =10
        
        self.Train(X,Y)

    def Train(self,X,Y):
        N = X.shape[0]

        for _ in range(self.Epochs):
            #Forward Propagation
            A = np.dot(self.Weights,X.T) + self.Biases.copy()
            
            #Backward Propagation
            LOSS = Y.T-A
            LOSSX = np.dot(LOSS,X) + self.Lambd*np.sign(self.Weights)
            
            #Updating Parameters
            self.Weights -= (self.Lrate/N)*LOSSX
            self.Biases -= (self.Lrate/N)*np.sum(LOSSX,axis=1)

    def Predict(self,X):
        A = np.dot(self.Weights,X.T) + self.Biases.copy()

        return A

X = np.random.rand(1000,10)
Y = np.random.rand(1000,1)
LR = LassoRegression(X,Y,0.01,lambd=0.1)
print(LR.Predict(np.random.rand(5,10)))