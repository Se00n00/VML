import numpy as np

class RidgeRegression:
    def __init__(self,X,Y,Lrate,lambd1=0.1,lambd2=0.1):
        Feature = X.shape[1]
        self.Weights = np.random.rand(1,Feature)
        self.Biases = np.random.rand(1,1)
        self.Lrate = Lrate
        self.Lambd1 = lambd1
        self.Lambd2 = lambd2
        self.Epochs =20
        
        self.Train(X,Y)

    def Train(self,X,Y):
        N = X.shape[0]

        for _ in range(self.Epochs):
            #Forward Propagation
            A = np.dot(self.Weights,X.T) + self.Biases.copy()
            
            #Backward Propagation
            LOSS = Y.T-A
            LOSSX = np.dot(LOSS,X)+ self.Lambd1*np.sign(self.Weights) + 2*self.Lambd2*self.Weights
            
            #Updating Parameters
            self.Weights -= (self.Lrate/N)*LOSSX
            self.Biases -= (self.Lrate/N)*np.sum(LOSSX,axis=1)

    def Predict(self,X):
        A = np.dot(self.Weights,X.T) + self.Biases.copy()

        return A

X = np.random.rand(1000,10)
Y = np.random.rand(1000,1)
LR = RidgeRegression(X,Y,0.01,lambd1=0.1,lambd2=0.1)
print(LR.Predict(np.random.rand(5,10)))