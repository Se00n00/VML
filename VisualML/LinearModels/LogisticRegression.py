import numpy as np

X = np.random.rand(1000,10)
Y = np.random.rand(1000,1)

class LinearRegression:
    def __init__(self,X,Y,Lrate):
        self.Weights = np.random.rand(1,10)
        self.Biases = np.random.rand(1,1)
        self.Lrate = Lrate
        self.Epochs =10
        
        self.Train(X,Y)

    def Sigmoid(self,Z):
        return 1/(1+np.e**-Z)
    
    def Train(self,X,Y):
        for _ in range(self.Epochs):
            print("Epochs : ",_)
            #Forward Propagation
            Z = np.dot(self.Weights,X.T) + self.Biases.copy()
            A = self.Sigmoid(Z)
            
            #Backward Propagation
            LOSS = Y.T-A
            LOSSX = np.dot(LOSS,X)
            
            #Updating Parameters
            self.Weights += (self.Lrate/1000)*LOSSX
            self.Biases += (self.Lrate/1000)*np.sum(LOSSX,axis=1)

    def Predict(self,X):
        Z = np.dot(self.Weights,X.T) + self.Biases.copy()
        A = self.Sigmoid(Z)

        return A

LR = LinearRegression(X,Y,0.01)
print(LR.Predict(np.random.rand(5,10)))