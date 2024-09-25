import numpy as np

class LinearRegression:
    ''' 
    Regression <type of Linear Regression> ::
    Linear: Linear Regression, 
    L1: Lasso Linear Regression, 
    L2: Ridge Linear Regression,
    ElasticNet: ElasticNet Linear Regression

    lambd1: when [Regression='L1' || Regression='ElasticNet']
    lambd2: when [Regression='L2' || Regression='ElasticNet']
    '''
    def __init__(self, X, Y, Lrate, Regression='Linear', lambd1=0.01, lambd2=0.01):
        Feature = X.shape[1]

        self.Regression = Regression

        #Hyper-Parameters Initiallization
        self.Lrate = Lrate
        self.Epochs = 100
        self.Lambd1 = lambd1
        self.Lambd2 = lambd2

        #Parameters Initiallization
        self.Weights = np.random.rand(1,Feature)
        self.Biases = np.random.rand(1,1)
        
        self.Train(X,Y)

    def Train(self,X,Y):
        N = X.shape[0]

        for _ in range(self.Epochs):
            #Forward Propagation
            A = np.dot(self.Weights,X.T) + self.Biases.copy()
            
            #Backward Propagation
            LOSS = Y.T-A
            
            if(self.Regression =='Linear'):
                LOSSX = np.dot(LOSS,X)
            elif(self.Regression == 'L1'):
                LOSSX = np.dot(LOSS,X) + self.Lambd1*np.sign(self.Weights)
            elif(self.Regression == 'L2'):
                LOSSX = np.dot(LOSS,X) + 2*self.Lambd2*self.Weights
            else:
                LOSSX = np.dot(LOSS,X)+ self.Lambd1*np.sign(self.Weights) + 2*self.Lambd2*self.Weights
            
            #Updating Parameters
            self.Weights -= (self.Lrate/N)*LOSSX
            self.Biases -= (self.Lrate/N)*np.sum(LOSSX,axis=1)

    def Predict(self,X):
        A = np.dot(self.Weights,X.T) + self.Biases.copy()
        return A


#Example Of Implementation

X = np.random.rand(1000,5)
Y = np.random.rand(1000,1)

LR = LinearRegression(X,Y,0.01)
print(LR.Predict(np.random.rand(5,5)))
LR = LinearRegression(X,Y,0.01,Regression='L1',lambd1=0.01)
print(LR.Predict(np.random.rand(5,5)))
LR = LinearRegression(X,Y,0.01,Regression='L2',lambd2=0.01)
print(LR.Predict(np.random.rand(5,5)))
LR = LinearRegression(X,Y,0.01,Regression='ElasticNet',lambd1=0.01,lambd2=0.01)
print(LR.Predict(np.random.rand(5,5)))