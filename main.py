import numpy as np

class NeuralNetwork:
    def __init__(self, hidden_layer_size=100):
        self.hidden_layer_size=hidden_layer_size
    #===================METRICHE=======================
    #metrica dell'accuracy
    def accuracy(self, y, y_pred):
        return np.sum(y==y_pred)/len(y) #vettorizzazione
    #cross entropy/Log loss
    def __log_loss(self, y_true, y_prob):
        return -np.sum(np.dot(y_true, np.log(y_prob))+(np.dot((1-y_true), np.log(1-y_prob))))/len(y_true)

    #=================PREDIZIONE=======================
    #ReLU
    def __relu(self, Z):
        return np.maximum(Z, 0)
    #Sigmoide per classificazione binaria
    def __sigmoid(self, Z):
        return 1/(1+np.power(np.e, -Z))
    def __forward_propagation(self, X):
        Z1=np.dot(X, W1)+b1
        A1=self.__relu(Z1)
        Z2=np.dot(A1, W2)+b2
        self.cache=(Z1, A1, Z2, A2)
        return A2.ravel()
    def predict(self, X):
        proba=self.__forward_propagation(X)
        y=np.zeros(X.shape[0])
        y[proba>=0.5]=1
        y[proba<0.5]=0
        return 0
    def predict_proba(self, X):
        return self.__forward_propagation(X)
    #==================ADDESTRAMENTO======================


