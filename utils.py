import numpy as np

class NeuralNetwork:
    def __init__(self, hidden_layer_size=100):
        self.hidden_layer_size=hidden_layer_size
    #===================METRICHE=======================
    #metrica dell'accuracy
    def __accuracy(self, y, y_pred):
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
        Z1=np.dot(X, self._W1)+self._b1
        A1=self.__relu(Z1)
        Z2=np.dot(A1, self._W2)+self._b2
        A2=self.__sigmoid(Z2)
        self.__forward_cache=(Z1, A1, Z2, A2)
        return A2.ravel()
    def predict(self, X, return_proba=False):
        proba=self.__forward_propagation(X)
        y=np.zeros(X.shape[0])
        y[proba>=0.5]=1
        y[proba<0.5]=0
        if (return_proba):
            return (y, proba)
        else:
            return proba
    def predict_proba(self, X):
        return self.__forward_propagation(X)
    #==================ADDESTRAMENTO======================

    def __relu_derivative(self, Z):
        dZ=np.zeros(Z.shape)
        dZ[Z>0]=1
        return dZ
    def __back_propagation(self, X, y):
        Z1, A1, Z2, A2 = self.__forward_cache

        m = A1.shape[1]
        dZ2 = A2 - y.reshape(-1, 1)  # il reshape ci serve per far combaciare le dimensioni dei due vettori
        dW2 = np.dot(A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0) / m
        dZ1 = np.dot(dZ2, self._W2.T) * self.__relu_derivative(Z1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0) / m  # eseguiamo la somma lungo le righe

        return dW1, db1, dW2, db2


    def _init_weights(self, input_size, hidden_size):
        self._W1 = np.random.randn(input_size, hidden_size)
        self._b1 = np.zeros(hidden_size)
        self._W2 = np.random.randn(hidden_size, 1)
        self._b2 = np.zeros(1)

    def fit(self, X, y, epochs=200, lr=0.01):

        self._init_weights(X.shape[1], self.hidden_layer_size)

        for _ in range(epochs):
            Y = self.__forward_propagation(X)
            dW1, db1, dW2, db2 = self.__back_propagation(X, y)
            self._W1 -= lr * dW1
            self._b1 -= lr * db1
            self._W2 -= lr * dW2
            self._b2 -= lr * db2

    def evaluate(self, X, y):
        y_pred, proba = self.predict(X, return_proba=True)
        accuracy = self.__accuracy(y, y_pred)
        log_loss = self.__log_loss(y, proba)
        return (accuracy, log_loss)