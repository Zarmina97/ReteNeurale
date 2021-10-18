import pandas as pd
import numpy as np
import utils
CSV_URL = "https://raw.githubusercontent.com/ProfAI/tutorials/master/Come%20Creare%20una%20Rete%20Neurale%20da%20Zero/breast_cancer.csv"
breast_cancer = pd.read_csv(CSV_URL)

X = breast_cancer.drop("malignant", axis=1).values
y = breast_cancer["malignant"].values


def train_test_split(X, y, test_size=0.3, random_state=None):
    if (random_state != None):
        np.random.seed(random_state)

    n = X.shape[0]
    test_indices = np.random.choice(n, int(n * test_size),
                                    replace=False)  # selezioniamo gli indici degli esempi per il test set

    # estraiamo gli esempi del test set
    # in base agli indici

    X_test = X[test_indices]
    y_test = y[test_indices]

    # creiamo il train set
    # rimuovendo gli esempi del test set
    # in base agli indici

    X_train = np.delete(X, test_indices, axis=0)
    y_train = np.delete(y, test_indices, axis=0)
    return (X_train, X_test, y_train, y_test)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


X_max = X_train.max(axis=0)
X_min = X_train.min(axis=0)
X_train = (X_train - X_min)/(X_max-X_min)
X_test = (X_test - X_min)/(X_max-X_min)



model = utils.NeuralNetwork()
model.fit(X_train, y_train, epochs=500, lr=0.01)
model.evaluate(X_test, y_test)

exams_df = pd.read_csv("https://raw.githubusercontent.com/ProfAI/tutorials/master/Come%20Creare%20una%20Rete%20Neurale%20da%20Zero/exam%20results.csv")
X_new = exams_df.values
X_new = (X_new - X_min)/(X_max-X_min)


y_pred, y_proba = model.predict(X_new, return_proba=True)


classes = ["benigno", "maligno"]
for i, (pred, proba) in enumerate(zip(y_pred, y_proba)):
  print("Risultato %d = %s (%.4f)" % (i+1, classes[int(pred)], proba))
