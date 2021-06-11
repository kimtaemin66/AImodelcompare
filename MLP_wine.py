from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

wine = datasets.load_wine()
x_train, x_test, y_train, y_test = train_test_split(wine.data, wine.target, train_size = 0.6)

sc = StandardScaler()
sc.fit(x_train)

x_train_std = sc.transform(x_train)
x_test_std = sc.transform(x_test)

mlp=MLPClassifier(hidden_layer_sizes=(10),
                  learning_rate_init=0.1,
                  batch_size=32,
                  solver='sgd',
                  verbose=True)
mlp.fit(x_train_std, y_train)

pre = mlp.predict(x_test_std)
print('정확도 :' ,accuracy_score(y_test,pre)*100,"%")



