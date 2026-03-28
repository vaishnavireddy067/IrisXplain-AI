import pickle
from sklearn import svm
from sklearn.datasets import load_iris
iris = load_iris()
X, y = iris.data, iris.target
model = svm.SVC(probability=True)
model.fit(X, y)
with open("svm_model.pkl", "wb") as f:
    pickle.dump(model, f)
