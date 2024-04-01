import pandas as pd
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

def Cal_GaussianNB(file_train, file_test):
  dftrn = pd.read_csv(file_train, header=None)
  dftst = pd.read_csv(file_test, header=None)
  row_count, column_count = dftrn.shape
  cc = column_count - 1
  x_train = dftrn.iloc[:,:-1]
  y_train = dftrn.iloc[:,cc]
  x_test = dftst.iloc[:,:-1]
  y_test = dftst.iloc[:, cc]
  NBModel = GaussianNB()
  NBModel.fit(x_train, y_train)
  y_predicted = NBModel.predict(x_test)
  ascore = accuracy_score(y_test, y_predicted)*100
  print("-----------------------TRAN HONG NHUT B2014938------------------------")
  print("-----------------THE GAUSSIANNB CLASSIFICATION RESULT-----------------")
  print("-Accuracy: ", ascore, "%")
  print("-Confusion Matrix: \n", metrics.confusion_matrix(y_test, y_predicted))


Cal_GaussianNB("data\\iris\\iris.trn", "data\\iris\\iris.tst")