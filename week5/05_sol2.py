# Gaussian Naive Bayes
import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 결과를 출력하기 위한 엘리스 유틸리티 툴을 불러옵니다.
import elice_utils
eu = elice_utils.EliceUtils()

np.random.seed(21)

# 유방암 데이터 로드
dataset = datasets.load_breast_cancer()

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size = 0.25, random_state=21)

# (1) Gaussian Naive Bayes 모델을 만들고 데이터에 fit 하세요.
model_gaussianNB = GaussianNB()
model_gaussianNB.fit(X_train, y_train)
print(model_gaussianNB)

# (2) Logistic Regression 모델을 만들고 데이터에 fit 하세요.
model_logisticReg = LogisticRegression()
model_logisticReg.fit(X_train, y_train)
print(model_logisticReg)

# (3) Fitting 된 모델을 이용해 test 데이터의 label을 predict 하세요.
expected = y_test
pred_gaussianNB = model_gaussianNB.predict(X_test)
pred_logisticReg = model_logisticReg.predict(X_test)

# Prediction 확인
print(metrics.classification_report(expected, pred_gaussianNB))
print(metrics.classification_report(expected, pred_logisticReg))
