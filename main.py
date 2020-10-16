import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

fake = pd.read_csv('data.csv')
fake = fake.drop(['URLs'], axis=1)
fake = fake.dropna()

x = fake.iloc[:, :-1].values
y = fake.iloc[:, -1].values

cv = CountVectorizer(max_features=5000)
mat_body = cv.fit_transform(x[:, 1]).todense()

cv_head = CountVectorizer(max_features=5000)
mat_head = cv_head.fit_transform(x[:, 1]).todense()

x_mat = np.hstack((mat_head, mat_body))
x_train, x_test, y_train, y_test = train_test_split(x_mat, y, test_size=0.2, random_state=0)
dtc = DecisionTreeClassifier(criterion= 'entropy')
dtc.fit(x_train, y_train)
y_pred = dtc.predict(x_test)

print('Confusion Matrix:\n',confusion_matrix(y_test, y_pred))

print("[[Correct detection, Incorrect detection]\n"
      "[Incorrect detection, Correct detection]]")