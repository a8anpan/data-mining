import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
df = pd.read_csv(r'C:\Users\User\Desktop\dm2020_1058112\winequality-red.csv', sep = ',')

y=df.quality
x=df.drop('quality',axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y,  test_size = 0.25)

for index, row in x_train.iterrows():
	if index % 3:
		row["pH"] = np.nan
        


imp = SimpleImputer(missing_values=np.nan, strategy='mean')
logreg = LogisticRegression()
steps = [('imputation', imp),('logistic_regression', logreg)]
pipeline = Pipeline(steps)

x_train, x_test, y_train, y_test = train_test_split(x, y,  test_size = 0.25)

pipeline.fit(x_train, y_train)
y_pred = pipeline.predict(x_test)
pipeline.score(x_test, y_test)


x_train, x_test, y_train, y_test = train_test_split(x, y,  test_size = 0.25)

param_grid = {'C': [20], 'gamma': [1],'kernel': ['linear']}
grid = GridSearchCV(SVC(),param_grid)
grid.fit(x_train,y_train)
print(grid.best_estimator_)
grid_predictions = grid.predict(x_test)
print(classification_report(y_test,grid_predictions))

