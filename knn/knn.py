import numpy as np
import pandas as pd
from sklearn import neighbors, metrics
from sklearn.model_selection import train_test_split

X = data[["radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean","radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se","radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst"]].values
y = data[['diagnosis']].copy()
X = np.array(X)
label_mapping = {
    'M':0,
    'B':1
}
y['diagnosis'] = y['diagnosis'].map(label_mapping)
y = np.array(y)


knn = neighbors.KNeighborsClassifier(n_neighbors=25, weights="uniform")
X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.2)
knn.fit(X_train, y_train.ravel())
prediction = knn.predict(X_test)

acuracy = metrics.accuracy_score(y_test, prediction)
print(f"accuracy: {acuracy}")
diagnosis = 60
print("actual value ", y[diagnosis])
print("predicted value", knn.predict(X)[diagnosis])
