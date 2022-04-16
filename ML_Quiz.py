import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

""" Creating Key for later conversion into class_type"""
a_classes = pd.read_csv("animal_classes.csv")

animals_key = {
    row[0]: row[1] for row in zip(a_classes["Class_Number"], a_classes["Class_Type"])
}
# print(animals_key)

"""Assigning data to our test/train variables"""

import numpy as np

df_train = pd.read_csv("animals_train.csv")
df_test = pd.read_csv("animals_test.csv")
# print(df_train.shape) # use for testing shape
a_data = df_train.to_numpy()

n_samples, n_features = a_data.shape
n_features -= 1
data_train = a_data[:, 0:n_features]
target_train = a_data[:, n_features]
print(data_train.shape, target_train.shape)

a_test = df_test.to_numpy()
# print(a_test.shape)  # use to confirm right shape

n_samples, n_features = a_test.shape
data_test = a_test[:, 1:n_features]
# print(data_test) # use for testing
target_test = a_test[:, 0]
print(data_test.shape, target_test.shape)

""" Fitting the model"""
knn = KNeighborsClassifier()

knn.fit(X=data_train, y=target_train)

predicted = knn.predict(X=data_test)
expected = target_test
# print(predicted[:20]) use to visualize a portion of the results
# print(expected[:20])


"""Translating class codes"""
"""Use following to iterate and test if predicted, expected worked as expected"""
"""
for p, e in zip(predicted[:10], expected[:10]):
    print(f"predicted: {animals_key[p]}, expected: {e}")
"""

"""Creating the CSV output"""
import csv

header = ["animal_name", "predicted_class"]
data = [[e, animals_key[p]] for p, e in zip(predicted, expected)]
# print(data) # use to check if list comprehension worked properly
with open("prediction_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(data)
print("Done")
