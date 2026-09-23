import pandas as pd

import os

import sklearn
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# For saving the trained classifier
import pickle

# Check versions
print(f'Pandas Version: {pd.__version__}')  # 3.0.0
print(f'sklearn Version: {sklearn.__version__}')  # 1.9.0

# import data
path_to_data = './heart.csv'
data = pd.read_csv(filepath_or_buffer=os.path.join(path_to_data))
# -> Warning: No overload of './heart.csv' matches the arguments
# Expected one of:
# str | PathLike[str] | ReadCsvBuffer[bytes] | ReadCsvBuffer[str]
# See: https://stackoverflow.com/a/72724850
# ("I would ignore this warning.")

# Column selection
data.drop(columns=['target'], inplace=True)

# Specify features, our target = ["thal"]
target = [data.columns[-1]]
print("Target column: ", target)  # ['thal']

all_features = data.columns[0:len(data.columns)-1]
print(all_features)
# Index(['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
#       'exang', 'oldpeak', 'slope', 'ca'],
#      dtype='object')

print('Shape of preprocessed data:', data.shape)  # (1025, 13)

# Split
data_train, data_test = train_test_split(data,
                                         test_size=0.2,
                                         stratify=data[target],
                                         random_state=42)

# Train the model
clf = RandomForestClassifier(n_estimators=50, random_state=1)
clf.fit(data_train[all_features], data_train[target[0]])

# performance on test set
preds_test = clf.predict(data_test[all_features])
print('-------')
print("Classification Report.")
print(classification_report(data_test[target], preds_test))
"""
Classification Report.
              precision    recall  f1-score   support

           0       1.00      1.00      1.00         1
           1       1.00      1.00      1.00        13
           2       0.97      1.00      0.99       109
           3       1.00      0.96      0.98        82

    accuracy                           0.99       205
   macro avg       0.99      0.99      0.99       205
weighted avg       0.99      0.99      0.99       205
"""


# Export the trained model:
# with open("./app/RF_classifier.pkl", "wb") as f:
#     pickle.dump(clf, f)

# Created a file of 1.1 MB
