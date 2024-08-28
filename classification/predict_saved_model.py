"""
Check prediction based on the fitted model
before replicating in the container.
A custom row number can be entered as command line argument,
the default number (2) was used for testing the app.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# for loading the trained classifier
import pickle

# for using a command-line parameter
import sys


def main():
    # 1. Load:
    # the trained model (pickle),
    # the dataset.
    data = pd.read_csv('../dat/heart.csv')
    data.drop(columns=['target'], inplace=True)  # (1025,13)
    target = [data.columns[-1]]  # ['thal']
    all_features = data.columns[0:len(data.columns) - 1]

    with open("../app/RF_classifier.pkl", 'rb') as f:
        clf_from_saved = pickle.load(f)

# 2. Subdivide the dataset into train-test in the same way as before.
    data_train, data_test = train_test_split(data, test_size=0.2, stratify=data[target], random_state=42)

# 3. predict on the dataset.
    preds_test = clf_from_saved.predict(data_test[all_features])

# 4. Print out the selected row of test dataset
# The input is converted to integer checking:
# that the 'str' is convertible
# that the integer is in bounds.
    nb_testing = 2

    if len(sys.argv) == 2:
        _, nb_testing = sys.argv
        if not (nb_testing.isdigit()):
            print('Values are not valid!')
            return
        nb_testing = int(nb_testing)
        if not (nb_testing >= 0 and nb_testing < data_test.shape[0]):
            print('A valid row index is needed (<205).')
            return
    else:
        print('using the default row (2) for testing')

# convert to integer checking:
# that the 'str' is convertible
# that the integer is in bounds.

    print("The testing data is")
    print(data_test.iloc[nb_testing])

    """
nb_testing = 2
->

age          58.0
sex           0.0
cp            0.0
trestbps    170.0
chol        225.0
fbs           1.0
restecg       0.0
thalach     146.0
exang         1.0
oldpeak       2.8
slope         1.0
ca            2.0
thal          1.0
Name: 612, dtype: float64

Agrees with the notebook!

    """

    # and the prediction.
    print("The predicted 'thal' is:")
    print(preds_test[nb_testing])  # 1
    # correct answer, agrees with the notebook!


if __name__ == '__main__':
    main()
