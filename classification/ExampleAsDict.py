"""
Save a row of the testing set to the folder 'app' for testing.
A custom row number can be entered as command line argument,
the default number (2) was used for testing the app.

The instructions for exportation are commented out.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# for loading the trained classifier
# import pickle

# for using a command-line parameter
import sys


def main():
    # 1. Load:
    # the trained model (pickle),
    # the dataset.
    data = pd.read_csv('../dat/heart.csv')
    data.drop(columns=['target'], inplace=True)  # (1025,13)
    target = [data.columns[-1]]  # ['thal']
    # all_features = data.columns[0:len(data.columns) - 1]


# 2. Subdivide the dataset into train-test in the same way as before.
    data_train, data_test = train_test_split(data, test_size=0.2, stratify=data[target], random_state=42)
    data_test.drop(columns=['thal'], inplace=True)


# 3. Print out the 3rd row -> a row of test dataset
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

    testing_row_feat = data_test.iloc[nb_testing]
    print("The testing data is")
    print(testing_row_feat)

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
Name: 612, dtype: float64

Agrees with the notebook!

    """

    # export the dict.
    TestingDict = testing_row_feat.to_dict()
    print('Feature vector converted to dict:')
    print(TestingDict)  # dict with all data above, OK

    # export to pickle
#    with open("../app/ex_dict_Features.pkl", 'wb') as f:
#        pickle.dump(TestingDict, f)
#    return TestingDict


if __name__ == '__main__':
    main()
