class Model:
    """
    Call the trained Random Forest classifier on a given vector of features.
    To be used in a fastapi API.
    """

    def __init__(self, name='') -> None:
        self.name = name

    def dict_2_DF(self, example):
        """
        dict -> dataframe with one row
        """
        import pandas as pd
        return pd.DataFrame([example])

    def dummy_predict(self, example={}):
        """
        Print the dataframe, which corresponds to the given dict.
        This dummy prediction is useful for testing
        the step of reading data into a dict, then dataframe.
        """
        inDF = self.dict_2_DF(example)
        return 'Dummy predict:\n' + str(inDF)

    def predict(self, example={}) -> str:
        """
        Loads the model from the file, evaluates on the given feature vector.

        Input.
            example: the feature vector
            (dict).
        Output.
            The predicted label.
        """

        import pickle
        inDF = self.dict_2_DF(example)

        # load the trained model
        with open("RF_classifier.pkl", 'rb') as f:
            clf_from_saved = pickle.load(f)

        # predict
        pred = clf_from_saved.predict(inDF)
        return str(pred)


def main():
    """
    Try both models on the saved vector from the testing set, used as example.
    """

    # load sample data
    import pickle
    with open("ex_dict_Features.pkl", 'rb') as f:
        example_data = pickle.load(f)

    print(type(example_data))  # 'dict'

    model = Model('RF')

    # dummy predict :
    dummy_str = model.dummy_predict(example_data)
    print(dummy_str)

    # random forest :
    predicted_cl = model.predict(example_data)  # [1]
    print('Predicted class: ', predicted_cl)


if __name__ == '__main__':
    main()
