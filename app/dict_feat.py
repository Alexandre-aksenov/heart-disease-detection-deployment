feat_names = [
    "age", "sex", "cp", "trestbps",
    "chol", "fbs", "restecg", "thalach",
    "exang", "oldpeak", "slope", "ca"
]


def dict_feat(lst_vals):
    """
        Places 12 features into a dict.
        Input: the values of 12 features.
        Output: these values placed in a dictionary.

        This function requires exactly 12 values.

        see: https://stackoverflow.com/a/209854
    """
    if not len(lst_vals) == len(feat_names):
        raise ValueError("12 features are expected")
    return dict(zip(feat_names, lst_vals))


if __name__ == '__main__':
    import pickle

    with open("ex_dict_Features.pkl", 'rb') as f:
        ex_dict_feat = pickle.load(f)
    print(type(ex_dict_feat))  # dict

    lst_vals_ex = [ex_dict_feat[key] for key in feat_names]
    print(len(lst_vals_ex))  # 12
    reconstructed_dict_feat = dict_feat(lst_vals_ex)
    print(reconstructed_dict_feat == ex_dict_feat)  # True

    # test 2: a list of length 11
    try:
        dict_error = dict_feat(list(range(11)))
    except ValueError:  # executed!
        print('The function correctly raised an exception.')
