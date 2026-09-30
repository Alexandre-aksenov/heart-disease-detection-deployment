from pydantic import BaseModel


class PredictionRequest(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
# reasonable default values,
# boundaries
# can be added using the domain knowledge.


feat_names = [
    "age", "sex", "cp", "trestbps",
    "chol", "fbs", "restecg", "thalach",
    "exang", "oldpeak", "slope", "ca"
]


def validate_feature_vals(features: dict[str, float]) -> bool:
    try:
        checked = PredictionRequest(**features)
        return True
    except ValueError:
        return False


def dict_feat(lst_vals: list[float]) -> dict[str, float]:
    """
        Places 12 features into a dict.
        Input: the values of 12 features.
        Output: these values placed in a dictionary.

        This function requires exactly 12 values.

        see: https://stackoverflow.com/a/209854
    """
    if not len(lst_vals) == len(feat_names):
        raise ValueError("12 features are expected")

    res = dict(zip(feat_names, lst_vals))

    # Validation against the constraints above: PredictionRequest
    if not validate_feature_vals(res):
        raise ValueError("Some features are not valid")

    return res


if __name__ == '__main__':
    import pickle

    with open("ex_dict_Features.pkl", 'rb') as f:
        ex_dict_feat = pickle.load(f)
    print(type(ex_dict_feat))  # dict

    lst_vals_ex = [ex_dict_feat[key] for key in feat_names]
    print(len(lst_vals_ex))  # 12
    reconstructed_dict_feat = dict_feat(lst_vals_ex)
    print("Could the dict be reconstructed?", reconstructed_dict_feat == ex_dict_feat)
    # True

    # test 2: a list of length 11
    try:
        dict_error = dict_feat(list(range(11)))
    except ValueError:  # executed!
        print('Exception correctly raised on the number of features.')

    # test 3: a str value
    try:
        data_error2 = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 1.0, 2.0]
        dict_error2 = dict_feat(data_error2)
    except ValueError:  # executed!
        print('Exception correctly raised on the types.')
