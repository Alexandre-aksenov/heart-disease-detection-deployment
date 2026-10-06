from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    age: float = Field(gt=0, lt=100)
    sex: float = Field(ge=0, le=1)
    cp: float = Field(ge=0, lt=3)
    trestbps: float = Field(gt=50, lt=250)
    chol: float = Field(gt=100, lt=600)
    fbs: float = Field(ge=0, le=1)
    restecg: float = Field(ge=0, le=2)
    thalach: float = Field(gt=70, le=220)
    exang: float = Field(gt=0, le=1)
    oldpeak: float = Field(gt=0, le=7)
    slope: float = Field(gt=-5, le=5)  # can be made more precise on basis of data knowledge
    ca: float = Field(gt=0, le=5)  # can be made more precise on basis of data knowledge
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
    # Validation of len
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
    print("Example dict of features: ", type(ex_dict_feat))  # dict
    print(ex_dict_feat)
    # {
    # 'age': 58.0, 'sex': 0.0, 'cp': 0.0, 'trestbps': 170.0, 'chol': 225.0,
    # 'fbs': 1.0, 'restecg': 0.0, 'thalach': 146.0, 'exang': 1.0,
    # 'oldpeak': 2.8, 'slope': 1.0, 'ca': 2.0
    # }

    lst_vals_ex = [ex_dict_feat[key] for key in feat_names]
    print("Length:", len(lst_vals_ex))  # 12
    print("--")
    reconstructed_dict_feat = dict_feat(lst_vals_ex)
    print("Could the dict be reconstructed?", reconstructed_dict_feat == ex_dict_feat)
    # Should be True

    # test 2: a list of length 11
    try:
        dict_error2 = dict_feat(list(range(11)))
    except ValueError:  # executed!
        print('Exception correctly raised on the number of features.')

    # test 3: a str value
    try:
        data_error3 = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 1.0, 2.0]
        dict_error3 = dict_feat(data_error3)
    except ValueError:  # executed!
        print('Exception correctly raised on the types.')

    # test 4: value out of bounds
    try:
        data_error4 = [
            58.0, 2.0, 0.0, 170.0, 225.0,
            1.0, 0.0, 146.0, 1.0,
            2.8, 1.0,  2.0
        ]
        dict_error4 = dict_feat(data_error4)
    except ValueError:  # executed!
        print('Exception correctly raised on a value out of range.')
