# Fast-API for the trained model of chest pain prediction.

from fastapi import FastAPI
from model import Model

import dict_feat

model = Model('RF model')
# dummy = dummy_model('FastAPI dummy model')

app = FastAPI()


# create a route
@app.get("/")
def index():
    return {"message": "FastAPI Hello World"}


@app.get("/dummypredict")
# def dummy_predict(text: str):
def dummy_predict(
    age: float, sex: float,
    cp: float, trestbps: float,
    chol: float, fbs: float,
    restecg: float, thalach: float,
    exang: float, oldpeak: float,
    slope: float, ca: float):
    """
    check that all 12 features are present,

    try converting them to numbers,
        See: https://fastapi.tiangolo.com/tutorial/query-params/
        '
        As they are part of the URL, they are "naturally" strings.

        But when you declare them with Python types
        (in the example above, as int),
        they are converted to that type and validated against it.
        '

    convert to dict,
    send to the model for prediction.
    """

    in_dict = dict_feat.dict_feat(
        [age, sex, cp, trestbps,
        chol, fbs, restecg, thalach,
        exang, oldpeak, slope, ca])

    response = model.dummy_predict(in_dict)
    return response


@app.get("/predict")
def predict(
    age: float, sex: float,
    cp: float, trestbps: float,
    chol: float, fbs: float,
    restecg: float, thalach: float,
    exang: float, oldpeak: float,
    slope: float, ca: float):
    """
    Take 12 features,

    try converting them to numbers,
        See: https://fastapi.tiangolo.com/tutorial/query-params/
        '
        As they are part of the URL, they are "naturally" strings.

        But when you declare them with Python types
        (in the example above, as int),
        they are converted to that type and validated against it.
        '

    convert to dict,
    send to the trained model for prediction.
    """
    in_dict = dict_feat.dict_feat(
        [age, sex, cp, trestbps,
        chol, fbs, restecg, thalach,
        exang, oldpeak, slope, ca])

    response = model.predict(in_dict)
    return response
