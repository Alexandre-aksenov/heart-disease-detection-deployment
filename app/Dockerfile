# The versions of python and libraries are fixed
# for calling the classifier with same versions as on my local machine.

# FROM python:3.11  
# RUN python -m pip install uvicorn==0.24.0.post1 fastapi==0.105.0 numpy==1.26.4 pandas==2.1.1 scikit-learn==1.3.0
# (update) ->

FROM python:3.14

RUN python -m pip install uvicorn==0.52.4 fastapi==0.138 pandas==3.0 scikit-learn==1.9


WORKDIR /app
COPY model.py model.py
COPY fast_api.py fast_api.py

# the trained Random-Forest classifier
COPY RF_classifier.pkl RF_classifier.pkl  

# the conversion list of values -> dict
COPY dict_feat.py dict_feat.py  

EXPOSE 5003

CMD ["uvicorn", "fast_api:app", "--host", "0.0.0.0", "--port", "5003", "--reload"]
