# The versions of python and libraries are fixed
# for calling the classifier with same versions as on my local machine.
FROM python:3.11  

RUN python -m pip install uvicorn==0.24.0.post1 fastapi==0.105.0 pandas==2.1.1 scikit-learn==1.3.0

WORKDIR /app
COPY model.py model.py
COPY fast_api.py fast_api.py
COPY RF_classifier.pkl RF_classifier.pkl  # the trained Random-Forest classifier
COPY dict_feat.py dict_feat.py  # the conversion list of values -> dict

EXPOSE 5002

CMD ["uvicorn", "fast_api:app", "--host", "0.0.0.0", "--port", "5002", "--reload"]
