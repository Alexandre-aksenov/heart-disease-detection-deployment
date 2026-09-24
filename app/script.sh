# The instructions for building a container and deploying a FastAPI,
# which predicts 'thal' using the trained Random-Forest classifier.


# N.B. The example vector of features (also provided as dictionary in the file 'ex_dict_Features.pkl')
# is the 3rd row of the test set obtained while training the classifier.
# Possible improvement. Allowing the user to upload a file to the web-page 
# can lead to a more practical UI.

docker build -t fast-api-rf-pl .

docker images  
# 'fast-api-rf-pl' appears with size near 1.56GB -> ..

docker run --name fast-api-rf-pl -d --rm -p 5003:5003 fast-api-rf-pl

docker ps
# 'fast-api-rf-pl' should appear in the list

# The following calls can be tested, e.g. in Postman:

# localhost:5003/
# -> (expected: status 200)
# "message": "FastAPI Hello World"

# localhost:5003/dummypredict?age=52&sex=0&cp=0&trestbps=170&chol=225&fbs=1&restecg=0&thalach=146&exang=1&oldpeak=2.8&slope=1&ca=2
# (dummy predict for testing that data is read and converted to a Dataframe of floats )
# -> (status 200)
#"Dummy predict:\n    age  sex   cp  trestbps   chol  ...  thalach  exang  oldpeak  slope   ca\n0  52.0  0.0  0.0     170.0  225.0  ...    146.0    1.0      2.8    1.0  2.0\n\n[1 rows x 12 columns]" 

# localhost:5003/predict?age=52&sex=0&cp=0&trestbps=170&chol=225&fbs=1&restecg=0&thalach=146&exang=1&oldpeak=2.8&slope=1&ca=2
# (getting the pedicted value of 'thal' for an example vector of features)
# -> (status 200 OK)
# "[1]"

# Equivalent calls in Command Line:

curl "http://localhost:5003"
# ->
# {"message":"FastAPI Hello World"}

curl "http://localhost:5003/dummypredict?age=52&sex=0&cp=0&trestbps=170&chol=225&fbs=1&restecg=0&thalach=146&exang=1&oldpeak=2.8&slope=1&ca=2"
#->
# "Dummy predict:\nshape: (1, 12)\n┌──────┬─────┬─────┬──────────┬───┬───────┬─────────┬───────┬─────┐\n│ age  ┆ sex ┆ cp  ┆ trestbps ┆ … ┆ exang ┆ oldpeak ┆ slope ┆ ca  │\n│ ---  ┆ --- ┆ --- ┆ ---      ┆   ┆ ---   ┆ ---     ┆ ---   ┆ --- │\n│ f64  ┆ f64 ┆ f64 ┆ f64      ┆   ┆ f64   ┆ f64     ┆ f64   ┆ f64 │\n╞══════╪═════╪═════╪══════════╪═══╪═══════╪═════════╪═══════╪═════╡\n│ 52.0 ┆ 0.0 ┆ 0.0 ┆ 170.0    ┆ … ┆ 1.0   ┆ 2.8     ┆ 1.0   ┆ 2.0 │\n└──────┴─────┴─────┴──────────┴───┴───────┴─────────┴───────┴─────┘"


# prettier form of the same answer:

# Dummy predict:
# shape: (1, 12)
# ┌──────┬─────┬─────┬──────────┬───┬───────┬─────────┬───────┬─────┐
# │ age  ┆ sex ┆ cp  ┆ trestbps ┆ … ┆ exang ┆ oldpeak ┆ slope ┆ ca  │
# │ ---  ┆ --- ┆ --- ┆ ---      ┆   ┆ ---   ┆ ---     ┆ ---   ┆ --- │
# │ f64  ┆ f64 ┆ f64 ┆ f64      ┆   ┆ f64   ┆ f64     ┆ f64   ┆ f64 │
# ╞══════╪═════╪═════╪══════════╪═══╪═══════╪═════════╪═══════╪═════╡
# │ 52.0 ┆ 0.0 ┆ 0.0 ┆ 170.0    ┆ … ┆ 1.0   ┆ 2.8     ┆ 1.0   ┆ 2.0 │
# └──────┴─────┴─────┴──────────┴───┴───────┴─────────┴───────┴─────┘


curl "http://localhost:5003/predict?age=52&sex=0&cp=0&trestbps=170&chol=225&fbs=1&restecg=0&thalach=146&exang=1&oldpeak=2.8&slope=1&ca=2"
#->
# "[1]"

# When all tests are done, the container can be closed using: 
# docker stop fast-api-rf-pl
