# heart-disease-detection-deployment
Example classifier for predicting heart disease is deployed using <code>fastapi</code>.

<b>About the dataset.</b>

This dataset comes from:
https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset/.

It contains 1025 columns and 12 features.
The column <code>thal</code> is predicted, and the previous <code>target</code> is removed.

<b>About the problem.</b>

The classifier is trained on the local machine (the environment for training the model is given by environment.yml), then deployed in a container using FastAPI.

The model is evaluated using its accuracy and the average F1-score.

<b>Selected model.</b>

All features are treated as numeric. The Random Forest classifier (50 estimators) is trained (folder <code>Classification</code>)
and the trained model is saved to the folder <code>app</code>. A container is built and deployed in the script <code>app/script.sh</code>.
A test example is provided in the end of this script.

<b>Results.</b>

This relatively simple classifier achieves 99% accuracy on test set.

<b>Feedback and additional questions.</b>

All questions about the source code should be adressed to its author Alexandre Aksenov:
* GitHub: Alexandre-aksenov
* Email: alexander1aksenov@gmail.com
