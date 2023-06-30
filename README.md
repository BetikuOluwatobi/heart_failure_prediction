# Heart Failure Prediction Project

This project aims to predict heart failure using a dataset containing various features related to patients' health. The dataset consists of 918 instances with 11 features, including age, sex, chest pain type, blood pressure, cholesterol level, resting electrocardiographic results, maximum heart rate achieved, exercise-induced angina, ST depression induced by exercise, the slope of the peak exercise ST segment, and the presence of heart disease.

## Dataset

The dataset has the following features:

- Age: Patient's age (integer)
- Sex: Patient's sex (categorical: "Male" or "Female")
- ChestPainType: Type of chest pain (categorical: "Typical Angina", "Atypical Angina", "Non-anginal Pain", or "Asymptomatic")
- RestingBP: Resting blood pressure (integer)
- Cholesterol: Serum cholesterol level (integer)
- FastingBS: Fasting blood sugar level > 120 mg/dl (integer: 1 for true, 0 for false)
- RestingECG: Resting electrocardiographic results (categorical: "Normal", "ST-T wave abnormality", or "Left ventricular hypertrophy")
- MaxHR: Maximum heart rate achieved (integer)
- ExerciseAngina: Exercise-induced angina (categorical: "Yes" or "No")
- Oldpeak: ST depression induced by exercise relative to rest (float)
- ST_Slope: Slope of the peak exercise ST segment (categorical: "Upsloping", "Flat", or "Downsloping")
- HeartDisease: Presence of heart disease (integer: 1 for true, 0 for false)

## Project Steps

1. Data Collection: The dataset was obtained from Kaggle using the `KaggleApi` module.

```python
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()
api.dataset_download_files('fedesoriano/heart-failure-prediction','heart.csv', unzip=True)
```

2. Exploratory Data Analysis: Performed initial data exploration and visualization to gain insights into the dataset using libraries such as Pandas, Matplotlib, and Seaborn.

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('heart_failure_dataset.csv')

# Perform exploratory data analysis
# ...

# Visualize correlations and important features
# ...
```

3. Feature Engineering: Applied feature scaling using StandardScaler and prepared the data for model development.

```python
from sklearn.preprocessing import StandardScaler

# Scale the numerical features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data[['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']])

# Update the scaled features in the dataset
data[['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']] = scaled_features
```

4. Model Development: Explored different classification models using scikit-learn and built custom pipelines for each model.

```python
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Logistic Regression
logistic_pipeline = make_pipeline(StandardScaler(), LogisticRegression(C=1.0))
logistic_pipeline.fit(X_train, y_train)
predictions = logistic_pipeline.predict(X_test)

# Support Vector Classifier
svc_pipeline = make_pipeline(StandardScaler(), SVC(C=0.3))
svc_pipeline.set_params(**grid_search.best_params_)
svc_pipeline.fit(X_train, y_train)
predictions = svc_pipeline.predict(X_test)

# Decision Tree Classifier


dt_pipeline = make_pipeline(StandardScaler(), DecisionTreeClassifier())
dt_pipeline.set_params(**grid_search.best_params_)
dt_pipeline.fit(X_train, y_train)
predictions = dt_pipeline.predict(X_test)

# Random Forest Classifier
rf_pipeline = make_pipeline(RandomForestClassifier())
rf_pipeline.set_params(**grid_search.best_params_)
rf_pipeline.fit(X_train, y_train)
predictions = rf_pipeline.predict(X_test)
```

5. Model Evaluation: Tuned the models' parameters and evaluated their performance using appropriate metrics.

```python
from sklearn.model_selection import RandomizedSearchCV,validation_curve,l
from sklearn.metrics import confusion_matrix,precision_score, recall_score,f1_score,make_scorer

# Grid search for RandomForestClassifier
param_grid = {
    'randomforestclassifier__n_estimators': np.append(np.arange(400,900,50),170),  # Number of trees in the forest
    'randomforestclassifier__criterion': ['entropy'],  # Splitting criterion
    'randomforestclassifier__max_depth': [ 5,6,7,8,9,10, 11,13,14,20,24,27,28,29],  # Maximum depth of the tree
    'randomforestclassifier__min_samples_split': np.arange(6,21,1),  # Minimum number of samples required to split an internal node
    'randomforestclassifier__min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
    'randomforestclassifier__max_features': [0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.66,0.95],  # Number of features to consider when looking for the best split
    'randomforestclassifier__max_samples': np.arange(0.45,1,0.05)
}  

grid_search = RandomizedSearchCV(pipe_rf, param_distributions=param_grid,n_iter=1000, scoring='accuracy',n_jobs=-1,random_state=12)
grid_search.fit(X_train, y_train)

# Evaluate the best model
best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
confmat = confusion_matrix(y_test, predictions)
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i,s=confmat[i, j],va='center', ha='center')
plt.xlabel('predicted label')
plt.ylabel('true label')
plt.show()

# Save the best model
import pickle
import os
dest = os.path.join('research/', 'model_pipelines')
if not os.path.exists(dest):
    os.makedirs(dest)
pickle.dump(pipe_lr,
            open(os.path.join(dest,'logistic_pipeline.pkl'),'wb'),
            protocol=4)
pickle.dump(pipe_tree,
            open(os.path.join(dest,'dt_pipeline.pkl'),'wb'),
            protocol=4)
pickle.dump(pipe_svc,
            open(os.path.join(dest,'svc_pipeline.pkl'),'wb'),
            protocol=4)
pickle.dump(pipe_rf,
            open(os.path.join(dest,'rf_pipeline.pkl'),'wb'),
            protocol=4)
```

For more detailed code examples and implementation details, please refer to the Jupyter Notebook provided in this repository.

## Conclusion

The Heart Failure Prediction project aimed to develop a model that can accurately predict the occurrence of heart disease based on patient data. Through feature engineering and experimentation with different models, we identified the RandomForestClassifier as the best-performing model. The code and resources provided in this repository can be used as a reference for similar projects in the healthcare domain.
