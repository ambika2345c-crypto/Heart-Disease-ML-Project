# Heart Disease Prediction using Machine Learning

## About the Project

This project focuses on predicting heart disease using patient health data and Machine Learning classification algorithms. The dataset contains different health-related attributes along with the corresponding heart disease status.

The main aim is to apply different classification algorithms and compare how well they perform on the same dataset.

## Dataset

The dataset was obtained from Kaggle and contains health-related information about patients.

Some of the features include:

- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Serum Cholesterol
- Fasting Blood Sugar
- Resting Electrocardiographic Results
- Maximum Heart Rate
- Exercise Induced Angina
- Oldpeak
- ST Segment
- Major Vessels
- Thal

The target variable used for classification is `heart disease`.

The dataset contains 270 records and 13 input features.

## Algorithms Used

The following algorithms were applied:

- K-Nearest Neighbors (KNN)
- Random Forest
- Support Vector Machine (SVM)

Before applying the algorithms, the data was checked for missing values and duplicate records. The dataset was then divided into training and testing sets. Feature scaling was applied where required.

## Performance Comparison

The models were evaluated using accuracy, precision, recall and F1-score.

| Model                        |   Accuracy | Precision |   Recall   |  F1-Score  |
| ---------------------------- | ---------- | --------- | ---------- | ---------- |
| K-Nearest Neighbors (KNN)    |   79.63%   |   88.00%  |   73.33%   |  80.00%    |
| Random Forest                |   81.4     |   85.71%  |   80.00%   |  82.76%    |
| Support Vector Machine (SVM) |   81.48%   |   85.71%  |   80.00%   |  82.76%    |


Confusion matrices were also used to evaluate how the models classified the heart disease categories.

## Result

Among the three algorithms, **Random Forest and Support Vector Machine performed equally well**, achieving an accuracy of **81.48%**.

Random Forest was identified as the best-performing model by the program because it was selected as the best model when the accuracy values were tied.

K-Nearest Neighbors achieved an accuracy of **79.63%**, while Random Forest and Support Vector Machine achieved **81.48%**.

## Tools Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- VS Code

## Conclusion

This project demonstrates how different Machine Learning classification algorithms can be applied to predict heart disease.

K-Nearest Neighbors, Random Forest, and Support Vector Machine were compared using Accuracy, Precision, Recall and F1-Score. Random Forest and Support Vector Machine achieved the highest accuracy of 81.48%, with Random Forest selected as the best-performing algorithm by the program.

The notebook was developed using Kaggle. 
