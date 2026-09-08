import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report

df = pd.read_csv("dataset_heart.csv")

print(df.head())

print("Shape:", df.shape)

print("\nColumns:")
print(df.columns)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDataset information:")
print(df.info())

df.columns = df.columns.str.strip()

print("Columns after cleaning:")
print(df.columns)

df.drop_duplicates(inplace=True)

print("Shape after removing duplicates:")
print(df.shape)

print("Heart disease values:")
print(df["heart disease"].value_counts())

print("Missing values after cleaning:")
print(df.isnull().sum())

sns.countplot(data=df, x="heart disease")

plt.title("Heart Disease Distribution")
plt.xlabel("Heart Disease")
plt.ylabel("Number of Patients")
plt.show()

sns.histplot(data=df, x="age", kde=True)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Patients")
plt.show()

sns.boxplot(
    data=df,
    x="heart disease",
    y="age"
)

plt.title("Age vs Heart Disease")
plt.xlabel("Heart Disease")
plt.ylabel("Age")
plt.show()

sns.countplot(
    data=df,
    x="sex",
    hue="heart disease"
)

plt.title("Sex vs Heart Disease")
plt.xlabel("Sex")
plt.ylabel("Number of Patients")
plt.show()

X = df.drop("heart disease", axis=1)
y = df["heart disease"]

print("Input features:")
print(X.columns)

print("\nTarget:")
print(y.name)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train_scaled, y_train)

y_pred_knn = knn.predict(X_test_scaled)

print("K-Nearest Neighbors")

print("Accuracy:",
      accuracy_score(y_test, y_pred_knn))

print("Precision:",
      precision_score(y_test, y_pred_knn))

print("Recall:",
      recall_score(y_test, y_pred_knn))

print("F1 Score:",
      f1_score(y_test, y_pred_knn))

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("\nRandom Forest")

print("Accuracy:",
      accuracy_score(y_test, y_pred_rf))

print("Precision:",
      precision_score(y_test, y_pred_rf))

print("Recall:",
      recall_score(y_test, y_pred_rf))

print("F1 Score:",
      f1_score(y_test, y_pred_rf))

svm = SVC()

svm.fit(X_train_scaled, y_train)

y_pred_svm = svm.predict(X_test_scaled)

print("\nSupport Vector Machine")

print("Accuracy:",
      accuracy_score(y_test, y_pred_svm))

print("Precision:",
      precision_score(y_test, y_pred_svm))

print("Recall:",
      recall_score(y_test, y_pred_svm))

print("F1 Score:",
      f1_score(y_test, y_pred_svm))

results = pd.DataFrame({
    "Model": [
        "K-Nearest Neighbors",
        "Random Forest",
        "Support Vector Machine"
    ],

    "Accuracy": [
        accuracy_score(y_test, y_pred_knn),
        accuracy_score(y_test, y_pred_rf),
        accuracy_score(y_test, y_pred_svm)
    ],

    "Precision": [
        precision_score(y_test, y_pred_knn),
        precision_score(y_test, y_pred_rf),
        precision_score(y_test, y_pred_svm)
    ],

    "Recall": [
        recall_score(y_test, y_pred_knn),
        recall_score(y_test, y_pred_rf),
        recall_score(y_test, y_pred_svm)
    ],

    "F1 Score": [
        f1_score(y_test, y_pred_knn),
        f1_score(y_test, y_pred_rf),
        f1_score(y_test, y_pred_svm)
    ]
})

print(results)

sns.barplot(
    data=results,
    x="Model",
    y="Accuracy"
)

plt.title("Model Accuracy Comparison")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.show()

cm = confusion_matrix(
    y_test,
    y_pred_knn
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d"
)

plt.title("KNN Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

cm = confusion_matrix(
    y_test,
    y_pred_rf
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d"
)

plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

cm = confusion_matrix(
    y_test,
    y_pred_svm
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d"
)

plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("K-Nearest Neighbors")
print(classification_report(y_test, y_pred_knn))

print("\nRandom Forest")
print(classification_report(y_test, y_pred_rf))

print("\nSupport Vector Machine")
print(classification_report(y_test, y_pred_svm))

best_model = results.loc[
    results["Accuracy"].idxmax()
]

print("\nBest Model:")
print(best_model)

print(
    "The models were used to predict whether a patient "
    "has heart disease."
)

print(
    "K-Nearest Neighbors, Random Forest and Support Vector "
    "Machine were compared using Accuracy, Precision, "
    "Recall and F1 Score."
)

print(
    "The model with the highest accuracy is:",
    best_model["Model"]
)