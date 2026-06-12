import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=[
        "SepalLengthCm",
        "SepalWidthCm",
        "PetalLengthCm",
        "PetalWidthCm"
    ]
)

df["Species"] = [iris.target_names[i] for i in iris.target]

print("Dataset Loaded Successfully\n")

print("First 5 Rows")
print(df.head())

print("\nDataset Shape")
print(df.shape)

print("\nMissing Values")
print(df.isnull().sum())

plt.figure(figsize=(8, 5))

plt.scatter(
    df["SepalLengthCm"],
    df["PetalLengthCm"]
)

plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")

plt.savefig("iris_graph.png")
plt.close()

X = df.drop("Species", axis=1)
y = df["Species"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = KNeighborsClassifier(n_neighbors=3)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy")
print(round(accuracy * 100, 2), "%")

print("\nConfusion Matrix")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report")
print(classification_report(y_test, predictions))

print("\nEnter Flower Measurements")

sl = float(input("Sepal Length : "))
sw = float(input("Sepal Width  : "))
pl = float(input("Petal Length : "))
pw = float(input("Petal Width  : "))

flower = pd.DataFrame(
    [[sl, sw, pl, pw]],
    columns=[
        "SepalLengthCm",
        "SepalWidthCm",
        "PetalLengthCm",
        "PetalWidthCm"
    ]
)

result = model.predict(flower)

print("\nPredicted Species:")
print(result[0])