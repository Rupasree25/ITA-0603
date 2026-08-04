import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

data = {
    "Temperature":[39.0,38.5,37.0,39.2,36.8,38.8,37.2,39.1,36.7,38.9],
    "HeartRate":[110,108,78,115,75,112,80,114,74,111],
    "OxygenLevel":[94,95,99,93,98,94,98,92,99,93],
    "Disease":["Positive","Positive","Negative","Positive","Negative",
               "Positive","Negative","Positive","Negative","Positive"]
}

df = pd.DataFrame(data)

print("Disease Diagnosis Dataset\n")
print(df)

le = LabelEncoder()
df["Disease"] = le.fit_transform(df["Disease"])

X = df[["Temperature","HeartRate","OxygenLevel"]]
y = df["Disease"]

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

new_patient = [[38.9, 110, 94]]

prediction = model.predict(new_patient)

print("\nNew Patient")
print("Temperature:", 38.9)
print("Heart Rate:", 110)
print("Oxygen Level:", 94)

print("\nDisease Prediction:", le.inverse_transform(prediction)[0])
