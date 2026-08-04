import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

data = {
    "Weight":[150,160,140,120,125,130,155,118,148,122],
    "Diameter":[7.5,7.8,7.2,5.5,5.8,6.0,7.6,5.4,7.4,5.7],
    "Sweetness":[90,88,91,70,72,74,89,69,92,71],
    "Fruit":["Apple","Apple","Apple","Orange","Orange",
             "Orange","Apple","Orange","Apple","Orange"]
}

df = pd.DataFrame(data)

print("Fruit Classification Dataset\n")
print(df)

le = LabelEncoder()
df["Fruit"] = le.fit_transform(df["Fruit"])

X = df[["Weight","Diameter","Sweetness"]]
y = df["Fruit"]

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

new_fruit = [[152,7.5,90]]

prediction = model.predict(new_fruit)

print("\nNew Fruit")
print("Weight:", 152)
print("Diameter:", 7.5)
print("Sweetness:", 90)

print("\nFruit Prediction:", le.inverse_transform(prediction)[0])
