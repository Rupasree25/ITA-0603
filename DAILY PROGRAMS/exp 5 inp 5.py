import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

data = {
    "Experience":[10,8,7,3,2,9,4,8,2,6],
    "PerformanceScore":[95,90,88,65,60,93,70,89,58,85],
    "TrainingHours":[50,45,40,20,18,48,25,42,15,38],
    "Promotion":["Yes","Yes","Yes","No","No","Yes","No","Yes","No","Yes"]
}

df = pd.DataFrame(data)

print("Employee Promotion Dataset\n")
print(df)

le = LabelEncoder()
df["Promotion"] = le.fit_transform(df["Promotion"])

X = df[["Experience","PerformanceScore","TrainingHours"]]
y = df["Promotion"]

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

new_employee = [[8,90,45]]

prediction = model.predict(new_employee)

print("\nNew Employee")
print("Experience:", 8)
print("Performance Score:", 90)
print("Training Hours:", 45)

print("\nPromotion Prediction:", le.inverse_transform(prediction)[0])
