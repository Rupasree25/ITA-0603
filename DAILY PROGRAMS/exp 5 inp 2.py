import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

data = {
    "CGPA":[9.2,8.8,8.5,7.2,6.8,9.0,7.5,8.6,6.5,8.9],
    "AptitudeScore":[90,85,80,72,68,92,75,84,60,88],
    "CommunicationScore":[88,82,79,70,65,90,74,81,58,86],
    "Placement":["Placed","Placed","Placed","Not Placed","Not Placed",
                 "Placed","Not Placed","Placed","Not Placed","Placed"]
}

df = pd.DataFrame(data)

print("Student Placement Dataset\n")
print(df)

le = LabelEncoder()
df["Placement"] = le.fit_transform(df["Placement"])

X = df[["CGPA","AptitudeScore","CommunicationScore"]]
y = df["Placement"]

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

new_student = [[8.7, 86, 84]]

prediction = model.predict(new_student)

print("\nNew Student")
print("CGPA:", 8.7)
print("Aptitude Score:", 86)
print("Communication Score:", 84)

print("\nPrediction:", le.inverse_transform(prediction)[0])
