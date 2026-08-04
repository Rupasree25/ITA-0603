import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier

data = {
    "CGPA":[9,8,7,6,5,9,8,6,7,5],
    "Communication":["Excellent","Good","Good","Average","Poor","Excellent","Good","Average","Excellent","Poor"],
    "Internship":["Yes","Yes","Yes","No","No","Yes","No","Yes","Yes","No"],
    "Programming":["Excellent","Good","Average","Average","Poor","Good","Good","Average","Good","Average"],
    "Placement":["Placed","Placed","Placed","Not Placed","Not Placed","Placed","Placed","Not Placed","Placed","Not Placed"]
}

df = pd.DataFrame(data)

print("Student Placement Dataset\n")
print(df)

le1 = LabelEncoder()
le2 = LabelEncoder()
le3 = LabelEncoder()
le4 = LabelEncoder()

df["Communication"] = le1.fit_transform(df["Communication"])
df["Internship"] = le2.fit_transform(df["Internship"])
df["Programming"] = le3.fit_transform(df["Programming"])
df["Placement"] = le4.fit_transform(df["Placement"])

X = df[["CGPA","Communication","Internship","Programming"]]
y = df["Placement"]

model = MLPClassifier(hidden_layer_sizes=(6,4), max_iter=1000, random_state=1)

model.fit(X,y)

new_student = [[8,
                le1.transform(["Excellent"])[0],
                le2.transform(["Yes"])[0],
                le3.transform(["Excellent"])[0]]]

prediction = model.predict(new_student)

print("\nPrediction:", le4.inverse_transform(prediction)[0])
