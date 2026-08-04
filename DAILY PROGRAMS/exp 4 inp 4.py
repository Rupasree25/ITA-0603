import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier

data = {
    "Fever": ["Yes","Yes","No","Yes","No","Yes","No","Yes","Yes","No"],
    "Cough": ["Yes","Yes","Yes","No","No","Yes","Yes","No","Yes","No"],
    "Headache": ["Yes","No","Yes","Yes","No","Yes","No","No","Yes","Yes"],
    "BodyPain": ["Yes","Yes","No","Yes","No","No","Yes","Yes","Yes","No"],
    "Fatigue": ["Yes","Yes","No","Yes","No","Yes","No","Yes","Yes","No"],
    "Disease": ["Positive","Positive","Negative","Positive","Negative",
                "Positive","Negative","Positive","Positive","Negative"]
}

df = pd.DataFrame(data)

print("Disease Diagnosis Dataset\n")
print(df)

le1 = LabelEncoder()
le2 = LabelEncoder()
le3 = LabelEncoder()
le4 = LabelEncoder()
le5 = LabelEncoder()
le6 = LabelEncoder()

df["Fever"] = le1.fit_transform(df["Fever"])
df["Cough"] = le2.fit_transform(df["Cough"])
df["Headache"] = le3.fit_transform(df["Headache"])
df["BodyPain"] = le4.fit_transform(df["BodyPain"])
df["Fatigue"] = le5.fit_transform(df["Fatigue"])
df["Disease"] = le6.fit_transform(df["Disease"])

X = df[["Fever","Cough","Headache","BodyPain","Fatigue"]]
y = df["Disease"]

model = MLPClassifier(hidden_layer_sizes=(6,4),
                      activation="relu",
                      solver="adam",
                      max_iter=1000,
                      random_state=1)

model.fit(X, y)

new_patient = [[
    le1.transform(["Yes"])[0],
    le2.transform(["Yes"])[0],
    le3.transform(["Yes"])[0],
    le4.transform(["Yes"])[0],
    le5.transform(["Yes"])[0]
]]

prediction = model.predict(new_patient)

print("\nNew Patient")
print("Fever : Yes")
print("Cough : Yes")
print("Headache : Yes")
print("Body Pain : Yes")
print("Fatigue : Yes")

print("\nDisease Prediction :", le6.inverse_transform(prediction)[0])
