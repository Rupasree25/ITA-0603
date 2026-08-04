import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier

data = {
    "Experience": ["High","High","Medium","Medium","Low","High","Medium","Low","High","Low"],
    "Performance": ["Excellent","Good","Good","Average","Poor","Excellent","Good","Average","Good","Poor"],
    "Leadership": ["Yes","Yes","Yes","No","No","Yes","No","No","Yes","No"],
    "Training": ["Yes","Yes","Yes","Yes","No","No","Yes","No","Yes","Yes"],
    "Promotion": ["Promoted","Promoted","Promoted","Not Promoted","Not Promoted",
                  "Promoted","Promoted","Not Promoted","Promoted","Not Promoted"]
}

df = pd.DataFrame(data)

print("Employee Promotion Dataset\n")
print(df)

le1 = LabelEncoder()
le2 = LabelEncoder()
le3 = LabelEncoder()
le4 = LabelEncoder()
le5 = LabelEncoder()

df["Experience"] = le1.fit_transform(df["Experience"])
df["Performance"] = le2.fit_transform(df["Performance"])
df["Leadership"] = le3.fit_transform(df["Leadership"])
df["Training"] = le4.fit_transform(df["Training"])
df["Promotion"] = le5.fit_transform(df["Promotion"])

X = df[["Experience","Performance","Leadership","Training"]]
y = df["Promotion"]

model = MLPClassifier(
    hidden_layer_sizes=(6,4),
    activation="relu",
    solver="adam",
    max_iter=1000,
    random_state=1
)

model.fit(X, y)

new_employee = [[
    le1.transform(["High"])[0],
    le2.transform(["Excellent"])[0],
    le3.transform(["Yes"])[0],
    le4.transform(["Yes"])[0]
]]

prediction = model.predict(new_employee)

print("\nNew Employee")
print("Experience : High")
print("Performance : Excellent")
print("Leadership : Yes")
print("Training : Yes")

print("\nPromotion Prediction :", le5.inverse_transform(prediction)[0])
