import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier

data = {
    "Income": ["High","High","Medium","Medium","Low","Low","High","Medium","High","Low"],
    "CreditScore": ["Good","Good","Good","Average","Poor","Average","Average","Good","Good","Poor"],
    "Employment": ["Permanent","Permanent","Permanent","Permanent","Temporary","Temporary","Permanent","Temporary","Permanent","Temporary"],
    "Property": ["Yes","No","Yes","No","No","Yes","Yes","No","Yes","Yes"],
    "LoanApproved": ["Yes","Yes","Yes","Yes","No","No","Yes","No","Yes","No"]
}

df = pd.DataFrame(data)

print("Loan Approval Dataset\n")
print(df)

le1 = LabelEncoder()
le2 = LabelEncoder()
le3 = LabelEncoder()
le4 = LabelEncoder()
le5 = LabelEncoder()

df["Income"] = le1.fit_transform(df["Income"])
df["CreditScore"] = le2.fit_transform(df["CreditScore"])
df["Employment"] = le3.fit_transform(df["Employment"])
df["Property"] = le4.fit_transform(df["Property"])
df["LoanApproved"] = le5.fit_transform(df["LoanApproved"])

X = df[["Income","CreditScore","Employment","Property"]]
y = df["LoanApproved"]

model = MLPClassifier(hidden_layer_sizes=(6,4),
                      activation='relu',
                      solver='adam',
                      max_iter=1000,
                      random_state=1)

model.fit(X, y)

new_applicant = [[
    le1.transform(["High"])[0],
    le2.transform(["Good"])[0],
    le3.transform(["Permanent"])[0],
    le4.transform(["Yes"])[0]
]]

prediction = model.predict(new_applicant)

print("\nNew Applicant")
print("Income : High")
print("Credit Score : Good")
print("Employment : Permanent")
print("Property : Yes")

print("\nLoan Approval Prediction :", le5.inverse_transform(prediction)[0])
