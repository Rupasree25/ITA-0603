import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

data = {
    "Income":[8,7,6,4,3,9,5,8,4,7],
    "CreditScore":[780,760,720,620,580,800,650,770,600,740],
    "LoanAmount":[5,4,6,8,10,4,7,5,9,6],
    "LoanApproved":["Yes","Yes","Yes","No","No","Yes","No","Yes","No","Yes"]
}

df = pd.DataFrame(data)

print("Loan Approval Dataset\n")
print(df)

le = LabelEncoder()
df["LoanApproved"] = le.fit_transform(df["LoanApproved"])

X = df[["Income","CreditScore","LoanAmount"]]
y = df["LoanApproved"]

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

new_applicant = [[8, 775, 5]]

prediction = model.predict(new_applicant)

print("\nNew Applicant")
print("Income:", 8)
print("Credit Score:", 775)
print("Loan Amount:", 5)

print("\nLoan Approval Prediction:", le.inverse_transform(prediction)[0])
