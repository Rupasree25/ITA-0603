import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier

data = {
    "ContainsLink": ["Yes","Yes","No","Yes","No","Yes","Yes","No","Yes","No"],
    "OfferWords": ["Yes","Yes","No","No","Yes","Yes","No","No","Yes","No"],
    "UnknownSender": ["Yes","Yes","No","Yes","No","Yes","Yes","Yes","No","No"],
    "Attachment": ["No","Yes","No","No","Yes","No","Yes","No","No","Yes"],
    "ManyRecipients": ["Yes","Yes","No","Yes","No","No","Yes","No","Yes","No"],
    "Spam": ["Spam","Spam","Not Spam","Spam","Not Spam",
             "Spam","Spam","Not Spam","Spam","Not Spam"]
}

df = pd.DataFrame(data)

print("Email Spam Detection Dataset\n")
print(df)

le1 = LabelEncoder()
le2 = LabelEncoder()
le3 = LabelEncoder()
le4 = LabelEncoder()
le5 = LabelEncoder()
le6 = LabelEncoder()

df["ContainsLink"] = le1.fit_transform(df["ContainsLink"])
df["OfferWords"] = le2.fit_transform(df["OfferWords"])
df["UnknownSender"] = le3.fit_transform(df["UnknownSender"])
df["Attachment"] = le4.fit_transform(df["Attachment"])
df["ManyRecipients"] = le5.fit_transform(df["ManyRecipients"])
df["Spam"] = le6.fit_transform(df["Spam"])

X = df[["ContainsLink","OfferWords","UnknownSender","Attachment","ManyRecipients"]]
y = df["Spam"]

model = MLPClassifier(
    hidden_layer_sizes=(6,4),
    activation="relu",
    solver="adam",
    max_iter=1000,
    random_state=1
)

model.fit(X, y)

new_email = [[
    le1.transform(["Yes"])[0],
    le2.transform(["Yes"])[0],
    le3.transform(["Yes"])[0],
    le4.transform(["No"])[0],
    le5.transform(["Yes"])[0]
]]

prediction = model.predict(new_email)

print("\nNew Email")
print("Contains Link : Yes")
print("Offer Words : Yes")
print("Unknown Sender : Yes")
print("Attachment : No")
print("Many Recipients : Yes")

print("\nSpam Prediction :", le6.inverse_transform(prediction)[0])
