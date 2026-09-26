import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Sample dataset (phishing aur safe emails)
data = {
    "text": [
        "Congratulations you won a lottery click here to claim now",
        "Your account has been suspended verify your password immediately",
        "Urgent update your bank details or account will be locked",
        "Click this link to reset your password http://fake-bank.com",
        "You have won a free gift card claim now limited offer",
        "Verify your identity by clicking this suspicious link",
        "Meeting scheduled for tomorrow at 10 AM in conference room",
        "Please find attached the report for this month",
        "Your order has been shipped and will arrive soon",
        "Lunch with the team this Friday, let me know if you can join",
        "Here is the project update as discussed in our meeting",
        "Happy birthday hope you have a wonderful day",
        "Reminder your electricity bill is due next week",
        "Thanks for your email, I will respond by tomorrow"
    ],
    "label": [
        "Phishing", "Phishing", "Phishing", "Phishing", "Phishing",
        "Phishing", "Safe", "Safe", "Safe", "Safe", "Safe", "Safe", "Safe", "Safe"
    ]
}

df = pd.DataFrame(data)

# Text ko numbers mein convert karo
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["label"]

# Train aur test data split karo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model train karo
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict karo
predictions = model.predict(X_test)

# Results dikhao
print("Phishing Email Detection Model")
print("Accuracy:", accuracy_score(y_test, predictions))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# Naye email ko test karo
def check_email(email_text):
    vec = vectorizer.transform([email_text])
    result = model.predict(vec)
    return result[0]

test_email = "Click here to verify your account password urgently"
print("\nTest Email:", test_email)
print("Prediction:", check_email(test_email))