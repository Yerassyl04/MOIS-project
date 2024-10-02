# Import necessary libraries
import numpy as np
import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score
import tkinter as tk
from tkinter import messagebox

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

# Load dataset from CSV file
df = pd.read_csv('spam.csv', encoding='ISO-8859-1')

# Data cleaning and preprocessing
df = df[['v1', 'v2']]  # Only keep relevant columns
df.columns = ['label', 'message']  # Rename columns
df.dropna(inplace=True)  # Drop missing values
df['label'] = df['label'].map({'ham': 0, 'spam': 1})  # Convert labels to 0 and 1

# Function to preprocess the text
ps = PorterStemmer()

def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    tokens = nltk.word_tokenize(text)  # Tokenize the text
    tokens = [word for word in tokens if word.isalnum()]  # Remove special characters
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
    tokens = [ps.stem(word) for word in tokens]  # Perform stemming
    return " ".join(tokens)

df['message'] = df['message'].apply(preprocess_text)  # Apply preprocessing

# Feature extraction using TfidfVectorizer
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['message']).toarray()
y = df['label'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.4f}")
print(f"Model Precision: {precision:.4f}")

# Function to predict spam or ham
def predict_spam(message):
    processed_message = preprocess_text(message)
    vectorized_message = tfidf.transform([processed_message])
    prediction = model.predict(vectorized_message)[0]
    return "Spam" if prediction == 1 else "Not Spam"

# Creating the Tkinter GUI
def check_spam():
    email_message = entry.get("1.0", tk.END).strip()
    if not email_message:
        messagebox.showwarning("Input Error", "Please enter an email message.")
        return
    result = predict_spam(email_message)
    messagebox.showinfo("Prediction", f"The message is: {result}")

# Initialize Tkinter window
window = tk.Tk()
window.title("Email Spam Prediction")

# Create and place widgets
label = tk.Label(window, text="Enter Email Message:")
label.pack(pady=10)

entry = tk.Text(window, height=10, width=50)
entry.pack(pady=10)

button = tk.Button(window, text="Check Spam", command=check_spam)
button.pack(pady=10)

# Run the Tkinter event loop
window.mainloop()
