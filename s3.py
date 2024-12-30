import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pytesseract
import cv2
pytesseract.pytesseract.tesseract_cmd = r'C:\Tesseract\tesseract.exe'  # Update with your path
df = pd.read_csv('resume_dataset.csv')
def preprocess_text(text):
    text = text.lower()  # Convert text to lowercase
    text = ''.join([char for char in text if char.isalpha() or char.isspace()])  # Remove non-alphabetic characters
    return text

df['CV_combined'] = df['Education'].fillna('') + ' ' + df['Work Experience'].fillna('') + ' ' + df['Skills'].fillna('')
df['CV_cleaned'] = df['CV_combined'].apply(preprocess_text)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['CV_cleaned'])
le = LabelEncoder()
y = le.fit_transform(df['Name'])  # Assuming 'Name' column as the target for personality traits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred))
