
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import string
import re
from nltk.corpus import stopwords
import nltk
import nltk.corpus
from nltk.tokenize import word_tokenize

import joblib
from sklearn.pipeline import Pipeline

exclude = string.punctuation
#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('omw-1.4')
def remove_tags(raw_text):
    cleaned_text = re.sub(re.compile('<.*?>'), '', raw_text)
    return cleaned_text
from nltk.stem import wordnet
from nltk.stem import WordNetLemmatizer

def lem(text):
    lemmatizer=WordNetLemmatizer()
    for i in text:
        lemmatizer.lemmatize(i)

def punc(text):
    return text.translate(str.maketrans('', '', exclude))

# Load the dataset
tf = pd.read_csv("C:/Users/USER/source/repos/Emotionprediction/Emotionprediction/data_train.csv")
data = tf.dropna()


X = data['text']  # Assuming 'text' is the column containing text data
y = data['emotion']  # Assuming 'emotion' is the column containing emotion labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Create a TfidfVectorizer to convert text into numerical features
tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)  # Fit and transform on training data
X_test_tfidf = tfidf.transform(X_test)

# Initialize the SVM classifier
svm = SVC(probability=True)

# Fit the SVM model on the training data
svm.fit(X_train_tfidf, y_train)

# Predict using the trained SVM model
y_pred_svm = svm.predict(X_test_tfidf)

# Evaluate the accuracy of the SVM model
accuracy_svm = accuracy_score(y_test, y_pred_svm)
raw_text="Despite the rainy weather, I felt quite cheerful when I saw my old friend"
new_sentence_tfidf = tfidf.transform([raw_text])
print("SVM Accuracy:", accuracy_svm)
probability_scores = svm.predict_proba(new_sentence_tfidf)

prediction = svm.predict(new_sentence_tfidf)
emotion_labels = svm.classes_  # This gives you the list of class labels
probability = dict(zip(emotion_labels, probability_scores[0]))  # Assuming you have one sample
print("Emotion Probabilities:", probability)


# Define the pipeline 
pipe_lr = Pipeline([('tfidf', tfidf),('svm', svm)])

joblib.dump(pipe_lr, 'C:/Users/USER/source/repos/Emotionprediction/Emotionprediction/emotion_classification_pipeline.pkl')


