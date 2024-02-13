import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import RandomOverSampler
import re

# Loading the SMS dataset
# Assuming a CSV file with 'text' and 'label' columns.
df = pd.read_csv('D:/New folder/spam.csv', encoding='latin-1')

# Preprocess the text to handle non-ASCII characters
df['v2'] = df['v2'].apply(lambda x: re.sub(r'[^\x00-\x7F]+', ' ', x))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['v2'], df['v1'], test_size=0.2, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Address Class Imbalance (oversampling)
ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train_tfidf, y_train)

# Train Naive Bayes classifier without specifying class_prior
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_resampled, y_train_resampled)
nb_predictions = nb_classifier.predict(X_test_tfidf)
result_df = pd.DataFrame({'Message': X_test, 'Actual Label': y_test, 'Predicted Label': nb_predictions})
print(result_df)

# Evaluate Naive Bayes model
accuracy = accuracy_score(y_test, nb_predictions)
classification_rep = classification_report(y_test, nb_predictions, zero_division=1)

# Display model performance
print("Naive Bayes Classifier:")
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_rep)

# Input a message for prediction
input_message = input("\nEnter a message for spam prediction: ")
input_tfidf = vectorizer.transform([input_message])
prediction = nb_classifier.predict(input_tfidf)

# Display prediction for the entered message
print("\nPrediction for the new message:", prediction[0])

# Plotting accuracy vs. training set size
train_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
accuracies = []

for size in train_sizes:
    X_train_subset, _, y_train_subset, _ = train_test_split(X_train, y_train, train_size=size, random_state=42)
    X_train_tfidf_subset = vectorizer.transform(X_train_subset)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train_tfidf_subset, y_train_subset)
    nb_classifier.fit(X_train_resampled, y_train_resampled)
    nb_predictions = nb_classifier.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, nb_predictions)
    accuracies.append(accuracy)

plt.plot(train_sizes, accuracies, marker='o')
plt.title('Accuracy vs. Training Set Size')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()
