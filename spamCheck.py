import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load the data into a pandas DataFrame
data = pd.read_csv("emails.csv")

# Create a feature matrix by using the text of the emails as input
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['text'])

# Create the target variable
y = data['label']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train a Multinomial Naive Bayes model on the training data
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Print the accuracy of the model
print("Accuracy: ", accuracy_score(y_test, y_pred))
