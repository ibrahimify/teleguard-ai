import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load preprocessed data
try:
    X_train, X_test, y_train, y_test = joblib.load("data/preprocessed_data.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    print("Preprocessed data loaded successfully!")
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# Step 1: Train Multinomial Naive Bayes
model_nb = MultinomialNB()
print("\nTraining Multinomial Naive Bayes...")
model_nb.fit(X_train, y_train)
print("Naive Bayes training completed!")

# Step 2: Train Logistic Regression with class weighting
model_lr = LogisticRegression(max_iter=1000, class_weight='balanced')
print("\nTraining Logistic Regression...")
model_lr.fit(X_train, y_train)
print("Logistic Regression training completed!")

# Step 3: Train XGBoost
model_xgb = XGBClassifier(eval_metric='logloss')
print("\nTraining XGBoost...")
model_xgb.fit(X_train, y_train)
print("XGBoost training completed!")

# Step 4: Voting Classifier
model_voting = VotingClassifier(
    estimators=[('nb', model_nb), ('lr', model_lr), ('xgb', model_xgb)],
    voting='soft'
)
print("\nTraining Voting Classifier...")
model_voting.fit(X_train, y_train)
print("Voting Classifier training completed!")

# Step 5: Evaluation
y_pred_voting = model_voting.predict(X_test)
accuracy_voting = accuracy_score(y_test, y_pred_voting)
print(f"\nVoting Classifier Accuracy: {accuracy_voting * 100:.2f}%")
print("\nVoting Classifier Classification Report:")
print(classification_report(y_test, y_pred_voting))
print("\nVoting Classifier Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_voting))

# Save the Voting Classifier
joblib.dump(model_voting, "models/spam_classifier_voting.pkl")
print("\nVoting model saved successfully!")
