import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load and vectorize raw data
try:
    df = pd.read_csv("data/cleaned_spam.csv")
    texts = df['message'].astype(str).tolist()
    labels = df['label'].tolist()

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)

    # Save vectorizer
    joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

    # Save preprocessed data
    joblib.dump((X_train, X_test, y_train, y_test), "data/preprocessed_data.pkl")
    print("✅ Vectorizer and preprocessed data saved successfully!\n")

except Exception as e:
    print(f"❌ Error during preprocessing: {e}")
    exit()

# Step 2: Train Multinomial Naive Bayes
model_nb = MultinomialNB()
print("🔧 Training Multinomial Naive Bayes...")
model_nb.fit(X_train, y_train)

# Step 3: Train Logistic Regression
model_lr = LogisticRegression(max_iter=1000, class_weight='balanced')
print("🔧 Training Logistic Regression...")
model_lr.fit(X_train, y_train)

# Step 4: Train XGBoost
model_xgb = XGBClassifier(eval_metric='logloss')
print("🔧 Training XGBoost...")
model_xgb.fit(X_train, y_train)

# Step 5: Create and train Voting Classifier
model_voting = VotingClassifier(
    estimators=[('nb', model_nb), ('lr', model_lr), ('xgb', model_xgb)],
    voting='soft'
)
print("🔧 Training Voting Classifier...")
model_voting.fit(X_train, y_train)

# Step 6: Evaluate
y_pred_voting = model_voting.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_voting)
print(f"\n✅ Voting Classifier Accuracy: {accuracy * 100:.2f}%\n")
print("📊 Classification Report:")
print(classification_report(y_test, y_pred_voting))
print("📊 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_voting))

# Step 7: Save models
joblib.dump(model_nb, "models/spam_classifier_nb.pkl")
joblib.dump(model_lr, "models/spam_classifier_lr.pkl")
joblib.dump(model_voting, "models/spam_classifier_voting.pkl")
print("\n💾 All models saved successfully!")
