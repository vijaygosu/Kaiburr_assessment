# Kaiburr Assessment - Task 5: Data Science Example
#Candidate: Vijay Gosu 
#Task: Perform Text Classification on Consumer Complaint Dataset  
#Steps: 
#1. Explanatory Data Analysis and Feature Engineering  
#2. Text Pre-Processing  
#3. Model Selection  
#4. Comparison of Model Performance  
#5. Model Evaluation  
#6. Prediction on New Complaint

# --- Import necessary libraries ---
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

# --- Download NLTK data (only needs to be done once) ---
try:
    stopwords.words('english')
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords')
try:
    # This is for the lemmatizer
    nltk.data.find('corpora/wordnet.zip')
except LookupError:
    print("Downloading NLTK wordnet...")
    nltk.download('wordnet')


# --- 1. Explanatory Data Analysis and Feature Engineering ---
print("--- Step 1: Loading and Preparing Data ---")

# Load the dataset from the provided URL
DATA_PATH = "complaints.csv"
try:
    df = pd.read_csv(DATA_PATH, on_bad_lines='skip')
    print(f"Dataset loaded successfully with {len(df)} rows.")
except Exception as e:
    print(f"Failed to load dataset. Error: {e}")
    exit()

# Define the target categories as specified in the task
TARGET_PRODUCTS = {
    "Credit reporting, repair, or other": "Credit reporting, credit repair services, or other personal consumer reports",
    "Debt collection": "Debt collection",
    "Consumer Loan": "Consumer Loan",
    "Mortgage": "Mortgage"
}

# Filter the dataframe to only include the products we are interested in
df_filtered = df[df['Product'].isin(TARGET_PRODUCTS.values())].copy()
print(f"Filtered to {len(df_filtered)} rows with target products.")

# Drop rows where the complaint narrative is missing, as it's our feature
df_filtered.dropna(subset=['Consumer complaint narrative'], inplace=True)
print(f"Kept {len(df_filtered)} rows with non-empty complaint narratives.")


# Create a mapping from product names to numerical labels (0, 1, 2, 3)
product_to_label = {
    "Credit reporting, credit repair services, or other personal consumer reports": 0,
    "Debt collection": 1,
    "Consumer Loan": 2,
    "Mortgage": 3
}
df_filtered['label'] = df_filtered['Product'].map(product_to_label)
label_to_product = {v: k for k, v in product_to_label.items()}

# Display the class distribution
print("\nClass Distribution:")
print(df_filtered['Product'].value_counts())
print("\n")


# --- 2. Text Pre-Processing ---
print("--- Step 2: Text Pre-Processing ---")

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Cleans and prepares text data for modeling."""
    # 1. Convert to lowercase
    text = text.lower()
    # 2. Remove punctuation and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    # 3. Tokenize
    words = text.split()
    # 4. Remove stopwords and lemmatize
    cleaned_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(cleaned_words)

# Apply the preprocessing function to the complaint narratives
print("Applying preprocessing to complaint narratives...")
df_filtered['cleaned_narrative'] = df_filtered['Consumer complaint narrative'].apply(preprocess_text)
print("Preprocessing complete.")
print("\n")


# Define features (X) and target (y)
X = df_filtered['cleaned_narrative']
y = df_filtered['label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Data split into {len(X_train)} training samples and {len(X_test)} testing samples.")
print("\n")

# --- 3. Selection of Multi Classification Models ---
models = {
    "Multinomial Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Linear SVM": LinearSVC(random_state=42)
}

# --- 4 & 5. Comparison of Model Performance & Evaluation ---
print("--- Steps 4 & 5: Training, Comparing, and Evaluating Models ---")

best_model = None
best_f1_score = 0.0
best_model_name = ""

for model_name, model_instance in models.items():
    print(f"--- Training and Evaluating {model_name} ---")

    # Create a pipeline that first vectorizes the text and then applies the classifier
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('classifier', model_instance)
    ])

    # Train the model
    pipeline.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = pipeline.predict(X_test)

    # Evaluate performance
    report = classification_report(y_test, y_pred, target_names=TARGET_PRODUCTS.keys(), output_dict=True)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=TARGET_PRODUCTS.keys()))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\n")

    # Check if this model is the best one so far based on weighted F1-score
    weighted_f1 = report['weighted avg']['f1-score']
    if weighted_f1 > best_f1_score:
        best_f1_score = weighted_f1
        best_model = pipeline
        best_model_name = model_name

print(f"Best performing model is: {best_model_name} with a weighted F1-score of {best_f1_score:.4f}")
print("\n")

# --- 6. Prediction ---
print("--- Step 6: Prediction on a New Complaint ---")

# A new, unseen complaint
new_complaint = """
I am writing to dispute a charge on my mortgage account.
My bank, Acme Bank, has incorrectly charged me a late fee,
but I sent the payment well before the due date. I have bank
records to prove it. This has negatively affected my credit score.
"""

# Use the best model to predict the category
predicted_label = best_model.predict([new_complaint])[0]
predicted_product = label_to_product[predicted_label]

print(f"New Complaint Text:\n'{new_complaint.strip()}'")
print("-" * 30)
print(f"Predicted Category ID: {predicted_label}")
print(f"Predicted Product: '{predicted_product}'")
