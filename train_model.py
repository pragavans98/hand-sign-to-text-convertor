import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
import pickle

# Load dataset
df = pd.read_csv("dataset/landmarks/landmarks.csv")
X = df.drop("label", axis=1)
y = df["label"]

# Define MLP model
model = MLPClassifier(hidden_layer_sizes=(128, 64),
                      activation='relu',
                      solver='adam',
                      max_iter=500,
                      random_state=42)

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv)
print("Cross-validation accuracy:", scores.mean())

# Train
model.fit(X, y)

# Evaluate
y_pred = model.predict(X)
print("Confusion Matrix:\n", confusion_matrix(y, y_pred))
print("Classification Report:\n", classification_report(y, y_pred))

# Save model
os.makedirs("models", exist_ok=True)
with open("models/sign_model_mlp.pkl", "wb") as f:
    pickle.dump(model, f)

print("MLP model trained and saved to models/sign_model_mlp.pkl") 