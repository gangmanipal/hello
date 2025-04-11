import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# Load the trained stacked model
with open('models/StackedModel.pkl', 'rb') as model_file:
    stacked_model = pickle.load(model_file)

# Load the label encoder
with open('models/LabelEncoder.pkl', 'rb') as encoder_file:
    label_encoder = pickle.load(encoder_file)

# Load dataset (same dataset used for training)
file_path = "Data/Crop_recommendation.csv"
df = pd.read_csv(file_path)
X = df.drop(columns=['label'])
y = df['label']

# Encode labels
y_encoded = label_encoder.transform(y)

# Split data (ensure same test size and stratification as before)
_, X_test, _, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Predictions using the loaded model
y_pred = stacked_model.predict(X_test)

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Stacked Model')
plt.show()
