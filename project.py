import pandas as pd
data = pd.read_csv('Credit_data.csv')  # Use the exact filename
print(data.head())
# Clean column names (strip extra spaces)
data.columns = data.columns.str.strip()

# Check for missing values and summary statistics again
print(data.isnull().sum())
print(data.describe())

# Encode categorical variables (for example, if 'default' is the target variable)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# Encode the 'default' column (since it's categorical)
data['default'] = le.fit_transform(data['default'])

# Encode other categorical columns
data['SEX'] = le.fit_transform(data['SEX'])
data['EDUCATION'] = le.fit_transform(data['EDUCATION'])
data['MARRIAGE'] = le.fit_transform(data['MARRIAGE'])

# Now, create the heatmap
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Define features and target
X = data.drop('default', axis=1)
y = data['default']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model: Random Forest
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)

# Evaluation metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
# Clean column names to remove trailing spaces
data.columns = data.columns.str.strip()

# Confirm column names
print("Columns in the dataset:", data.columns)

# Check if 'default' is in the dataset
if 'default' in data.columns:
    X = data.drop('default', axis=1)
    y = data['default']


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Data split successful!")
else:
    print("'default' column not found in the dataset!")
model = LogisticRegression(class_weight='balanced')
model.fit(X_train_scaled, y_train)
metrics import classification_report, accuracy_score
import LogisticRegression
# Initialize and train the model without class_weight='balanced'
model_no_weight = LogisticRegression()
model_no_weight.fit(X_train_scaled, y_train)

# Make predictions
y_pred_no_weight = model_no_weight.predict(X_test_scaled)

# Evaluate the model without class_weight='balanced'
print("Results without class_weight='balanced':")
print("Accuracy:", accuracy_score(y_test, y_pred_no_weight))
print("Classification Report:\n", classification_reporfrom sklearn.linear_model 
from skt(y_test, y_pred_no_weight))

# Initialize and train the model with class_weight='balanced'
model_weighted = LogisticRegression(class_weight='balanced')
model_weighted.fit(X_train_scaled, y_train)learn

# Make predictions
y_pred_weighted = model_weighted.predict(X_test_scaled)

# Evaluate the model with class_weight='balanced'
print("Results with class_weight='balanced':")
print("Accuracy:", accuracy_score(y_test, y_pred_weighted))
print("Classification Report:\n", classification_report(y_test, y_pred_weighted))
from sklearn.model_selection import GridSearchCV

params = {'C': [0.1, 1, 10, 100], 'solver': ['lbfgs', 'liblinear']}
grid = GridSearchCV(LogisticRegression(), param_grid=params, scoring='accuracy', cv=5)
grid.fit(X_train_scaled, y_train)

print("Best Parameters:", grid.best_params_)
model = grid.best_estimator_
from sklearn.metrics import roc_auc_score, confusion_matrix
import seaborn as sns

# ROC-AUC Score
y_prob = model.predict_proba(X_test_scaled)[:, 1]
roc_auc = roc_auc_score(y_test, y_prob)
print("ROC-AUC Score:", roc_auc)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
import pandas as pd
coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_[0]})
coefficients = coefficients.sort_values(by='Coefficient', ascending=False)
print(coefficients)
import joblib

# Save the model
joblib.dump(model, 'scaled_logistic_model.pkl')

# Load the model
loaded_model = joblib.load('scaled_logistic_model.pkl')
