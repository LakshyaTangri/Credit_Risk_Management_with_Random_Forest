import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

data = pd.read_csv('credit_risk_dataset.csv')

data = data.dropna()

print (data.isnull().sum()) 

# Encode categorical variables
data = pd.get_dummies(data, columns=['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file'])

# Split features and target variable
X = data.drop('loan_status', axis=1)
y = data['loan_status']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = rf_model.predict(X_test_scaled)

# Evaluate the model
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Plot feature importance
feature_importance = pd.DataFrame({'feature': X.columns, 'importance': rf_model.feature_importances_})
feature_importance = feature_importance.sort_values('importance', ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
plt.title('Top 10 Most Important Features')
plt.show()

def assess_credit_risk(new_application):
    # Preprocess the new application data
    new_application_encoded = pd.get_dummies(new_application, columns=['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file'])
    
    # Ensure all columns from training data are present
    for col in X.columns:
        if col not in new_application_encoded.columns:
            new_application_encoded[col] = 0
    
    # Reorder columns to match training data
    new_application_encoded = new_application_encoded[X.columns]
    
    # Scale the features
    new_application_scaled = scaler.transform(new_application_encoded)
    
    # Predict risk
    risk_probability = rf_model.predict_proba(new_application_scaled)[0][1]
    risk_label = "High Risk" if risk_probability > 0.5 else "Low Risk"
    
    return risk_label, risk_probability