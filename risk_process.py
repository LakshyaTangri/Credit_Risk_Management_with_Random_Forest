from data_preprocessing import load_and_preprocess_data, split_data
from model_training import scale_features, train_model, evaluate_model, plot_feature_importance
from risk_assessment import assess_credit_risk

data_path = "credit_risk_dataset.csv"
data = load_and_preprocess_data(data_path)

X_train, X_test, y_train, y_test = split_data(data)
X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

rf_model = train_model(X_train_scaled, y_train)

y_pred = rf_model.predict(X_test_scaled)
evaluate_model(y_test, y_pred)
plot_feature_importance(X, rf_model)

# Example risk assessment
new_application = {
    # ... new application data
}
risk_label, risk_probability = assess_credit_risk(new_application, X, scaler)
print(f"Risk Label: {risk_label}, Risk Probability: {risk_probability}")
