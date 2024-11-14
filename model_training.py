from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from data_preprocessing import split_data

def scale_features(X_train, X_test):
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)
  return X_train_scaled, X_test_scaled

def train_model(X_train_scaled, y_train, model_params={'n_estimators': 100, 'random_state': 42}):
  rf_model = RandomForestClassifier(**model_params)
  rf_model.fit(X_train_scaled, y_train)
  return rf_model

def evaluate_model(y_test, y_pred):
  print(classification_report(y_test, y_pred))
  print(confusion_matrix(y_test, y_pred))

def plot_feature_importance(X, rf_model):
  import matplotlib.pyplot as plt
  import seaborn as sns

  feature_importance = pd.DataFrame({'feature': X.columns, 'importance': rf_model.feature_importances_})
  feature_importance = feature_importance.sort_values('importance', ascending=False)
  plt.figure(figsize=(10, 6))
  sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
  plt.title('Top 10 Most Important Features')
  plt.show()
