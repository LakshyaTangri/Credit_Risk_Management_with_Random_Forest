import pandas as pd

def load_and_preprocess_data(data_path):
  data = pd.read_csv(data_path)
  data = data.dropna()
  print(data.isnull().sum())

  # Encode categorical variables
  data = pd.get_dummies(data, columns=['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file'])

  return data

def split_data(data, test_size=0.2, random_state=42):
  X = data.drop('loan_status', axis=1)
  y = data['loan_status']

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

  return X_train, X_test, y_train, y_test
