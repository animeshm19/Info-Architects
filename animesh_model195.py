import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
# from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df):
    # Drop unnecessary columns
    df.drop(['PatientID', 'DoctorInCharge'], axis=1, inplace=True)
    
    # Normalize and standardize numerical columns
    numerical_columns = [col for col in df.columns if df[col].nunique() > 10]
    min_max_scaler = MinMaxScaler()
    standard_scaler = StandardScaler()
    df[numerical_columns] = min_max_scaler.fit_transform(df[numerical_columns])
    df[numerical_columns] = standard_scaler.fit_transform(df[numerical_columns])
    
    return df

def split_data(df, target_column='Diagnosis'):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    return X_train, X_test, y_train, y_test

def plot_histogram(df, column, color='blue'):
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], color=color, kde=True)
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

def plot_boxplot(df, column, color='green'):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df[column], color=color)
    plt.title(f'Box Plot of {column}')
    plt.xlabel(column)
    plt.show()

def plot_correlation_matrix(df):
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()


def train_model(X_train, y_train, model, param_grid):
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1')
    grid_search.fit(X_train, y_train)
    return grid_search

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    return report

def main():
    # Load and preprocess the data
    df = load_data('alzheimers_disease_data.csv')
    df = preprocess_data(df)

    plot_histogram(df, 'Age')
    plot_boxplot(df, 'MMSE')
    plot_correlation_matrix(df)
    
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Define models and their parameter grids
    models = {
        'Decision Tree': (DecisionTreeClassifier(), {'max_depth': [3, 5, 7, 12, None]}),
        'Random Forest': (RandomForestClassifier(), {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7, 12, None]}),
        'K-Nearest Neighbors': (KNeighborsClassifier(), {'n_neighbors': [3, 5, 7]}),
        'Logistic Regression': (LogisticRegression(), {'C': [0.1, 1, 10]}),
        'Support Vector Machine': (SVC(), {'C': [0.1, 1, 10], 'gamma': [0.1, 1, 'scale', 'auto']})
    }
    
    # Train and evaluate each model
    for name, (model, param_grid) in models.items():
        trained_model = train_model(X_train, y_train, model, param_grid)
        report = evaluate_model(trained_model, X_test, y_test)
        print(f"{name} Classification Report:\n{report}\nBest Parameters: {trained_model.best_params_}\n")

if __name__ == "__main__":
    main()