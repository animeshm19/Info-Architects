import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
pd.set_option('display.max_columns', None)

def load_data(filepath):
    """
    Load the dataset from a specified filepath.
    """
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df):
    """
    Preprocess the dataset by removing unnecessary columns, and applying normalization
    and standardization to the numerical features.
    """
    # Drop columns that are not required for the model training
    df.drop(['PatientID', 'DoctorInCharge'], axis=1, inplace=True)

    # Identify numerical columns
    numerical_columns = [col for col in df.columns if df[col].dtype in ['int64', 'float64'] and col != 'Diagnosis']

    # Apply MinMaxScaler and StandardScaler in sequence to normalize and standardize data
    min_max_scaler = MinMaxScaler()
    standard_scaler = StandardScaler()
    df[numerical_columns] = min_max_scaler.fit_transform(df[numerical_columns])
    df[numerical_columns] = standard_scaler.fit_transform(df[numerical_columns])

    return df

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
    mask = np.triu(np.ones_like(df.corr(), dtype=bool))

    # Plot heatmap of the correlation matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(),cmap="coolwarm", cbar_kws={"shrink": .5}, mask=mask)

    plt.show()

def split_data(df, target_column='Diagnosis'):
    """
    Split the dataset into training and testing sets.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column].astype('category')  # Ensure target is categorical
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, model, param_grid):
    """
    Train a model using GridSearchCV to find the best parameters.
    """
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1_micro', error_score='raise')
    grid_search.fit(X_train, y_train)
    return grid_search

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model using the test set.
    """
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    return report

def main():
    # Load and preprocess the data
    df = load_data('alzheimers_disease_data.csv')
    df = preprocess_data(df)

    print(df.head().T)
    print("---"*25)
    print(df.info())
    print("---"*25)
    print(df.describe().T)
    print("---"*25)
    print("Duplicated: ", sum(df.duplicated()))
    print("---"*25)
    
    plot_histogram(df, 'Age')
    plot_boxplot(df, 'MMSE')
    plot_correlation_matrix(df)
    
    X_train, X_test, y_train, y_test = split_data(df)

    # Defined models and their parameter grids
    models = {
        'Decision Tree': (DecisionTreeClassifier(), {'max_depth': [3, 5, 7, 12, None]}),
        'Random Forest': (RandomForestClassifier(), {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7, 12, None]}),
        'Gradient Boosting': (GradientBoostingClassifier(), {'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.1, 1]}),
        'AdaBoost': (AdaBoostClassifier(), {'n_estimators': [50, 100, 150]}),
    }

    # Train and evaluate each model
    for name, (model, param_grid) in models.items():
        try:
            trained_model = train_model(X_train, y_train, model, param_grid)
            report = evaluate_model(trained_model, X_test, y_test)
            print(f"{name} Classification Report:\n{report}\nBest Parameters: {trained_model.best_params_}\n")
            print("---"*20)
        except Exception as e:
            print(f"Error training {name}: {str(e)}")

if __name__ == "__main__":
    main()
