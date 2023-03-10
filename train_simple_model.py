import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from deploy import register_model


def train():
    # Load the dataset from a CSV file
    data = pd.read_csv('Churn_Modelling_undersampled.csv', delimiter=',')
    data.dropna(inplace=True)
    # Select the features and target variable
    X = data[['Age', 'EstimatedSalary']]
    y = data['Exited']

    # Create a Random Forest classifier with 100 trees
    rf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Fit the classifier to the training data
    rf.fit(X, y)
    return rf


if __name__ == '__main__':
    trained_model = train()
    register_model(trained_model)