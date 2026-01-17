import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, PrecisionRecallDisplay, precision_recall_curve, average_precision_score
def exploratory_data_analysis(df):
    # Looking at first five rows
    print(df.head())

    # Checking the size of the dataset
    print(f"Shape of Dataset: {df.shape}")

    # Checking for missing values
    print("Max no. of Null Values:", df.isnull().sum().max())

    # Checking the number of positive and negative samples
    print("\n", df['Class'].value_counts())


def visualize(df):
    # THIS VISUALIZES THE SIZE OF THE DATASET
    plt.figure(figsize=(8, 6))

    # Plot the counts
    sns.countplot(x='Class', data=df, palette='viridis')

    # Add labels and title
    plt.title('Distribution of Transactions (0: Legitimate, 1: Fraud)')
    plt.xlabel('Class')
    plt.ylabel('Number of Transactions')

    # Show the plot
    plt.show()

    # THIS VISUALIZES THE AMOUNT WITHDRAWN
    # Create two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

    # Plot for Fraudulent transactions
    ax1.hist(df.Amount[df.Class == 1], bins=30, color='red', alpha=0.7)
    ax1.set_title('Fraudulent Transactions')

    # Plot for Legitimate transactions
    ax2.hist(df.Amount[df.Class == 0], bins=30, color='blue', alpha=0.7)
    ax2.set_title('Legitimate Transactions')

    plt.xlabel('Amount ($)')
    plt.ylabel('Number of Transactions')
    plt.yscale('log')  # Using log scale because legitimate transactions are so numerous
    plt.show()

    plt.figure(figsize=(12, 10))
    # Calculate correlation
    corr = df.corr()

    # THIS SHOWS A CORELATION HEATMAP
    # Plot heatmap
    sns.heatmap(corr, cmap='coolwarm_r', annot=False)
    plt.title('Correlation Matrix Heatmap')
    plt.show()

def feature_scaling(df):
    # Create a Scaler Object (This shifts the data so the mean is 0 and standard deviation is 1)
    scaler = StandardScaler()

    # Scale 'Amount' and 'Time' (reshape is needed because scaler expects a 2D array)
    df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))

    # Now we drop the original 'Time' and 'Amount' columns as they are redundant
    df.drop(['Time', 'Amount'], axis=1, inplace=True)


def main():
    df = pd.read_csv("creditcard.csv")

    # exploratory_data_analysis(df)
    # visualize(df)

    feature_scaling(df)

    # X = all columns except 'Class'
    X = df.drop('Class', axis=1)

    # y = only the 'Class' column
    y = df['Class']

    # Stratify=y is CRITICAL here. It ensures the 0.17% fraud is
    # distributed evenly between training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17, stratify=y)

    # Initialize the Decision Tree model
    model_1 = DecisionTreeClassifier(max_depth=5, random_state=17)
    # Train the model
    model_1.fit(X_train, y_train)
    # Make predictions on the test set
    y_hat_decision_tree = model_1.predict(X_test)

    print("--- Decision Tree Results [Confusion Matrix] ---")
    print(confusion_matrix(y_test, y_hat_decision_tree))
    print(classification_report(y_test, y_hat_decision_tree))

    # 1. Get the probability scores (not just 0 or 1)
    # Models give a probability for each class; we want the probability of being '1' (fraud)
    y_scores_decision_tree = model_1.predict_proba(X_test)[:, 1]

    # 2. Calculate the Average Precision (the area under the curve)
    average_precision = average_precision_score(y_test, y_scores_decision_tree)

    # 3. Create the plot
    display = PrecisionRecallDisplay.from_predictions(y_test, y_scores_decision_tree, name="Decision Tree")
    _ = display.ax_.set_title(f"Precision-Recall Curve (AP={average_precision:.2f})")

    plt.show()

if __name__ == '__main__':
    main()