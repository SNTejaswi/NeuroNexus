import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data_path = 'C:\Users\tejas\Downloads\titanic.zip'  # Change this if needed
df = pd.read_csv(data_path)

# Display first few rows
print(df.head())

# Preprocessing
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]  # selected features
df['Age'].fillna(df['Age'].median(), inplace=True)  # fill missing age

# Encode 'Sex' column
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])

# Split features and labels
X = df.drop('Survived', axis=1)
y = df['Survived']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Prediction and evaluation
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
