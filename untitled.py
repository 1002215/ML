import pandas as pd
from sklern.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('train.csv', skiprows=2)
df.columns = ['PassengerId', 'Survival', 'Pclass', 'Name', 'Sex', 'Age', 
              'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']

df = df.drop(columns=['PassengerId','Ticket','Fare','Embarked'])

print(df.head())

X = df[df.columns[1:]]
y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)