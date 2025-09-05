import pandas as pd
from sklern.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('sheet.csv', skiprows=2)
df.columns = ['Label', 'Yellow', 'green', 'red', 'brown', 'orange', 
              'Length', 'Width', 'Height', 'Dull', 'Shiny', 'Rough', 'Smooth']

print(df.head())

X = df[df.columns[1:]]
y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = LogisticRegression(random_state=42, max_iter=1000)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2%}")

print("\nClassifiation Report:")
print(classification_report(y_test, y_pred))
