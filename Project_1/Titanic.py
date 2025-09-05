# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

for dirname, _, filenames in os.walk("./"):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
"""In this porject, you must predict what features contributed the most the passenger's
survival on the Titanic. You must graoh the distribution of attributes and select the attribute
that was the most critical to survival. The program must be written in Python and use Kaggle's
Titanic - Machine Learning from Disaster datasets."""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import numpy as np

""" I received this data from the Kaggle Titanic - Machine Learning from Disaster training
data set called titanic.csv."""
df = pd.read_csv('data/titanic.csv', skiprows=2)
df.columns = ['PassengerId', 'Survival', 'Pclass', 'Name', 'Sex', 'Age', 
              'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']

"""I believe some of the data columns are not helpful to determining what contributed to their
survival, and as such can be eleimnated to improve run time and performance. My bias is that I
believe women and children should have been prioritized at the time of the sinking."""

# The PassengerId, ticket, fare, and where they embarked from have nothing to do with their
# survival. They are nonessential and as such can be removed to increase performance and runtime.
df = df.drop(columns=['PassengerId','Name','Ticket','Fare','Cabin','Embarked'])

print(df.head())


for i in df.columns[1:]:
    print(i)
    X = df[i]
    survivors = df['Survival']
    
    plt.hist(X,bins=30,edgecolor='black')
    plt.xlabel(X)
    plt.ylabel("Frequency")
    plt.show()