from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
import pandas as pd
import numpy as np

file_path = 'D:/midterm_2/'
file_name = 'car_data.csv'

df = pd.read_csv(file_path + file_name)

le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

X = df.drop('Purchased', axis=1)
y = df['Purchased']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

