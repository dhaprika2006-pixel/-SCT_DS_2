import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Show first 5 rows
print(train.head())
print("\nINFO:\n")
print(train.info())

print("\nMISSING VALUES:\n")
print(train.isnull().sum())

print("\nSTATISTICS:\n")
print(train.describe())
# Fill missing Age
train['Age'].fillna(train['Age'].median(), inplace=True)

# Fill Embarked
train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)

# Drop Cabin
train.drop('Cabin', axis=1, inplace=True)
# Convert Sex
train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})

# Convert Embarked
train = pd.get_dummies(train, columns=['Embarked'], drop_first=True)
# Create Family Size
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1

# Create IsAlone
train['IsAlone'] = 1
train.loc[train['FamilySize'] > 1, 'IsAlone'] = 0
plt.tight_layout()
sns.countplot(x='Survived', data=train)
plt.title("Survival Count (0 = No, 1 = Yes)")
plt.show()
sns.countplot(x='Sex', hue='Survived', data=train)
plt.title("Gender vs Survival")
plt.show()
sns.countplot(x='Pclass', hue='Survived', data=train)
plt.title("Passenger Class vs Survival")
plt.show()
sns.histplot(train['Age'], bins=30, kde=True)
plt.title("Age Distribution")
plt.show()
sns.boxplot(x='Survived', y='Fare', data=train)
plt.title("Fare vs Survival")
plt.show()
plt.figure(figsize=(10,6))
sns.heatmap(train.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()
X = train.drop(['Survived','Name','Ticket','PassengerId'], axis=1)
