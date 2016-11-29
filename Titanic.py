import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv(r"C:\Users\user\Desktop\python projects\Titanic\train.csv")
test = pd.read_csv(r"C:\Users\user\Desktop\python projects\Titanic\test.csv")
train = pd.DataFrame(train)
test = pd.DataFrame(test)
df = pd.concat([train, test], ignore_index=True)

##Extract title
title = df.ix[:, 'Name'].str.extract(', ([A-Z][a-z]+).')

#Group titles)
for i in range(len(title)):
    if title[i] == 'Mme':
        title[i] = "Mrs"
    if title[i] == 'Mlle':
        title[i] = "Miss"
    if title[i] == 'Jonkheer':
        title[i] = "Master"
    if title[i] == "Miss":
        title[i] = "Ms"
    if (title[i] != 'Mrs') & (title[i] !='Mr') & (title[i] !='Ms') & (title[i] !='Master'):
        title[i] = "Special"

df['Title'] = title
df['Title_int']  = df['Title'].map( {'Mrs': 0, 'Ms': 1, 'Mr': 2, 'Master': 3, 'Special': 4} ).astype(int)

#Gender
df['Sex_int'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

#Missing Embarked
# print(df[(df.Cabin.str.contains("C128")) & (df.Pclass == 1)][['Embarked', 'Ticket', 'Fare', 'Cabin']])
df.ix[df['Embarked'].isnull() ==True, 'Embarked'] = 'C'
df['Embarked_int'] = df['Embarked'].map( {'C': 0, 'Q': 1, 'S': 2} ).astype(int)

#Missing Fare
# print(df[df['Fare'].isnull() ==True]['Pclass'])
mean_fare = df[df.Pclass == 3]['Fare'].mean()
df.ix[df['Fare'].isnull() ==True, 'Fare'] = mean_fare


#Missing Ages
median_ages = np.zeros((5,3))

for i in range(5):
    for j in range(3):
        median_ages[i,j] = df[(df['Title_int']==i) & \
                                 (df['Pclass']==j+1)]['Age'].dropna().median()


df['Age_fill'] = df.Age

for i in range(5):
     for j in range(3):
         df.ix[(df.Age_fill.isnull() == True) & \
                        (df.Title_int == i) & \
                        (df.Pclass == j+1), 'Age_fill'] = median_ages[i, j]

df['AgeIsNull'] = pd.isnull(df.Age).astype(int)

#Family size
df['FamilySize'] = (df.Parch + df.SibSp).astype(int)

#drop non-numeric columns
df = df.drop(['Age', 'PassengerId', 'Name','Sex', 'Ticket', 'Cabin', 'Embarked', 'Title'], axis = 1)

#split train and test sets
train = df[0:891]
S = train.Survived
train = train.drop(['Survived'], axis = 1)
train.insert(0, "Survived", S)

train_data = train.values
test = df[891:1310]
test = test.drop(['Survived'], axis = 1)
test_data = test.values

#Random Forests
forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit(train_data[:,1::],train_data[:,0])

output = forest.predict(test_data).astype(int)

test['Survived'] = output

test_ids = list(range(892,1310))
predictions = pd.DataFrame({"PassengerId": test_ids, "Survived": output})

predictions.to_csv("submission1.csv", index = False)
