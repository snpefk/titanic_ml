import numpy as np
from pandas import read_csv, DataFrame, Series
import matplotlib.pyplot as plt
import pymongo
from pymongo import MongoClient
from sklearn import cross_validation
from sklearn.preprocessing import LabelEncoder
from scipy.stats import shapiro
import json


def write_in_db(json_data):
    client = MongoClient()
    connection = MongoClient()
    db = connection.mach_learn
    collection = db.collect
    collection.insert(json_data)
    connection.close()


def fill_data(data):
    data.Age[data.Age.isnull()] = data.Age.mean()
    data.Fare[data.Fare.isnull()] = data.Fare.median()  # заполняем пустые значения средней ценой билета
    MaxPassEmbarked = data.groupby('Embarked').count()['PassengerId']
    label.fit(data['Sex'])
    data.Embarked[data.Embarked.isnull()] = MaxPassEmbarked[MaxPassEmbarked == MaxPassEmbarked.max()].index[0]
    data.Sex = label.transform(data.Sex)
    label.fit(data['Embarked'])
    data.Embarked = label.transform(data.Embarked)
    return data


def information_about_data(input):
    type_of_column = {}
    max = {}  # max of column
    min = {}  # min of column
    dis = {}  # distribution of column
    for x in input:
        type_of_column[x] = str(input[x].dtype)
        max[x] = int(input[x].max())
        min[x] = int(input[x].min())
        dis[x] = shapiro(input[x])
    return json.dumps({'type_of_column': type_of_column, 'max': max, 'min': min, 'dis': dis})


if __name__ == "__main__":
    empty = {}
    label = LabelEncoder()
    path = "train.csv"
    data = read_csv(path, delimiter=",")

    for x in data:
        empty[x] = int(data[x].isnull().sum())
    data = fill_data(data)
    data = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    json_data = json.loads(information_about_data(data))
    json_data['empty'] = empty
    write_in_db(json_data)
    target =  data.Survived     #указываем показатель для исследования
    train = data.drop(['Survived'], axis=1)
    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(train, target, test_size=0.25)
    print(Y_train.to_json())
    write_in_db({'X_train': X_train.to_json(), 'X_test': X_test.to_json(),'Y_train': Y_train.to_json(), 'Y_test': Y_test.to_json() })
