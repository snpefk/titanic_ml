from pandas import read_csv
from pymongo import MongoClient
from scipy.stats import shapiro
from sklearn import cross_validation
from sklearn.preprocessing import LabelEncoder

PASSENGER_DATA_LOCATION = "train.csv"


def write_in_db(data: dict):
    collection = mach_learn_table.collect
    collection.insert(data)


def fill_passengers(passengers):
    label_encoder = LabelEncoder()
    passengers.Age[passengers.Age.isnull()] = passengers.Age.mean()
    # заполняем пустые значения средней ценой билета
    passengers.Fare[passengers.Fare.isnull()] = passengers.Fare.median()
    max_pass_embarked = passengers.groupby('Embarked').count()['PassengerId']
    label_encoder.fit(passengers['Sex'])
    passengers.Embarked[passengers.Embarked.isnull()] = \
        max_pass_embarked[max_pass_embarked == max_pass_embarked.max()].index[0]
    passengers.Sex = label_encoder.transform(passengers.Sex)
    label_encoder.fit(passengers['Embarked'])
    passengers.Embarked = label_encoder.transform(passengers.Embarked)

    return passengers


def evaluate_meta_information(passengers) -> dict:
    default_passengers = {key: int(value.isnull().sum()) for key, value in
                          passengers.items()}

    dropped_fields = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    full_passenger = fill_passengers(passengers).drop(dropped_fields, axis=1)

    type_of_column = {}
    max_column = {}  # max of column
    min_column = {}  # min of column
    distribution = {}  # distribution of column

    for key, value in full_passenger.items():
        type_of_column[key] = str(value.dtype)
        max_column[key] = int(value.max())
        min_column[key] = int(value.min())
        distribution[key] = shapiro(value)

    return {
        'type_of_column': type_of_column,
        'max': max_column,
        'min': min_column,
        'dis': distribution,
        'empty': default_passengers
    }


def main():
    passengers = read_csv(PASSENGER_DATA_LOCATION, delimiter=",")
    write_in_db(evaluate_meta_information(passengers))

    target = passengers.Survived  # указываем показатель для исследования
    train = passengers.drop(['Survived'], axis=1)
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(
        train,
        target,
        test_size=0.25
    )

    write_in_db({
        'X_train': x_train.to_json(),
        'X_test': x_test.to_json(),
        'Y_train': y_train.to_json(),
        'Y_test': y_test.to_json()
    })


if __name__ == "__main__":
    db_connection = MongoClient()
    mach_learn_table = db_connection.mach_learn

    main()

    db_connection.close()
