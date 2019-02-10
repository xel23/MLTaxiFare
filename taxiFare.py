import pandas as pd
import datetime
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.cluster import KMeans

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def main():
    train = pd.read_csv('train.csv', nrows=15_000_000)
    print("read")
    train.head(3)
    pd.set_option('float_format', '{:f}'.format)  # float format

    train = train[train.fare_amount > 0]

    train['longitude_diff'] = train['dropoff_longitude'] - train['pickup_longitude']
    train['latitude_diff'] = train['dropoff_latitude'] - train['pickup_latitude']
    train = train.loc[train.passenger_count > 0]
    train_df = train.drop('fare_amount', axis=1)
    test = pd.read_csv('test.csv')
    test['longitude_diff'] = test['dropoff_longitude'] - test['pickup_longitude']
    test['latitude_diff'] = test['dropoff_latitude'] - test['pickup_latitude']
    test_df = test
    train_df['is_train'] = 1
    test_df['is_train'] = 0
    train_test = pd.concat([train_df, test_df], axis=0)

    # work with date
    train_test['year'] = train_test.pickup_datetime.apply(lambda x: x[:4])
    train_test['month'] = train_test.pickup_datetime.apply(lambda x: x[5:7])
    train_test['hour'] = train_test.pickup_datetime.apply(lambda x: x[11:13])
    train_test['pickup_datetime'] = train_test.pickup_datetime.apply(
        lambda x: datetime.datetime.strptime(x[:10], '%Y-%m-%d'))
    train_test['day_of_week'] = train_test.pickup_datetime.apply(lambda x: x.weekday())
    train_test['pickup_date'] = train_test.pickup_datetime.apply(lambda x: x.date())
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start='2009-01-01', end='2017-12-31').to_pydatetime()
    train_test['holidat_or_not'] = train_test.pickup_datetime.apply(lambda x: 1 if x in holidays else 0)
    train_test = train_test.drop(['key', 'pickup_datetime', 'pickup_date'], axis=1)
    train_test['year'] = train_test['year'].astype('int')
    train_test['hour'] = train_test['hour'].astype('int')
    # -------------------
    clusters = pd.get_dummies(train_test['day_of_week'], prefix='day_week', drop_first=False)
    train_test = pd.concat([train_test, clusters], axis=1).drop('day_of_week', axis=1)

    clusters = pd.get_dummies(train_test['month'], prefix='month_', drop_first=False)
    train_test = pd.concat([train_test, clusters], axis=1).drop('month', axis=1)

    clusters = pd.get_dummies(train_test['hour'], prefix='hour_', drop_first=False)
    train_test = pd.concat([train_test, clusters], axis=1).drop('hour', axis=1)

    y = train[['fare_amount']]
    train = train_test[train_test.is_train == 1].drop(['is_train'], axis=1)
    train.info()
    X = train
    test = train_test[train_test.is_train == 0].drop(['is_train'], axis=1)

    model = RandomForestRegressor(n_estimators=25, max_features=0.3, max_depth=30, min_samples_leaf=2, random_state=1,
                                  n_jobs=-1)
    model.fit(X, y)
    prediction = model.predict(test)
    submission = pd.read_csv('sample_submission.csv')
    submission['fare_amount'] = prediction
    submission.to_csv('submission1.csv', index=False)

if __name__ == "__main__":
    main()