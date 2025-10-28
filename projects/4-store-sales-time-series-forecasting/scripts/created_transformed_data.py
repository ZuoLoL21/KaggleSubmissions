import os
import pandas as pd
import numpy as np
import shutil

from common.libs.PandasHelper import one_hot_encode

RAW_DATA_DIRECTORY = os.path.normpath(os.path.join(os.getcwd(), '..', 'inputs', 'raw'))
TRANSFORMED_DATA_DIRECTORY = os.path.normpath(os.path.join(os.getcwd(), '..', 'inputs', 'transformed'))

OIL_FILE = 'oil.csv'
HOLIDAY_FILE = 'holidays_events.csv'
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
STORES_FILE = "stores.csv"

OTHER_RAWS = ["transactions.csv"]


def treat_oil_data():
    oil_df = pd.read_csv(os.path.join(RAW_DATA_DIRECTORY, OIL_FILE), delimiter=",")

    oil_df['date'] = pd.to_datetime(oil_df['date'])
    oil_df_t = oil_df.set_index('date').asfreq('D')

    data = oil_df_t["dcoilwtico"].to_numpy()

    for i, n in enumerate(data):
        if not np.isnan(n):
            continue
        data[i] = np.nanmean(data[max(0, i - 5): min(i + 5, data.shape[0])])

    oil_df_t["dcoilwtico"] = data

    oil_df_t.to_csv(os.path.join(TRANSFORMED_DATA_DIRECTORY, OIL_FILE))


def treat_holiday_data():
    holiday_events_df = pd.read_csv(os.path.join(RAW_DATA_DIRECTORY, HOLIDAY_FILE), delimiter=",")

    holiday_events_df_t = holiday_events_df.loc[holiday_events_df["type"] != "Transfer"].copy()
    holiday_events_df_t.drop(["transferred", "description"], axis=1, inplace=True)

    holiday_events_df_t.to_csv(os.path.join(TRANSFORMED_DATA_DIRECTORY, HOLIDAY_FILE))


def treat_stores_data():
    stores_df = pd.read_csv(os.path.join(RAW_DATA_DIRECTORY, STORES_FILE), delimiter=",")
    stores_df.set_index('store_nbr', inplace=True)

    stores_df = one_hot_encode(stores_df, ["type"])

    stores_df.to_csv(os.path.join(TRANSFORMED_DATA_DIRECTORY, STORES_FILE))


COLUMNS_TO_KEEP = ['date', 'store_nbr']
COLUMN_TO_MODIFY = 'family'
COLUMN_TO_TEST = 'sales'


def treat_train_data():
    train_df = pd.read_csv(os.path.join(RAW_DATA_DIRECTORY, TRAIN_FILE), delimiter=",")
    df_pivot = train_df.pivot(index=COLUMNS_TO_KEEP, columns=COLUMN_TO_MODIFY, values=COLUMN_TO_TEST)

    df_pivot.to_csv(os.path.join(TRANSFORMED_DATA_DIRECTORY, TRAIN_FILE))


def treat_test_data():
    test_df = pd.read_csv(os.path.join(RAW_DATA_DIRECTORY, TEST_FILE), delimiter=",")
    unique_pairs = test_df[COLUMNS_TO_KEEP].drop_duplicates().reset_index(drop=True)
    unique_pairs.set_index('date', inplace=True)

    unique_pairs.to_csv(os.path.join(TRANSFORMED_DATA_DIRECTORY, TEST_FILE))


def copy_file(source, destination):
    shutil.copy(source, destination)


def main():
    treat_oil_data()
    treat_holiday_data()
    treat_train_data()
    treat_test_data()
    treat_stores_data()

    for file_name in OTHER_RAWS:
        copy_file(
                os.path.join(RAW_DATA_DIRECTORY, file_name),
                os.path.join(TRANSFORMED_DATA_DIRECTORY, file_name))


if __name__ == '__main__':
    main()
