import os
import pandas as pd
import numpy as np
import shutil

RAW_DATA_DIRECTORY = os.path.normpath(os.path.join(os.getcwd(), '..', 'inputs', 'raw'))
TRANSFORMED_DATA_DIRECTORY = os.path.normpath(os.path.join(os.getcwd(), '..', 'inputs', 'transformed'))

OIL_FILE_NAME = 'oil.csv'
HOLIDAY_FILE_NAME = 'holidays_events.csv'

OTHER_RAWS = ["stores.csv", "transactions.csv", "train.csv", "test.csv"]


def treat_oil_data():
    oil_df = pd.read_csv(os.path.join(RAW_DATA_DIRECTORY, OIL_FILE_NAME), delimiter=",")

    oil_df['date'] = pd.to_datetime(oil_df['date'])
    oil_df_t = oil_df.set_index('date').asfreq('D')

    data = oil_df_t["dcoilwtico"].to_numpy()

    for i, n in enumerate(data):
        if not np.isnan(n):
            continue
        data[i] = np.nanmean(data[max(0, i - 5): min(i + 5, data.shape[0])])

    oil_df_t["dcoilwtico"] = data

    oil_df_t.to_csv(os.path.join(TRANSFORMED_DATA_DIRECTORY, OIL_FILE_NAME))


def treat_holiday_data():
    holiday_events_df = pd.read_csv(os.path.join(RAW_DATA_DIRECTORY, HOLIDAY_FILE_NAME), delimiter=",")

    holiday_events_df_t = holiday_events_df.loc[holiday_events_df["type"] != "Transfer"].copy()
    holiday_events_df_t.drop(["transferred", "description"], axis=1, inplace=True)

    holiday_events_df_t.to_csv(os.path.join(TRANSFORMED_DATA_DIRECTORY, HOLIDAY_FILE_NAME))


def copy_file(source, destination):
    shutil.copy(source, destination)


def main():
    treat_oil_data()
    treat_holiday_data()

    for file_name in OTHER_RAWS:
        copy_file(
                os.path.join(RAW_DATA_DIRECTORY, file_name),
                os.path.join(TRANSFORMED_DATA_DIRECTORY, file_name))


if __name__ == '__main__':
    main()
