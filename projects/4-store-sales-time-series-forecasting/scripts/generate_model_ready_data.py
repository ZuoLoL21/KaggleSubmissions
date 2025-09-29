import csv
import os
from typing import Iterable

import pandas as pd
import numpy as np
from tqdm import tqdm

TRANSFORMED_2_DATA_DIRECTORY = os.path.normpath(os.path.join(os.getcwd(), '..', 'inputs', 'model_ready'))
TRANSFORMED_DATA_DIRECTORY = os.path.normpath(os.path.join(os.getcwd(), '..', 'inputs', 'transformed'))

HOLIDAY_DATA = pd.read_csv(os.path.join(TRANSFORMED_DATA_DIRECTORY, 'holidays_events.csv'))

UNIQUE_STORE_NUMBERS = 54


def _is_holiday(date, city, state):
    entries = HOLIDAY_DATA.loc[HOLIDAY_DATA["date"] == date]

    returns = []
    for index, entry in entries.iterrows():
        if entry["locale"] == "National":
            returns.append(entry["type"])
        elif entry["locale"] == "Regional":
            if entry["locale_name"] == state:
                returns.append(entry["type"])
            continue
        elif entry["locale"] == "Local":
            if entry["locale_name"] == city:
                returns.append(entry["type"])
            continue
        else:
            raise RuntimeError("Locale not handled")

    return returns


def _left_join(df1, df2, axis_name: Iterable[str] = 'index'):
    return df1.merge(df2, on=axis_name, how='left')


def add_oil_data(train_df, oil_df):
    return _left_join(train_df, oil_df, axis_name='date')


def add_store_data(train_df, store_df):
    return _left_join(train_df, store_df, axis_name='store_nbr')


def add_transaction_data(train_df, transaction_df):
    return _left_join(train_df, transaction_df, axis_name=('date', 'store_nbr'))


def add_holiday_information(input_filename):
    write_files = [open(os.path.join(TRANSFORMED_2_DATA_DIRECTORY, f"{i}.csv"), 'w', newline='') for i in
                   range(UNIQUE_STORE_NUMBERS)]
    with (
        open(os.path.join(TRANSFORMED_DATA_DIRECTORY, input_filename), 'r', newline='') as f,
    ):
        reader = csv.reader(f)
        writers = [csv.writer(file) for file in write_files]

        header_line = next(reader)

        date_index = header_line.index("date")
        city_index = header_line.index("city")
        state_index = header_line.index("state")

        store_index = header_line.index("store_nbr")

        unique_vacation_types = HOLIDAY_DATA["type"].unique()
        unique_vacation_types_h = ["vacation " + vac_type.lower() for vac_type in unique_vacation_types]

        def transform(array_of_vac_types):
            to_ret = ["0"] * len(unique_vacation_types)
            for vac_type in array_of_vac_types:
                index = np.where(unique_vacation_types == vac_type)
                if len(index) == 0 or len(index[0]) == 0:
                    raise RuntimeError("No vacation type found")
                to_ret[index[0][0]] = "1"
            return to_ret

        header_line.extend(unique_vacation_types_h)

        for writer in writers:
            writer.writerow(header_line)

        print(len(writers))
        for line in tqdm(reader):
            to_add = transform(_is_holiday(line[date_index], line[city_index], line[state_index]))
            line.extend(to_add)

            writers[int(line[store_index]) - 1].writerow(line)

    for file in write_files:
        file.close()


def write_to_file(df, name):
    print("\tSaving data to file", name)
    df.to_csv(os.path.join(TRANSFORMED_DATA_DIRECTORY, name))


def main():
    # oil_df = pd.read_csv(os.path.join(TRANSFORMED_DATA_DIRECTORY, "oil.csv"), delimiter=",")
    # stores_df = pd.read_csv(os.path.join(TRANSFORMED_DATA_DIRECTORY, "stores.csv"),
    #                         delimiter=",").set_index("store_nbr")
    # transactions_df = pd.read_csv(os.path.join(TRANSFORMED_DATA_DIRECTORY, "transactions.csv"), delimiter=",")
    # train_df = pd.read_csv(os.path.join(TRANSFORMED_DATA_DIRECTORY, "train.csv"), delimiter=",").set_index("id")
    # print("Starting process ", train_df.shape)
    #
    # train_df = add_oil_data(train_df, oil_df)
    # print("Added oil data ", train_df.shape)
    #
    # train_df = add_store_data(train_df, stores_df)
    # print("Added store data ", train_df.shape)
    #
    # train_df = add_transaction_data(train_df, transactions_df)
    # print("Added transaction data ", train_df.shape)
    # write_to_file(train_df, "train_df_added.csv")

    add_holiday_information("train_df_added.csv")


if __name__ == "__main__":
    main()
