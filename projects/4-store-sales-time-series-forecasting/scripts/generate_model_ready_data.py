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


def _create_dict_for_holiday():
    holiday_dict = {}

    for index, row in HOLIDAY_DATA.iterrows():
        obj_to_add = {
            "type": row['type'],
            "locale": row['locale'],
            "locale_name": row['locale_name'],
        }
        if holiday_dict.get(row["date"]) is None:
            holiday_dict[row["date"]] = [obj_to_add]
        else:
            holiday_dict[row["date"]].append(obj_to_add)
    return holiday_dict


def _is_holiday(date, city, state, holiday_dict=None):
    returns = []

    def _process_entry(entry):
        if entry["locale"] == "National":
            returns.append(entry["type"])
        elif entry["locale"] == "Regional":
            if entry["locale_name"] == state:
                returns.append(entry["type"])
            return
        elif entry["locale"] == "Local":
            if entry["locale_name"] == city:
                returns.append(entry["type"])
            return
        else:
            raise RuntimeError("Locale not handled")

    if holiday_dict is None:
        entries = HOLIDAY_DATA.loc[HOLIDAY_DATA["date"] == date]

        for index, entry in entries.iterrows():
            _process_entry(entry)

        return returns

    else:
        output = holiday_dict.get(date)
        if output is None:
            return returns

        for entry in output:
            _process_entry(entry)

        return returns


def _left_join(df1, df2, axis_name: Iterable[str] = 'index'):
    return df1.merge(df2, on=axis_name, how='left')


def _left_join_insert_left(df1, df2, axis_name: Iterable[str] = 'index'):
    return df2.merge(df1, on=axis_name, how='right')


def add_oil_data(train_df, oil_df):
    return _left_join(train_df, oil_df, axis_name='date')


def add_store_data(train_df, store_df):
    return _left_join(train_df, store_df, axis_name='store_nbr')


def add_transaction_data(train_df, transaction_df):
    return _left_join(train_df, transaction_df, axis_name=('date', 'store_nbr'))


def add_holiday_information_split(input_filename, holiday_dict=None, preface="default"):
    if not preface:
        raise RuntimeError("Preface must be specified")

    write_files = [
        open(os.path.join(TRANSFORMED_2_DATA_DIRECTORY, preface, f"{i}.csv"), 'w', newline='')
        for i in range(UNIQUE_STORE_NUMBERS)]

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

        for line in tqdm(reader):
            to_add = transform(_is_holiday(line[date_index], line[city_index], line[state_index], holiday_dict))
            line.extend(to_add)

            writers[int(line[store_index]) - 1].writerow(line)

    for file in write_files:
        file.close()


def write_to_file(df, name):
    print("\tSaving data to file", name)
    df.to_csv(os.path.join(TRANSFORMED_DATA_DIRECTORY, name))


FILES_TO_PROCESS = ["train.csv", "test.csv"]


def main():
    for f in FILES_TO_PROCESS:
        train_df = pd.read_csv(os.path.join(TRANSFORMED_DATA_DIRECTORY, f), delimiter=",")
        print("Starting process ", train_df.shape)

        oil_df = pd.read_csv(os.path.join(TRANSFORMED_DATA_DIRECTORY, "oil.csv"), delimiter=",")
        train_df = add_oil_data(train_df, oil_df)
        print("Added oil data ", train_df.shape)

        stores_df = pd.read_csv(os.path.join(TRANSFORMED_DATA_DIRECTORY, "stores.csv"),
                                delimiter=",").set_index("store_nbr")
        train_df = add_store_data(train_df, stores_df)
        train_df.set_index("date", inplace=True)
        print("Added store data ", train_df.shape)

        # transactions_df = pd.read_csv(os.path.join(TRANSFORMED_DATA_DIRECTORY, "transactions.csv"), delimiter=",")
        # train_df = add_transaction_data(train_df, transactions_df)
        # print("Added transaction data ", train_df.shape)

        write_to_file(train_df, f"{f}_added.csv")

        holiday = _create_dict_for_holiday()
        add_holiday_information_split(f"{f}_added.csv", holiday_dict=holiday, preface=f.split(".")[0])


if __name__ == "__main__":
    main()
