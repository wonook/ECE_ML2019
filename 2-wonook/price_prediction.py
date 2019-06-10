from pathlib import Path

import math
import click
import pandas as pd
import numpy as np

current_directory = Path(__file__).absolute().parent
default_data_directory = current_directory.joinpath('..', 'trivagoRecSysChallengeData2019_v2')

GR_COLS = ["user_id", "session_id", "timestamp", "step"]


def string_to_array(s):
    """Convert pipe separated string to array."""

    if isinstance(s, str):
        out = s.split("|")
        if out[0].isdigit():
            out = [int(x) for x in out]
    elif math.isnan(s):
        out = []
    else:
        raise ValueError("Value must be either string of nan")
    return out


# ##################


item_meta_list = []

def save_item_meta_array(arr):
    for i in arr:
        if i not in item_meta_list:
            item_meta_list.append(i)


def array_to_encoding(arr):
    encoding = np.zeros(len(item_meta_list), dtype=int)
    for i in arr:
        encoding[item_meta_list.index(i)] += 1
    return "|".join([str(x) for x in encoding])


def encode_items(df_in):
    col_expl = "properties"
    # print("DF_IN:\n", df_in)
    df = df_in.copy()
    df.loc[:, col_expl] = df[col_expl].apply(string_to_array)

    # print("DF:\n", df)
    df[col_expl].apply(save_item_meta_array)
    item_meta_list.sort()

    print("ITEM_META_LIST_LEN:", len(item_meta_list))
    print("ITEM_META:", item_meta_list)

    df = df_in.copy()
    df.loc[:, col_expl] = df[col_expl].apply(array_to_encoding)
    # print("ENCODED:\n", df)
    return df


def retract_price_info(df_in):
    price_dict = {}
    for i in df[['impressions', 'prices']].itertuples(index=False):
        price_dict.update(dict(zip(string_to_array(i[0]), string_to_array(i[1]))))

    return price_dict


# ####################################


def explode(df_in, col_expl):
    """Explode column col_expl of array type into multiple rows."""

    df = df_in.copy()
    df.loc[:, col_expl] = df[col_expl].apply(string_to_array)

    df_out = pd.DataFrame(
        {col: np.repeat(df[col].values,
                        df[col_expl].str.len())
         for col in df.columns.drop(col_expl)}
    )

    df_out.loc[:, col_expl] = np.concatenate(df[col_expl].values)
    df_out.loc[:, col_expl] = df_out[col_expl].apply(int)

    return df_out


def group_concat(df, gr_cols, col_concat):
    """Concatenate multiple rows into one."""

    df_out = (
        df
        .groupby(gr_cols)[col_concat]
        .apply(lambda x: ' '.join(x))
        .to_frame()
        .reset_index()
    )

    return df_out


def calc_recommendation(df_expl, df_pop):
    """Calculate recommendations based on popularity of items.
    The final data frame will have an impression list sorted according to the number of clicks per item in a reference data frame.
    :param df_expl: Data frame with exploded impression list
    :param df_pop: Data frame with items and number of clicks
    :return: Data frame with sorted impression list according to popularity in df_pop
    """

    df_expl_clicks = (
        df_expl[GR_COLS + ["impressions"]]
        .merge(df_pop,
               left_on="impressions",
               right_on="reference",
               how="left")
    )

    df_out = (
        df_expl_clicks
        .assign(impressions=lambda x: x["impressions"].apply(str))
        .sort_values(GR_COLS + ["n_clicks"],
                     ascending=[True, True, True, True, False])
    )

    df_out = group_concat(df_out, GR_COLS, "impressions")
    df_out.rename(columns={'impressions': 'item_recommendations'}, inplace=True)

    return df_out


@click.command()
@click.option('--data-path', default=None, help='Directory for the CSV files')
def main(data_path):

    # calculate path to files
    data_directory = Path(data_path) if data_path else default_data_directory
    train_csv = data_directory.joinpath('train.csv')
    item_price_info_csv = data_directory.joinpath('item_price_info.csv')
    item_csv = data_directory.joinpath('item_metadata.csv')
    encoded_item_csv = data_directory.joinpath('item_metadata_encoded.csv')


    df_item_price_info = None
    df_item_encoded = None
    if not encoded_train_csv.is_file():
        print(f"Reading {train_csv} ...")
        df_train = pd.read_csv(train_csv)
        # df_train_masked = df_train[df_train['session_id'].isin(df_target['session_id'])]
        print("TRAIN:\n", df_train)
        df_item_price_info = retract_price_info(df_train)
        df_item_price_info.to_csv(item_price_info_csv, index=False)
    else:
        df_item_price_info = pd.read_csv(item_price_info_csv)
    print("DF_TRAIN:\n", df_train_encoded)

    if not encoded_item_csv.is_file():
        print(f"Reading {item_csv} ...")
        df_item = pd.read_csv(item_csv)
        df_item_masked = df_item[df_item['item_id'].isin(df_train_encoded['item_id'])]
        print("MASKED_ITEM:\n", df_item_masked)
        df_item_encoded = encode_items(df_item_masked)
        df_item_encoded.to_csv(encoded_item_csv, index=False)
    else:
        df_item_encoded = pd.read_csv(encoded_item_csv)
    print("DF_ITEM:\n", df_item_encoded)

    item_dict = dict(zip(df_item_encoded.item_id, df_item_encoded.properties))
    # print("ITEM_DICT:\n", item_dict)

    item_vec_len = len(item_meta_list)



    print("Finished calculating recommendations.")


if __name__ == '__main__':
    main()