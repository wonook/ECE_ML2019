from pathlib import Path

import math
import click
import pandas as pd
import numpy as np
import xgboost as xgb

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

def string_to_dmatrix_array(s):
    if isinstance(s, str):
        st = ""
        out = s.split("|")
        for x, y in list(zip(range(len(out)), out)):
            st = st + "{}:{} ".format(x, y)
        return st
    else:
        raise ValueError("Value must be either string of nan")

# ##################

def write_rows_to_file(filename, rows):
  f = open(filename, 'w')
  for row in rows:
    f.write(row + "\n")
  f.close()


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
    df.loc[:, col_expl] = df[col_expl].apply(string_to_array).apply(array_to_encoding)
    # print("ENCODED:\n", df)
    return df


def retract_price_info(df_in):
    price_dict = {}
    for i in df_in[['impressions', 'prices']].itertuples(index=False):
        price_dict.update(dict(zip(string_to_array(i[0]), string_to_array(i[1]))))

    return pd.DataFrame([(k, v) for k, v in price_dict.items()], columns = ['item_id', 'price'])


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

def merge_items_left(df1, df2):
    df_out = df1.merge(df2, left_on='item_id', right_on='item_id', how='left')
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
    if not item_price_info_csv.is_file():
        print(f"Reading {train_csv} ...")
        df_train = pd.read_csv(train_csv)
        # df_train_masked = df_train[df_train['session_id'].isin(df_target['session_id'])]
        print("TRAIN:\n", df_train)
        df_item_price_info = retract_price_info(df_train)
        df_item_price_info.to_csv(item_price_info_csv, index=False)
    else:
        df_item_price_info = pd.read_csv(item_price_info_csv)
    print("DF_PRICE_INFO:\n", df_item_price_info)

    if not encoded_item_csv.is_file():
        print(f"Reading {item_csv} ...")
        df_item = pd.read_csv(item_csv)
        print("MASKED_ITEM:\n", df_item)
        df_item_encoded = encode_items(df_item)
        df_item_encoded.to_csv(encoded_item_csv, index=False)
    else:
        df_item_encoded = pd.read_csv(encoded_item_csv)
    print("DF_ITEM:\n", df_item_encoded)

    item_info = merge_items_left(df_item_encoded, df_item_price_info)
    print("DF_ITEM_INFO:\n", item_info)

    no_missing = item_info[item_info['price'].notnull()]
    print("NOT MISSING:\n", no_missing)

    item_len = len(no_missing.index)

    no_missing.loc[:, 'properties'] = no_missing['properties'].apply(string_to_dmatrix_array)
    dataset = [str(int(z)) + ' ' + y for x, y, z in no_missing.values]
    write_rows_to_file('dataset.txt', dataset)

    ddata = xgb.DMatrix('dataset.txt')
    dtrain = ddata.slice([i for i in range(0, item_len) if i % 6 != 5])  # mod is not 5
    dtest = ddata.slice([i for i in range(0, item_len) if i % 6 == 5])  # mod is 5

    print("DTRAIN:\n", dtrain)
    print("DTEST:\n", dtest)

    labels = dtest.get_label()

    param = {'max_depth': 10, 'colsample_bytree': 0.3, 'eta': 0.9, 'verbosity': 1, 'objective': 'reg:linear', 'alpha': 10}
    # param = {'max_depth': 10, 'eta': 0.8, 'verbosity': 1, 'objective': 'reg:squarederror'}
    
    watchlist = [(dtest, 'eval'), (dtrain, 'train')]
    num_round = 100
    bst = xgb.train(param, dtrain, num_round, watchlist, early_stopping_rounds=5)
    
    avg_20_price = np.mean(ddata.get_label()[:20])
    allowance = avg_20_price // 25  # 4%


    preds = bst.predict(dtest)
    error = (sum(1 for i in range(len(preds)) if abs(preds[i] - labels[i]) > allowance) / float(len(preds))) if len(preds) > 0 else 1.0
    print('error=%f' % error)
    error = (sum(1 for i in range(len(preds)) if abs(preds[i] - labels[i]) > 1) / float(len(preds))) if len(preds) > 0 else 1.0
    print('$1 error=%f' % error)
    error = (sum(1 for i in range(len(preds)) if abs(preds[i] - labels[i]) > 5) / float(len(preds))) if len(preds) > 0 else 1.0
    print('$5 error=%f' % error)
    error = (sum(1 for i in range(len(preds)) if abs(preds[i] - labels[i]) > 10) / float(len(preds))) if len(preds) > 0 else 1.0
    print('$10 error=%f' % error)
    error = (sum(1 for i in range(len(preds)) if abs(preds[i] - labels[i]) > 20) / float(len(preds))) if len(preds) > 0 else 1.0
    print('$20 error=%f' % error)


    cv_results = xgb.cv(dtrain=dtrain, params=param, nfold=3,
                        num_boost_round=50,early_stopping_rounds=10,metrics="rmse", seed=123, callbacks=[xgb.callback.print_evaluation(show_stdv=True)])
    cv_results.head()
    print((cv_results["test-rmse-mean"]).tail(1))
    

    print("Finished calculating recommendations.")


if __name__ == '__main__':
    main()