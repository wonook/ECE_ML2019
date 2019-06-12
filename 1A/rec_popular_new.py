from pathlib import Path

import math
import click
import pandas as pd
import numpy as np
import tensorflow as tf

current_directory = Path(__file__).absolute().parent
default_data_directory = current_directory.joinpath('..', 'trivagoRecSysChallengeData2019_v2')

GR_COLS = ["user_id", "session_id", "timestamp", "step"]


def get_submission_target(df):
    """Identify target rows with missing click outs."""

    mask = df["reference"].isnull() & (df["action_type"] == "clickout item")
    df_out = df[mask]

    return df_out


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


def get_popularity(df):
    """Get number of clicks that each item received in the df."""

    df_item_clicks = (
        df
        .groupby("item_id")
        .size()
        .reset_index(name="n_clicks")
        .transform(lambda x: x.astype(int))
    )

    return df_item_clicks


# ######################


def get_yhat(Xs, Xi, Ws, Wi, Vs, Vi, w0):
    linear_terms = tf.add(w0, 
      tf.add(tf.reduce_sum(tf.multiply(Ws, Xs), 1, keepdims=True), 
        tf.reduce_sum(tf.multiply(Wi, Xi), 1, keepdims=True)))

    interactions = (tf.multiply(0.5,
        tf.reduce_sum(
            tf.matmul(
                tf.matmul(Xs, tf.transpose(Vs)),
                tf.transpose(tf.matmul(Xi, tf.transpose(Vi)))),
            1, keepdims=True)))

    return tf.add(linear_terms, interactions)

def bpr(yhat_pos, yhat_neg):
    return tf.reduce_mean(-tf.log_sigmoid(yhat_pos-yhat_neg))
def top1(yhat_pos, yhat_neg):
    # term1 = 
    return tf.reduce_mean(tf.nn.sigmoid(-yhat_pos+yhat_neg)+tf.nn.sigmoid(yhat_neg**2))  #, axis=0)
    # term2 = tf.nn.sigmoid(tf.diag_part(yhat_neg)**2) / self.batch_size
    # return tf.reduce_mean(term1 - term2)


# ######################

def process_to_feeddata(df, item_dict):
    x_s_vectors = []
    x_ip_vectors = []
    x_in_vectors = []
    for i in df[['session_vec', 'item_id', 'impressions']].itertuples(index=False):
        # print("Session_vec: {}({}), Item_id: {}({}), Impressions: {}({})".format(i[0], type(i[0]), i[1], type(i[1]), i[2], type(i[2])))
        for i_n in string_to_array(i[2]):
            if i[1] in item_dict and i_n in item_dict:
                x_s_vectors.append(string_to_array(i[0]))
                x_ip_vectors.append(string_to_array(item_dict[i[1]]))
                x_in_vectors.append(string_to_array(item_dict[i_n]))

    x_s_data = np.array(x_s_vectors)
    x_ip_data = np.array(x_ip_vectors)
    x_in_data = np.array(x_in_vectors)
    print("Xs_DATA:\n", x_s_data, "(",  x_s_data.shape, ")")
    print("Xi+_DATA:\n", x_ip_data, "(",  x_ip_data.shape, ")")
    print("Xi-_DATA:\n", x_in_data, "(",  x_in_data.shape, ")")
    return x_s_data, x_ip_data, x_in_data


def merge_results(df, item_dict, results):
    arr = []
    for i in df[['session_vec', 'item_id', 'impressions', "user_id", "session_id", "timestamp", "step"]].itertuples(index=False):
        resultset = {}
        for i_n in string_to_array(i[2]):
            if i_n in item_dict and (i[0], item_dict[i_n]) in results:
                resultset[results[(i[0], item_dict[i_n])].item(0)] = i_n
        result = []
        for k, v in sorted(resultset.items(), reverse=True):
            result.append(v)
        arr.append([i[3], i[4], i[5], i[6], " ".join([str(x) for x in result])])
    if arr:
        df_out = pd.DataFrame(np.array(arr), 
                              columns=["user_id", "session_id", "timestamp", "step", "item_recommendations"])
        return df_out


# ##################


item_meta_list = []

def save_item_meta_array(arr):
    for i in arr:
        if i not in item_meta_list:
            item_meta_list.append(i)


def array_to_str(arr):
    return "|".join([str(x) for x in arr])


def array_to_encoding(arr):
    encoding = np.zeros(len(item_meta_list), dtype=int)
    for i in arr:
        encoding[item_meta_list.index(i)] += 1
    return array_to_str(encoding)


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


session_meta_list = ['clickout item', 
                     'interaction item rating',
                     'interaction item info',
                     'interaction item image',
                     'interaction item deals', 
                     'change of sort order', 
                     'filter selection',
                     'search for item',
                     'search for destination',
                     'search for poi',
                     ]
session_ref_meta = {}
for i in session_meta_list:
    session_ref_meta[i] = []

def encode_session(session_df):
    # print("SESSION:\n", session_df)
    arr = []
    encoding = np.zeros(len(session_meta_list), dtype=int)
    for index, row in session_df.iterrows():
        # print("ROW:", row)
        action_type = row['action_type']
        reference = row['reference']
        if action_type == 'clickout item':
            arr.append([row['user_id'], row['timestamp'], row['step'], reference if reference is None or (not isinstance(reference, str) and math.isnan(reference)) else int(reference), row['impressions'], row['prices'], array_to_str(encoding)])
        if reference not in session_ref_meta[action_type]:
            session_ref_meta[action_type].append(reference)
        encoding[session_meta_list.index(action_type)] = session_ref_meta[action_type].index(reference) + 1

    # print("ARR:", arr)
    return arr

def encode_sessions(df_in):
    arr = []
    for session_id, group in df_in.groupby(['session_id']):
        res = encode_session(group)
        if res:
            for r in res:
                arr.append([session_id] + r)
    if arr:
        df_out = pd.DataFrame(np.array(arr), 
                              columns=['session_id', 'user_id', 'timestamp', 'step', 'item_id', 'impressions', 'prices', 'session_vec'])
        return df_out


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
    encoded_train_csv = data_directory.joinpath('train_encoded.csv')
    test_csv = data_directory.joinpath('test.csv')
    encoded_test_csv = data_directory.joinpath('test_encoded.csv')
    item_csv = data_directory.joinpath('item_metadata.csv')
    encoded_item_csv = data_directory.joinpath('item_metadata_encoded.csv')
    subm_csv = data_directory.joinpath('submission_popular.csv')
    subm_csv_bpr = data_directory.joinpath('submission_popular_bpr.csv')
    subm_csv_top1 = data_directory.joinpath('submission_popular_top1.csv')


    df_train_encoded = None
    df_item_encoded = None
    if not encoded_train_csv.is_file():
        print(f"Reading {train_csv} ...")
        df_train = pd.read_csv(train_csv)
        # df_train_masked = df_train[df_train['session_id'].isin(df_target['session_id'])]
        print("TRAIN:\n", df_train)
        df_train_encoded = encode_sessions(df_train)
        df_train_encoded.to_csv(encoded_train_csv, index=False)
    else:
        df_train_encoded = pd.read_csv(encoded_train_csv)
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

    df_test_encoded = None
    print("Get recommendations...")
    if not encoded_test_csv.is_file():
        print(f"Reading {test_csv} ...")
        df_test = pd.read_csv(test_csv)
        print("TEST:\n", df_test)
        df_test_encoded = encode_sessions(df_test)
        df_test_encoded.to_csv(encoded_test_csv, index=False)
    else:
        df_test_encoded = pd.read_csv(encoded_test_csv)


    # print("Identify target rows...")
    # df_target = get_submission_target(df_test_encoded)

    # df_expl = explode(df_target, "impressions")
    # df_expl.to_csv(encoded_test_csv, index=False)

    # print("Get popular clicks...")
    # df_popular = get_popularity(df_train_encoded)
    # print("Click popularity:\n", df_popular)

    # print("DF_EXPL:\n", df_expl)
    # df_out = calc_recommendation(df_expl, df_popular)

    # print(f"Writing {subm_csv}...")
    # df_out.to_csv(subm_csv, index=False)



    sec_vec_len = len(session_meta_list)
    item_vec_len = len(string_to_array(next(iter(item_dict.values()))))



    # number of latent factors
    k = 5
    batch_size = 1024 * 2

    # design matrix
    Xs = tf.placeholder('float', shape=[None, sec_vec_len])
    Xip = tf.placeholder('float', shape=[None, item_vec_len])
    Xin = tf.placeholder('float', shape=[None, item_vec_len])

    # bias and weights
    w0 = tf.Variable(tf.zeros([1]))
    Ws = tf.Variable(tf.zeros([sec_vec_len]))
    Wi = tf.Variable(tf.zeros([item_vec_len]))

    # interaction factors, randomly initialized 
    Vs = tf.Variable(tf.random_normal([k, sec_vec_len], stddev=0.01))
    Vi = tf.Variable(tf.random_normal([k, item_vec_len], stddev=0.01))

    # estimate of y, initialized to 0.
    y_hat_pos = tf.Variable(tf.zeros([batch_size, 1]))
    y_hat_neg = tf.Variable(tf.zeros([batch_size, 1]))
    
    y_hat_pos = get_yhat(Xs, Xip, Ws, Wi, Vs, Vi, w0)
    y_hat_neg = get_yhat(Xs, Xin, Ws, Wi, Vs, Vi, w0)

    loss_bpr = bpr(y_hat_pos, y_hat_neg)
    loss_top1 = top1(y_hat_pos, y_hat_neg)

    eta = tf.constant(0.001)
    optimizer_bpr = tf.train.AdagradOptimizer(eta).minimize(loss_bpr)
    optimizer_top1 = tf.train.AdagradOptimizer(eta).minimize(loss_top1)



    N_EPOCHS = 2
    # Launch the graph.
    init = tf.global_variables_initializer()
    print("Get popular items for BPR...")
    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(N_EPOCHS):
            epoch_loss = []
            for j in range(0, len(df_train_encoded.index), batch_size):
                upper_bound = min(j + batch_size, len(df_train_encoded.index))
                x_s_data, x_ip_data, x_in_data = process_to_feeddata(df_train_encoded.iloc[j:upper_bound], item_dict)
                print("==BPR==", j, " to ", upper_bound, " among ", len(df_train_encoded.index))
                optimizer_bpr_res, loss_bpr_res, yp, yn = sess.run([optimizer_bpr, loss_bpr, y_hat_pos, y_hat_neg], feed_dict={Xs: x_s_data, Xip: x_ip_data, Xin: x_in_data})
                print('Loss: ', loss_bpr_res)
                print('Y+: ', yp)
                print('Y-: ', yn)
                epoch_loss.append(loss_bpr_res)
            print('[epoch {}]: mean target value: {}'.format(epoch, np.mean(epoch_loss)))

        results_bpr = {}
        for j in range(0, len(df_test_encoded.index), batch_size):
            upper_bound = min(j + batch_size, len(df_test_encoded.index))
            print("==BPR EVAL==", j, " to ", upper_bound, " among ", len(df_test_encoded.index))
            test_x_s_data, test_x_ip_data, test_x_in_data = process_to_feeddata(df_test_encoded.iloc[j:upper_bound], item_dict)
            results_bpr.update(dict(zip( list(zip (list(map(array_to_str, test_x_s_data)), list(map(array_to_str, test_x_in_data)))), sess.run(y_hat_neg, feed_dict={Xs: test_x_s_data, Xin: test_x_in_data}))))
        # print('Predictions:', results_bpr)
        df_out_bpr = merge_results(df_test_encoded, item_dict, results_bpr)
        print("DF_OUT:\n", df_out_bpr)
        df_out_bpr.to_csv(subm_csv_bpr, index=False)


    print("Get popular items for TOP1...")
    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(N_EPOCHS):
            epoch_loss = []
            for j in range(0, len(df_train_encoded.index), batch_size):
                upper_bound = min(j + batch_size, len(df_train_encoded.index))
                x_s_data, x_ip_data, x_in_data = process_to_feeddata(df_train_encoded.iloc[j:upper_bound], item_dict)
                print("==TOP1==", j, " to ", upper_bound, " among ", len(df_train_encoded.index))
                optimizer_top1_res, loss_top1_res = sess.run([optimizer_top1, loss_top1], feed_dict={Xs: x_s_data, Xip: x_ip_data, Xin: x_in_data})
                print('Loss:', loss_top1_res)
                epoch_loss.append(loss_top1_res)
            print('[epoch {}]: mean target value: {}'.format(epoch, np.mean(epoch_loss)))

        results_top1 = {}
        for j in range(0, len(df_test_encoded.index), batch_size):
            upper_bound = min(j + batch_size, len(df_test_encoded.index))
            print("==TOP1 EVAL==", j, " to ", upper_bound, " among ", len(df_test_encoded.index))
            test_x_s_data, test_x_ip_data, test_x_in_data = process_to_feeddata(df_test_encoded.iloc[j:upper_bound], item_dict)
            results_top1.update(dict(zip( list(zip (list(map(array_to_str, test_x_s_data)), list(map(array_to_str, test_x_in_data)))), sess.run(y_hat_neg, feed_dict={Xs: test_x_s_data, Xin: test_x_in_data}))))
        # print('Predictions:', results_top1)
        df_out_top1 = merge_results(df_test_encoded, item_dict, results_top1)
        print("DF_OUT:\n", df_out_top1)
        df_out_top1.to_csv(subm_csv_top1, index=False)


    print("Finished calculating recommendations.")


if __name__ == '__main__':
    main()