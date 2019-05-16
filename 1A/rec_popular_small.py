from pathlib import Path

import click
import pandas as pd
import math
import numpy as np
import ast
import tensorflow as tf

current_directory = Path(__file__).absolute().parent
default_data_directory = current_directory.joinpath('..', 'trivagoRecSysChallengeData2019_v2')

GR_COLS = ["user_id", "session_id", "timestamp", "step"]


def get_submission_target(df_test):
    """Identify target rows with missing click outs."""

    mask = df_test["reference"].isnull() & (df_test["action_type"] == "clickout item")
    df_out = df_test[mask]

    return df_out


# def get_popularity(df_train):
#     """Get number of clicks that each item received in the df_train."""

#     mask = df_train["action_type"] == "clickout item"
#     df_clicks = df_train[mask]
#     df_item_clicks = (
#         df_clicks
#         .groupby("reference")
#         .size()
#         .reset_index(name="n_clicks")
#         .transform(lambda x: x.astype(int))
#         .sort_values(by='n_clicks', ascending=False)
#     )

#     return df_item_clicks


def string_to_array(s):
    """Convert pipe separated string to array."""

    if isinstance(s, str):
        out = s.split("|")
    elif math.isnan(s):
        out = []
    else:
        raise ValueError("Value must be either string of nan")
    return out


item_meta_list = []

def save_item_meta_array(arr):
    for i in arr:
        if i not in item_meta_list:
            item_meta_list.append(i)


def array_to_encoding(arr):
    encoding = np.zeros(len(item_meta_list), dtype=int)
    for i in arr:
        encoding[item_meta_list.index(i)] += 1
    return encoding


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

    df.loc[:, col_expl] = df[col_expl].apply(array_to_encoding)
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
            arr.append([row['user_id'], row['timestamp'], row['step'], int(reference), row['impressions'], row['prices'], encoding])
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


def merge_dfs(df_train_encoded, df_item_encoded):
    df_out = (
        df_train_encoded.merge(df_item_encoded, left_on="item_id", right_on="item_id", how="left"))
    return df_out[df_out['properties'].notnull()]


# def explode(df_in, col_expl):
#     """Explode column col_expl of array type into multiple rows."""

#     df = df_in.copy()
#     df.loc[:, col_expl] = df[col_expl].apply(string_to_array)  # set value for the column

#     df_out = pd.DataFrame(
#         {col: np.repeat(df[col].values,
#                         df[col_expl].str.len())
#          for col in df.columns.drop(col_expl)}
#     )

#     df_out.loc[:, col_expl] = np.concatenate(df[col_expl].values)
#     df_out.loc[:, col_expl] = df_out[col_expl].apply(int)

#     return df_out

# def group_concat(df, gr_cols, col_concat):
#     """Concatenate multiple rows into one."""

#     df_out = (
#         df
#         .groupby(gr_cols)[col_concat]
#         .apply(lambda x: ' '.join(x))
#         .to_frame()
#         .reset_index()
#     )

#     return df_out

# def calc_recommendation(df_expl, df_pop):
#     """Calculate recommendations based on popularity of items.

#     The final data frame will have an impression list sorted according to the number of clicks per item in a reference data frame.

#     :param df_expl: Data frame with exploded impression list
#     :param df_pop: Data frame with items and number of clicks
#     :return: Data frame with sorted impression list according to popularity in df_pop
#     """

#     df_expl_clicks = (
#         df_expl[GR_COLS + ["impressions"]]
#         .merge(df_pop,
#                left_on="impressions",
#                right_on="reference",
#                how="left")
#     )

#     df_out = (
#         df_expl_clicks
#         .assign(impressions=lambda x: x["impressions"].apply(str))
#         .sort_values(GR_COLS + ["n_clicks"],
#                      ascending=[True, True, True, True, False])
#     )
#     print("DF_OUT1:\n", df_out)

#     df_out = group_concat(df_out, GR_COLS, "impressions")
#     df_out.rename(columns={'impressions': 'item_recommendations'}, inplace=True)

#     return df_out



@click.command()
@click.option('--data-path', default=None, help='Directory for the CSV files')
def main(data_path):

    # calculate path to files
    data_directory = Path(data_path) if data_path else default_data_directory
    train_csv = data_directory.joinpath('train_small.csv')
    test_csv = data_directory.joinpath('test_small.csv')
    item_csv = data_directory.joinpath('item_metadata.csv')
    merged_csv = data_directory.joinpath('merged_small.csv')
    subm_csv = data_directory.joinpath('submission_popular_small.csv')

    print(f"Reading {test_csv} ...")
    df_test = pd.read_csv(test_csv)

    print("Identify target rows...")
    df_target = get_submission_target(df_test)
    print("DF_TARGET:\n", df_target)

    merged_dfs = None

    if not merged_csv.is_file():
        print(f"Reading {train_csv} ...")
        df_train = pd.read_csv(train_csv)
        # df_train_masked = df_train[df_train['session_id'].isin(df_target['session_id'])]
        print("TRAIN:\n", df_train)
        df_train_encoded = encode_sessions(df_train)
        print("DF_SESSION:\n", df_train_encoded)


        print(f"Reading {item_csv} ...")
        df_item = pd.read_csv(item_csv)
        df_item_masked = df_item[df_item['item_id'].isin(df_train_encoded['item_id'])]
        print("MASKED_ITEM:\n", df_item_masked)
        df_item_encoded = encode_items(df_item_masked)
        print("DF_ITEM:\n", df_item_encoded)


        merged_dfs = merge_dfs(df_train_encoded, df_item_encoded)
        merged_dfs.to_csv(merged_csv, index=False)
    
    else:
        merged_dfs = pd.read_csv(merged_csv)


    print("MERGED:\n", merged_dfs)

    vectors = []
    for i in merged_dfs[['session_vec', 'properties']].itertuples(index=False):
        print("I0", i[0].replace("[", "").replace("]", ""))
        print("I1", i[1].replace("\n", "").replace("[", "").replace("]", ""))
        if i[0] is not None and i[1] is not None:
            vectors.append((np.fromstring(i[0].replace("[", "").replace("]", ""), dtype='int', sep=' '), 
                            np.fromstring(i[1].replace("\n", "").replace("[", "").replace("]", ""), dtype='int', sep=' ')))
            # print(vectors)
    x_data = np.array(list(vectors))
    print("X_DATA:\n", x_data)

    y_data = merged_dfs['item_id'].values
    y_data.shape += (1, )
    print(y_data.dtype)
    print("Y_DATA:\n", y_data)



    n, p = x_data.shape

    # number of latent factors
    k = 5

    # design matrix
    X = tf.placeholder('float', shape=[n, p])
    # target vector
    y = tf.placeholder('float', shape=[n, 1])

    # bias and weights
    w0 = tf.Variable(tf.zeros([1]))
    W = tf.Variable(tf.zeros([p]))

    # interaction factors, randomly initialized 
    V = tf.Variable(tf.random_normal([k, p], stddev=0.01))

    # estimate of y, initialized to 0.
    y_hat = tf.Variable(tf.zeros([n, 1]))
    

    linear_terms = tf.add(w0,
      tf.reduce_sum(tf.multiply(W, X), 1, keep_dims=True))

    interactions = (tf.multiply(0.5,
        tf.reduce_sum(
            tf.sub(
                tf.pow( tf.matmul(X, tf.transpose(V)), 2),
                tf.matmul(tf.pow(X, 2), tf.transpose(tf.pow(V, 2)))),
            1, keep_dims=True)))

    y_hat = tf.add(linear_terms, interactions)


    # L2 regularized sum of squares loss function over W and V
    lambda_w = tf.constant(0.001, name='lambda_w')
    lambda_v = tf.constant(0.001, name='lambda_v')

    l2_norm = (tf.reduce_sum(
                tf.add(
                    tf.multiply(lambda_w, tf.pow(W, 2)),
                    tf.multiply(lambda_v, tf.pow(V, 2)))))

    error = tf.reduce_mean(tf.square(tf.sub(y, y_hat)))
    loss = tf.add(error, l2_norm)

    eta = tf.constant(0.1)
    optimizer = tf.train.AdagradOptimizer(eta).minimize(loss)


    # that's a lot of iterations
    N_EPOCHS = 1000
    # Launch the graph.
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(N_EPOCHS):
            indices = np.arange(n)
            np.random.shuffle(indices)
            x_data, y_data = x_data[indices], y_data[indices]
            sess.run(optimizer, feed_dict={X: x_data, y: y_data})

        print('MSE: ', sess.run(error, feed_dict={X: x_data, y: y_data}))
        print('Loss (regularized error):', sess.run(cost, feed_dict={X: x_data, y: y_data}))
        print('Predictions:', sess.run(y_hat, feed_dict={X: x_data, y: y_data}))
        print('Learnt weights:', sess.run(W, feed_dict={X: x_data, y: y_data}))
        print('Learnt factors:', sess.run(V, feed_dict={X: x_data, y: y_data}))

    # print("Get popular items...")
    # df_popular = get_popularity(df_train)
    # print("DF_POPULAR:\n", df_popular)

    # print("Get recommendations...")
    # df_expl = explode(df_target, "impressions")
    # print("EXPLODE:\n", df_expl)
    # df_out = calc_recommendation(df_expl, df_popular)
    # print("DF_OUT:\n", df_out)

    # print(f"Writing {subm_csv}...")
    # df_out.to_csv(subm_csv, index=False)

    print("Finished calculating recommendations.")


if __name__ == '__main__':
    main()
