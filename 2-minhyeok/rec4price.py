

import sys

sys.path.append('../..')

import json
import numpy as np
import pandas as pd
import gru4rec
import baseGRU
import ActionGRU
import evaluation
import base_evaluation
import action_evaluation
import session_aware_gru
import sessionGRU

from keras.models import Sequential, Model
from keras.layers import SimpleRNN, Dense, Input, Embedding, GRU, ELU, concatenate, Dropout
import keras.backend as K
from keras.utils import np_utils
from theano import tensor as T
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from keras.utils import plot_model

tqdm.pandas()

PATH_TO_TRAIN = '../data/processed/train_price.txt'
PATH_TO_TEST = '../data/processed/test_price.txt'


#PATH_TO_TRAIN = '../data/processed/train_full_action.txt'
#PATH_TO_TEST = '../data/processed/test_full_action.txt'

# PATH_TO_TRAIN = '../data/processed/rsc15_train_full.txt'
# PATH_TO_TEST = '../data/processed/rsc15_test.txt'

PATH_TO_ORIGINAL_DATA = '../data/raw/'
PATH_TO_PROCESSED_DATA = '../data/processed/'

print("reading")
data = pd.read_csv(PATH_TO_TRAIN, sep='\t', dtype={0:str, 1:str, 2:str, 3:float, 4:float}, usecols=[0,1,2,3,4])
valid = pd.read_csv(PATH_TO_TEST, sep='\t', dtype={0:str, 1:str, 2:str, 3:float, 4:float}, usecols=[0,1,2,3,4])
#valid = pd.read_csv(PATH_TO_TEST, sep='\t', dtype={'ItemId':np.int64})
# test = pd.read_csv(PATH_TO_TEST, sep='\t', dtype={'ItemId':np.float64})
print("reading Done")
# mask = test["ItemId"].isnull()
# test = test[mask]


del(data["impression_list"])
del(valid["impression_list"])

data = data.rename(index=str, columns={"property_vector": "_property_vector"})
valid = valid.rename(index=str, columns={"property_vector": "_property_vector"})

data['property_vector'] = data._property_vector.progress_apply(lambda x: list(map(float , x.replace("[", " ").replace("]", " ").replace("\n", "").split(" ")[1:-1])))
valid['property_vector'] = valid._property_vector.progress_apply(lambda x: list(map(float , x.replace("[", " ").replace("]", " ").replace("\n", "").split(" ")[1:-1])))
del(valid['_property_vector'])
del(data['_property_vector'])

mask = data["item_counts"] != None
data = data[mask]
mask = data["item_counts"] != float('nan')
data = data[mask]
mask = np.isnan(data["item_counts"]) != True
data = data[mask]
mask = data["price"] != None
data = data[mask]
mask = data["price"] != float('nan')
data = data[mask]
mask = np.isnan(data["price"]) != True
data = data[mask]


mask = valid["item_counts"] != None
valid = valid[mask]
mask = valid["item_counts"] != float('nan')
valid = valid[mask]
mask = np.isnan(valid["item_counts"]) != True
valid = valid[mask]
mask = valid["price"] != None
valid = valid[mask]
mask =  valid["price"] != float('nan')
valid = valid[mask]
mask = np.isnan(valid["price"]) != True
valid = valid[mask]

print(data)
print(valid)

dprice_min, dprice_max = data.price.min(), data.price.max()
vprice_min, vprice_max = valid.price.min(), valid.price.max()

data['norm_price'] = data.price.progress_apply(lambda x: (x - dprice_min) / (dprice_max - dprice_min + 1.0e-7))
valid['norm_price'] = valid.price.progress_apply(lambda x: (x - vprice_min) / (vprice_max - vprice_min + 1.0e-7))

property_len = len(data.iloc[0].property_vector)
property_in = Input(shape=(property_len, ), name="property_in")
property_out = Dense(property_len, activation = "linear")(property_in)

count_in = Input(shape=(1, ), name="count_in")
count_out = Dense(1, activation = "linear")(count_in)

merged = concatenate([property_out, count_out])

hidden1 = Dense(property_len + 1)(merged)
hidden1_act = ELU(alpha=0.5)(hidden1)

hidden2 = Dense(property_len + 1)(hidden1)
hidden2_act = ELU(alpha=0.5)(hidden2)


final_out = Dense(1, activation="tanh")(hidden2_act)

model = Model(inputs=[property_in, count_in], outputs=final_out)


print("mean: ", valid.price.mean())

model.compile(loss='mse', optimizer='adagrad')


model.fit([np.array(data.property_vector.values.tolist()), np.array(data.item_counts.values.tolist())]
		, np.array(data.norm_price.values.tolist())
		, epochs=20, batch_size=20, verbose=1)


property_array = np.array(valid.property_vector.values.tolist())
count_array = np.array(valid.item_counts.values.tolist())
output_array = model.predict([property_array, count_array])

print(np.array(output_array))
recovered_output = np.array(list(map(lambda x: x * (dprice_max - dprice_min + 1.0e-7) + dprice_min, output_array)))
print(recovered_output)
ytrue = np.array(valid.price.values.tolist()).reshape(1, -1)
yhat = recovered_output.reshape(1, -1)
print(ytrue)
print(yhat)
mse = mean_squared_error(ytrue, yhat)
mae = mean_absolute_error(ytrue, yhat)
print(f"mse: {mse}")
print(f"mae: {mae}")
print(property_len)

exit()
