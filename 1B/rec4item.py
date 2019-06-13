

import sys

sys.path.append('../..')

import time
import json
import heapq
import operator
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

# PATH_TO_TRAIN = '../data/processed/rsc15_train_full_action_mini.txt'
# PATH_TO_TEST = '../data/processed/rsc15_test_action_mini.txt'


# PATH_TO_TRAIN = '../data/processed/train_full_action_mini.txt'
# PATH_TO_TEST = '../data/processed/test_full_action_mini.txt'


# PATH_TO_TRAIN = '../data/processed/train_full_action.txt'
# PATH_TO_TEST = '../data/processed/test_full_action.txt'


PATH_TO_TRAIN = '../data/processed/train_full_action2.txt'
PATH_TO_TEST = '../data/processed/test_full_action2.txt'

# PATH_TO_TRAIN = '../data/processed/rsc15_train_full.txt'
# PATH_TO_TEST = '../data/processed/rsc15_test.txt'

PATH_TO_ORIGINAL_DATA = '../data/raw/'
PATH_TO_PROCESSED_DATA = '../data/processed/'

print("reading")

train = pd.read_csv(PATH_TO_ORIGINAL_DATA + 'train.csv', sep=',', header=0
					, usecols=[1,2,4,5]
					, dtype={1:str, 2:np.int32,4:str, 5:str})
#test = pd.read_csv(PATH_TO_ORIGINAL_DATA + 'test.csv', sep=',', header=0, usecols=[1,2,4,5], dtype={1:str, 2:np.int32, 4:str, 5:str})

print("processing")

train.columns = ['session_id','timestamp','action_type','reference']

mask = train["action_type"] == "clickout item"
train = train[mask]
train = train.sort_values(['session_id', 'timestamp'])
itemids = train["reference"].unique()
n_items = len(itemids)
itemidMap = pd.Series(data=np.arange(1, n_items+1), index=itemids)
itemMap = pd.Series(data=itemids, index=np.arange(1, n_items+1))
del(train)
del(itemids)
print(itemidMap)


#user_id	session_id	timestamp	step	action_type	reference	platform	city	device			current_filters	impressions	prices		item_idx_x	action_idx	item_idx_y
#0			1			2			3		4			5			6			7		8				9				10			11			12			13			14	
#user_id	session_id	timestamp	step	action_type	platform	city		device	current_filters	impressions		prices		action_idx	reference	item_idx_x	item_idx_y
# data = pd.read_csv(PATH_TO_TRAIN, sep='\t', 
# 					usecols=[0,1,2,3, 4,10,12,13,14,15], 
					# dtype={0: str, 1:str, 2:str, 3:str, 4:str, 10:str, 12:str, 13:str, 14:int, 15:str})
data = pd.read_csv(PATH_TO_TRAIN, sep='\t', dtype={"item_idx_x":int})
data = data.rename(index=str, columns={"item_idx_x": "item_idx", "item_idx_y":"actions"})
valid = pd.read_csv(PATH_TO_TEST, sep='\t', dtype={"item_idx_x":int})
valid = valid.rename(index=str, columns={"item_idx_x": "item_idx", "item_idx_y":"actions"})
# data.columns = ['user_id', 'session_id', 'timestamp', 'step', 'action_type', 'impressions', 'action_idx', 'reference', 'item_idx', 'actions']
#valid.columns = ['user_id', 'session_id', 'timestamp', 'step', 'action_type', 'impressions', 'action_idx', 'reference', 'item_idx', 'actions']

# test = pd.read_csv(PATH_TO_TEST, sep='\t', dtype={'ItemId':np.float64})

print("reading Done")
# mask = test["ItemId"].isnull()
# test = test[mask]

def bpr_loss(layer):

	# Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
	def loss(y_true, y_pred):
	    return K.mean(-K.log(K.sigmoid(y_pred - y_true)))

	# Return a function
	return loss

max_itemid = data.item_idx.max()
#data = data.iloc[:int(len(data)/2)]
#data = data.iloc[:int(len(data)/10)]
#data = data.iloc[:100]
#valid = valid.iloc[:100]

data['action_list'] = data.actions.apply(lambda x: np.array(list(map(int, x.split('|')))).reshape(-1, 1) )
#data['action_list'] = data.actions.apply(lambda x: list(map( lambda y: [y], json.loads(x)[::-1])))
valid['action_list'] = valid.actions.apply(lambda x: np.array(list(map(int, x.split('|')))).reshape(-1, 1) )
#valid['action_list'] = valid.actions.apply(lambda x: list(map( lambda y: [y], json.loads(x)[::-1])))
#data['impression_list'] = data.impressions.apply(lambda x: list(map(lambda y: itemidMap[y], x.split('|'))))
#valid['impression_list'] = valid.impressions.apply(lambda x: list(map(lambda y: itemidMap[y], x.split('|'))))
#print(data.action_list)
#print(valid.action_list)
#exit()


action_len = len(data['action_list'].iloc[0])
#impression_len = len(data['impression_list'].iloc[0])


embedding_dim = 50
input_len = 1
batch_size = 150
print(f"max item id: {max_itemid}")
print(f"action len: {action_len}")

np.random.seed(0)



action_in = Input(shape=(action_len, 1), name="action_in")
action_rnn = SimpleRNN(1, input_shape=(action_len, 1))(action_in)
action_dense = Dense(1, activation = "linear")(action_rnn)


item_in = Input(shape=(1,), name="item_in")
item_embedding = Embedding(max_itemid+1, embedding_dim, input_length=input_len, name="item_embedding")(item_in)
item_gru = GRU(units=50, activation='tanh', recurrent_activation='tanh', use_bias=True, name="item_gru")(item_embedding)



merged = concatenate([item_gru, action_dense], name='merge')
#model = Model(inputs=[action_in], outputs=action_dense)

final_dense =  Dense(max_itemid+1, activation='softmax', name="final_dense")(merged)

final_drop = Dropout(0.5)(final_dense)
final_dense2 = Dense(1, activation='softmax', name='final_dense2')(final_drop)
model  = Model(inputs=[item_in, action_in], outputs=final_dense2)


#model.compile(loss='mse', optimizer='adam')
model.compile(loss=bpr_loss(final_dense2), optimizer='rmsprop')

# get_softmax_output = K.function([model.get_layer("item_in").input, model.get_layer("action_in").input],
#                                   [model.get_layer("final_dense").output])



model.summary()

items = np.array(data.item_idx.values.tolist())
actions = np.array(data.action_list.values.tolist())
print("item and action shape")
print(items.shape)
print(actions.shape)

#y_array = np.zeros((len(input_array), max_itemid+1))
#y_array[np.arange(len(input_array)), input_array] = 1
#one_hot_y = np_utils.to_categorical(items)
#print("one hot shape")
#print(one_hot_y.shape)
model.fit([items[:-1], actions[:-1]], items[1:]#one_hot_y[1:]
	, epochs=10, batch_size=batch_size, verbose=1)

import random
maplen = len(itemMap)
def toRef(x):
	a = []
	for y in x:
		if y in itemMap:
			a.append(itemMap[y])
		else:
			idx = random.randint(0, maplen)
			a.append(itemMap.iloc[idx])

	if len(a) < 25:
		for i in range(0, 25-len(a)):
			idx = random.randint(0, maplen)
			a.append(itemMap.iloc[idx])
	return a
		#else:
		#	a.append(itemMap[y])
#input_array = np.array(data.item_idx.values.tolist())


intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer("final_dense").output)
#print(len(data), len(output_array), len(outputs))
#print(outputs)
fout = open("submit.csv", "w")
fout.write("user_id,session_id,timestamp,step,item_recommendations\n")

titems = np.array(valid.item_idx.values.tolist())

tactions = np.array(valid.action_list.values.tolist())

stime = time.time()
#total = int(len(titems) / 40)
print("test total: ", total)
#total = 175
for start in range(0, total, 100):
	#sys.stdout.write(f"{start} / {total}\t\t{time.time()-stime}\r")
	end = start + 100
	if end > total:
		end = total
	
	output_array = intermediate_layer_model.predict([titems[start:end], tactions[start:end]])

	#output_array = get_softmax_output([items[:-1], actions[:-1]])[0]
	#output_array = model.predict([items[:-1], actions[:-1]])
	outputs = list(map(lambda x: sorted(range(len(x)), key=lambda i: x[i], reverse=True)[:25], output_array))
	outputRef = list(map(toRef, outputs))

	for i in range(start, end):
		row = valid.iloc[i]
		if row.reference == -1:
			
			#fout.write()
			if i > 0:
				ref_str = ""
			
				for ref in outputRef[i-start-1]:
					ref_str = ref_str + str(ref) + " "
				out_str = f"{row.user_id},{row.session_id},{row.timestamp},{row.step},"
			
				fout.write(out_str + ref_str)
				fout.write('\n')

	sys.stdout.write(f"{end} / {total}\t\t{time.time()-stime}\r")
#print(zip(*heapq.nlargest(25, enumerate(output_array), key=operator.itemgetter(1)))[0])

# outputs = list(map(lambda x: x.sort(reverse=True)[:25]))
# #outputIdx = np.argmax(output_array, axis=1)
# outputRef = list(map(lambda x: itemMap[x], outputIdx))


# print(items[1:])
# print(output_array)
# #print(np.argmax(output_array, axis=0))
# print(outputIdx)
# print(outputRef)

#print(a)
#print(a.shape)
#model.fit(np.array(data.action_list.values.tolist()), np.array(data.action_idx.values.tolist()), epochs=1, batch_size=3, verbose=1)


#print(data.ItemIdx.values.tolist())

#item_act = ELU(alpha=0.5)(item_dense)
#item_dense2 =  Dense(n_items, activation='softmax')(item_act)


#item_dense3 =  Dense(1)(item_dense)
#item_act3 = ELU(alpha=0.5)(item_dense3)



#output = model.get_layer("item_dense").output
#for layer in model.layers:
#	if layer.name == "item_dense":
#		outputs = model.get_layer
#exit()
#model = Sequential()
#model.add(Embedding(max_itemid, embedding_dim, input_length=input_len))

# the model will take as input an integer matrix of size (batch, input_length).
# the largest integer (i.e. word index) in the input should be
# no larger than 999 (vocabulary size).
# now model.output_shape == (None, 10, 64), where None is the batch dimension.

#input_array = np.random.randint(1000, size=(32, 1))

#print(input_array)

#model.compile(loss='mse', optimizer='rmsprop')


#input_array = np.array(data.ItemIdx.values.tolist()).reshape((-1, batch_size, input_len))


#assert output_array.shape == (32, 10, 64)






