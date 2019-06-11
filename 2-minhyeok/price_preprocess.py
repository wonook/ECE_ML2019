# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 16:20:12 2015

@author: Balázs Hidasi
"""

import sys
import operator
import numpy as np
import pandas as pd
import datetime as dt
import json
from keras.utils import np_utils
from tqdm import tqdm
tqdm.pandas()

PATH_TO_ORIGINAL_DATA = '../data/raw/'
PATH_TO_PROCESSED_DATA = '../data/processed/'

TRAIN_FILENAME = 'train_price.txt'
TEST_FILENAME = 'test_price.txt'


################################################## ITEM META PARSING ##################################################

print("read item meta")

item_meta = pd.read_csv(PATH_TO_ORIGINAL_DATA + 'item_metadata.csv', sep=',', header=0, usecols=[0,1], dtype={0:str, 1:str}, index_col=0)

print("set columns")
item_meta.columns = ['properties']
item_meta.index.names = ['reference']

print("split properties")
item_meta["property_list"] = item_meta.properties.apply(lambda x: x.replace(" ", "_").split('|'))

print("make property map")
properties = np.array([y for x in item_meta.property_list.values.tolist() for y in x])

unique_properties = pd.Series(pd.Series(properties).unique()).sort_values(ascending=True)

propertyMap = pd.Series(data=np.arange(0, len(unique_properties)), index=unique_properties.values)

property_max = len(propertyMap)

print("make property vector")
item_meta["property_vector"] = item_meta.property_list.progress_apply(lambda x: np.sum(np.eye(property_max)[list(map(lambda y: propertyMap[y], x))], axis=0) )
print(item_meta)


################################################## ITEM CLICK PARSING ##################################################
print("read item click")
train = pd.read_csv(PATH_TO_ORIGINAL_DATA + 'train.csv', sep=',', header=0
					, usecols=[4,5,10,11]
					, dtype={4:str, 5:str, 10:str,  11:str})

# 0,		1,			2,			3,		4,				5,			6,			7,		8,		9,					10,				11
# user_id, 	session_id, timestamp, 	step,	action_type,	reference,	platform,	city,	device,	current_filters,	impressions,	prices
# 0RL8Z82B2Z1,aff3928535f48,1541037460,1,search for poi,Newtown,AU,"Sydney, Australia",mobile,,,
train.columns = ['action_type','referenceStr', 'impressions','prices']

mask = train["action_type"] == "clickout item"
clicks = train[mask]
clicks["reference"] = clicks.referenceStr.progress_apply(lambda x: int(x))

print(clicks)
print("item_counts")
item_counts = clicks.groupby(["reference"]).count()['action_type'].to_frame().sort_index()
item_counts.columns = ['item_counts']


print(item_counts, type(item_counts))


print("make impression_list")

clicks["impression_list"] = clicks.impressions.progress_apply(lambda x: x.split('|'))

print("make price_list")
clicks["price_list"] = clicks.prices.progress_apply(lambda x: x.split('|'))

print(clicks)

print("get price")
def get_price(row):
	try:
		price_index = row['impression_list'].index(row['referenceStr'])
	except:
		return None
	else:
		return row['price_list'][price_index]


clicks["price"] = clicks.progress_apply(get_price, axis=1)


clicks = clicks.groupby(["reference"]).first()

#clicks = clicks.set_index("reference")
print(clicks)

#clicks["price"] = clicks.progress_apply(lambda x: print(x, type(x)))

print("merge")
items = pd.merge(clicks, item_counts, left_index=True, right_index=True)
print(items)
del(items["action_type"])
del(items["impressions"])
del(items["prices"])
del(items["price_list"])

print("merge")
items = pd.merge(item_meta, items, left_index=True, right_index=True)
del(items["properties"])
del(items["property_list"])
del(items["impression_list"])
print(items)



train_len = int(len(items) * 0.7)
#print('Full train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train), train.session_id.nunique(), train.reference.nunique()))
items.iloc[:train_len].to_csv(PATH_TO_PROCESSED_DATA + TRAIN_FILENAME, sep='\t', index=False)

#print('Test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(test), test.session_id.nunique(), test.reference.nunique()))
items.iloc[train_len:].to_csv(PATH_TO_PROCESSED_DATA + TEST_FILENAME, sep='\t', index=False)

