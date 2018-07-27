import os
import pandas as pd
import numpy as np

import gc
gc.enable()



#https://www.kaggle.com/mlisovyi/lightgbm-hyperparameter-optimisation-lb-0-746

#import sys
#sys.path.append('./utils/*')
import  utils.Data_preprocessing as dp

from utils.EDA import EDA 
import matplotlib.pyplot as plt

PATH = '../input/'
Processed = '../input/processed/'
directory = os.path.dirname(Processed)
if not os.path.exists(directory):
    os.makedirs(directory)


###########################
##### JOIN Data Sets######
###########################

#load Bureau datasets
bureau_balance_agg_df = pd.read_csv(Processed+"bureau_balance_agg.csv")
bureau_balance_agg_df.shape

bureau_df=pd.read_csv(Processed+"bureau.csv")
bureau_df.shape

#merge Bureau and bureau_balances
merged_bureau_df = pd.merge(left=bureau_df, right=bureau_balance_agg_df, how='left', on= 'SK_ID_BUREAU' )
merged_bureau_df.shape

merged_bureau_df.head()
del bureau_df, bureau_balance_agg_df
gc.collect()

merged_bureau_df = dp.reduce_mem_usage(merged_bureau_df)
merged_bureau_df = dp.fill_numeric(merged_bureau_df,0)
dp.check_missing_data(merged_bureau_df)

merged_bureau_df.head()
#check loan counts for each customer
groupby_loans = merged_bureau_df.groupby('SK_ID_CURR')['SK_ID_BUREAU'].count()
groupby_loans.describe()
#consider last 6 loans for each customer

#group by SK_ID_CURR and sort by CREDIT_ACTIVE and DAYS_CREDIT
merged_bureau_df.CREDIT_ACTIVE.head() 
# merged_bureau_df[merged_bureau_df.CREDIT_ACTIVE=='Bad debt']

#load application df
app_df = pd.read_csv(Processed+"application.csv")

#One Hot encoding
cat_feat=dp.get_categorical_features(app_df)
app_df=dp.onehot_encoding(app_df,cat_feat)

#devide train and test
train_df=app_df[app_df.TARGET!=-999]
test_df = app_df[app_df.TARGET==-999]

#write to csv file
train_df.to_csv(PATH+'Final_app_train.csv', index=False)
test_df.to_csv(PATH+'Final_app_test.csv', index=False)
