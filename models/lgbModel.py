import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from lightgbm import LGBMClassifier

import gc


version='base'
gc.enable()

# CSV Data Loading
data = pd.read_csv('../input/Final_app_train.csv')
test = pd.read_csv('../input/Final_app_test.csv')

y = data['TARGET']
del data['TARGET']

excluded_feats = ['SK_ID_CURR']
features = [f_ for f_ in data.columns if f_ not in excluded_feats]


# Modeling
folds = KFold(n_splits=5, shuffle=True, random_state=123)
oof_preds = np.zeros(data.shape[0])
sub_preds = np.zeros(test.shape[0])

for n_fold, (trn_idx, val_idx) in enumerate(folds.split(data)):
    trn_x, trn_y = data[features].iloc[trn_idx], y.iloc[trn_idx]
    val_x, val_y = data[features].iloc[val_idx], y.iloc[val_idx]
    
    clf = LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.1,
        num_leaves=123,
        colsample_bytree=.8,
        subsample=.9,
        max_depth=15,
        reg_alpha=.1,
        reg_lambda=.1,
        min_split_gain=.01,
        min_child_weight=2
    )
    
    clf.fit(trn_x, trn_y, 
            eval_set= [(trn_x, trn_y), (val_x, val_y)], 
            eval_metric='auc', verbose=250, early_stopping_rounds=150
           )
    
    oof_preds[val_idx] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)[:, 1]
    sub_preds += clf.predict_proba(test[features], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits
    
    print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))
    del clf, trn_x, trn_y, val_x, val_y
    gc.collect()
    
print('Full AUC score %.6f' % roc_auc_score(y, oof_preds))   

test['TARGET'] = sub_preds

test[['SK_ID_CURR', 'TARGET']].to_csv('../output/lgb_submission_{}.csv'.format(version), index=False, float_format='%.8f')
