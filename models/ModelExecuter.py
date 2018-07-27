
import pandas as pd
import numpy as np
import gc
gc.enable()
import  utils.Data_preprocessing as dp

def runModels(version):
    # CSV Data Loading
    X_data = pd.read_csv('../input/Final_app_train.csv')
    X_data=dp.reduce_mem_usage(X_data)

    ydata = X_data['TARGET']
    del X_data['TARGET']
    
    #remove unused colums
    excluded_feats = ['SK_ID_CURR','TARGET']
    features = [f_ for f_ in X_data.columns if f_ not in excluded_feats]
    
    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_data[features] = sc.fit_transform(X_data[features])
    
    #load test data and apply scaling 
    test = pd.read_csv('../input/Final_app_test.csv')    
    test=dp.reduce_mem_usage(test)    
    
    test[features] = sc.transform(test[features])
    #no of folds
    k=10   
    
    
    
    #############Naive Bayes ########################################
    print('Running Model - Naive Bayes/n###########################')
    clf,param_grid,fit_params   = naive_bayes()
    filename='../output/NB_submission_{}.csv'.format(version)
    executeModel(clf,param_grid,fit_params,k,X_data,ydata, test, features,filename )
    del clf,param_grid,fit_params
    gc.collect()
    #############################################################################
    
    #############Random Forest ########################################
    print('Running Model - Random Forest/n###########################')
    clf,param_grid,fit_params   = randomForest()   
    filename='../output/RF_submission_{}.csv'.format(version)
    executeModel(clf,param_grid,fit_params,k,X_data,ydata, test, features,filename )
    del clf,param_grid,fit_params
    gc.collect()
    #############################################################################
    
    
    # ##############lightGBM ###################
    '''
    # Splitting the dataset into the Training set and Test set
    from sklearn import cross_validation
    
    X_train, X_val, ytrain, yval = cross_validation.train_test_split(X_data, ydata, test_size = 0.2, random_state = 0)
    
    del X_data, ydata
    gc.collect()
    
    X_train = X_train[features]
    X_val = X_val[features]
    
    clf,param_grid,fit_params   = lightGBM()
    fit_params['eval_set']=[(X_train,ytrain),(X_val,yval)]
    
    
    filename='../output/lgb_submission_{}.csv'.format(version)
    executeModel(clf,param_grid,fit_params,k,X_train,ytrain, test, features,filename )
    del clf,param_grid,fit_params
    gc.collect()
    '''
    #############################################################################

def randomForest():
    # Fitting Random Forest Classification to the Training set
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier( random_state=123)    
    param_grid ={
                    'n_estimators':[10,50,100],
                    'max_features':['auto','sqrt','log2'],
                    'criterion':['gini','entropy'],
                    #'':123
                    'bootstrap':[True,False],
                    'min_samples_leaf':[1,5,10,]
                    
                }
    fit_params ={}
    
    return clf,param_grid, fit_params
def naive_bayes():
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    param_grid ={}
    fit_params ={}
    
    return clf,param_grid, fit_params 

    
    
def executeModel(clf,param_grid,fit_params,k,X_train,ytrain, test, features,filename ):    
    trainedModel=trainTuneModel(clf,param_grid,fit_params,k,X_train,ytrain)    
    #predict and save
    test['TARGET']=trainedModel.predict(test[features])
    test[['SK_ID_CURR','TARGET']].to_csv(filename, index=False, float_format='%.8f')
    

def trainTuneModel(clf,param_grid,fit_params,k,X_train,ytrain):    
    #train Model using grid search
    
    from sklearn.model_selection import GridSearchCV
    grid = GridSearchCV(estimator = clf, 
                           param_grid = param_grid, 
                           fit_params = fit_params,
                           scoring = 'roc_auc', 
                           cv=k, 
                           refit=True)
    #Fit Data
    BestModel = grid.fit(X_train, ytrain)    
    print (BestModel.best_score_)
    print (BestModel.best_params_)
    return BestModel
    
    #predict and save resutls
    

    

def lightGBM():
    #import model
    from lightgbm import LGBMClassifier
    
    clf = LGBMClassifier()
    param_grid ={ 
                    'learning_rate': [0.1,0.05],
                    'n_estimators': [2000],
                    'num_leaves': [128],
                    'min_data_in_leaf':[100,500,1000],
                    'boosting_type' : ['gbdt'],
                    'objective' : ['binary'],
                    'random_state' : [501], # Updated from 'seed'
                    'colsample_bytree' : [0.75,0.5],
                    'subsample' : [0.7,0.5],
                    'max_depth': [5,10,15,20],
                    'reg_alpha' : [0.1],
                    'reg_lambda': [0.1]
                    }
    fit_params ={
                #'eval_metric':['auc'], 
                'verbose':[250], 
                'early_stopping_rounds':50
                }
    
    return clf,param_grid, fit_params 

def cross_val(clf,k, X_train,y_train,score='auc'):
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(estimator = clf, X = X_train, y = y_train, cv = k)
    print('MEAN Cross Validation Score({}) {}'.format(score, np.mean(scores)))

runModels('baseline')
