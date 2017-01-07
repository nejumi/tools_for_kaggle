import glob
import os
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss, roc_auc_score
from pandas import DataFrame
import numpy as np
import pandas as pd

class Generalizer(object):
    def __init__(self, estimater, name):
        """
        Define the each unit of the base estimater for stacking or blending.
        
        -----Parameters-----
        estimater:    Estimater such as XGBClassifier() or RandomForestClassifier, which have "predict_proba" method.
        name :        Name of the estimater. str is expected. 
        """
        
        self.name = name
        self.estimater = estimater

    def stacking(self, X, y, X_sub, file_val, file_test, n_fold=5, fit_params=None, random_state=42, eval_metric="roc_auc_score"):
        """
        Generate meta-features of training data and test data by stacked generalization.
        
        -----Parameters-----
        X:            Features of training data. Pandas DataFrame is expected
        y:            Target of training data. Pandas DataFrame is expected.
        X_sub:        Features of test data. Pandas DataFrame is expected.
        file_val:     Pass of folder to save predicted meta-features for training data. str is expected.
        file_test:    Pass of folder to save predicted meta-features for test data. str is expected.
        n_fold:       Number of folds to generate meta-features for training data. int is expected. default is 5.
        random_state: Random seed for StratifiedKFold. int is expected. default is 42.
        eval_metric:  Evaluation function to calculate CV-scores. None, "log_loss", or "roc_auc_score" are expected.
        
        -----Returns-----
        hyp_1:        Meta-features for training data. It will be also saved as csv file.
        hyp_2:        Meta-features for test data. It will be also saved as csv file.
        """
        if not os.path.exists(file_val):
            os.makedirs(file_val)
            
        if not os.path.exists(file_test):
            os.makedirs(file_test)
            
        fit_params = fit_params if fit_params is not None else {}
            
        self.estimater.fit(X.as_matrix(), y.as_matrix(), **fit_params)
        hyp_2 = DataFrame(self.estimater.predict_proba(X_sub.as_matrix())[:,-1], 
                          index=X_sub.index, 
                          columns=[self.name])
        
        hyp_1 = DataFrame(index=X.index, 
                          columns=[self.name])
        
        skf = StratifiedKFold(y.values, n_folds=n_fold, shuffle=True, random_state=random_state)
        
        evals = {"log_loss": log_loss, "roc_auc_score": roc_auc_score}
        eval_scores = np.zeros(n_fold)
        
        for k, (train_iloc, test_iloc) in enumerate(skf):
            
            train_index = X.iloc[train_iloc].index
            test_index = X.iloc[test_iloc].index
            
            X_train, X_test = X.ix[train_index], X.ix[test_index]
            y_train, y_test = y.ix[train_index], y.ix[test_index]
            
            self.estimater.fit(X_train.as_matrix(), y_train.as_matrix(), **fit_params)
            
            if eval_metric is not None:
                eval_scores[k] = evals[eval_metric](y_test, self.estimater.predict_proba(X_test.as_matrix())[:,-1])
            
            output_temp = DataFrame(self.estimater.predict_proba(X_test.as_matrix())[:,-1], 
                                    index=X_test.index, 
                                    columns=[self.name])
            
            hyp_1.ix[test_index] = output_temp.ix[test_index]

        if eval_metric is not None:
            print("%s of %s: %-6.6f"%(eval_metric, self.name, eval_scores.mean()))
        
        hyp_1.to_csv(file_val + self.name + ".csv")
        hyp_2.to_csv(file_test + self.name + ".csv")
        
        return hyp_1, hyp_2
    
    def blending(self, X_train, y_train, X_sub, file_val, file_test, val_size = 0.1, random_state = 42, fit_params=None, eval_metric ="roc_auc_score"):
        """
        Generate meta-features of training data and test data by simple blending.
        
        -----Parameters-----
        X:            Features of training data. Pandas DataFrame is expected
        y:            Target of training data. Pandas DataFrame is expected.
        X_sub:        Features of test data. Pandas DataFrame is expected.
        file_val:     Pass of folder to save predicted meta-features for training data. str is expected.
        file_test:   Pass of folder to save predicted meta-features for test data. str is expected.
        val_size:     Size of the valdation data separated from training data. float is expected.
        random_state: Random seed for train_test_split. int is expected. default is 42.
        eval_metric:  Evaluation function to calculate CV-scores. None, "log_loss", or "roc_auc_score" are expected.
        
        -----Returns-----
        hyp_1:        Meta-features for training data. It will be also saved as csv file.
        hyp_2:        Meta-features for test data. It will be also saved as csv file.
        """
        if not os.path.exists(file_val):
            os.makedirs(file_val)
            
        if not os.path.exists(file_test):
            os.makedirs(file_test)
            
        fit_params = fit_params if fit_params is not None else {}
            
        x_train_, x_val, y_train_, y_val = train_test_split(X_train, y_train, test_size = val_size, random_state = random_state)
        
        self.estimater.fit(x_train_.as_matrix(), y_train_.as_matrix(), **fit_params)
        hyp_1 = DataFrame(self.estimater.predict_proba(x_val.as_matrix())[:,-1], 
                          index=x_val.index, 
                          columns=[self.name])
        
        if eval_metric is not None:
            eval_score = evals[eval_metric](y_val, self.estimater.predict_proba(x_val.as_matrix())[:,-1])
            print("%s of %s: %-6.6f"%(eval_metric, self.name, eval_score))
        
        self.estimater.fit(X_train.as_matrix(), y_train.as_matrix(), **fit_params)
        hyp_2 = DataFrame(self.estimater.predict_proba(X_sub.as_matrix())[:,-1], 
                          index=X_sub.index, 
                          columns=[self.name])
        
        hyp_1.to_csv(file_val + self.name + ".csv")
        hyp_2.to_csv(file_test + self.name + ".csv")
        
        return hyp_1, hyp_2, y_val

class Metafeatures_Generator(object):
    def bundle(self, file_val, file_test):
        """
        Bundle each individual meta-featues and generates training and test data for meta-classfiers.
        
        -----Parameters-----
        file_val:     Pass of folder to read the individual meta-features for training data. str is expected.
        file_test:   Pass of folder to read the individual meta-features for test data. str is expected.
        
        -----Returns-----
        hyp_1:        Training data for meta-classifiers. Pandas Dataframe is returned
        hyp_2:        Test data for meta-classifiers. Pandas Dataframe is returned
        """
        
        val_files =  [r.split('/')[-1] for r in glob.glob(file_val+'*')]
        print("Ensemble of val_files:" + file_val, val_files)
        test_files = [s.split('/')[-1] for s in glob.glob(file_test+'*')]
        print("Ensemble of test_files:" + file_test, test_files)
        print("Number of estimators is: " + str(len(test_files)))
        hyp1 = DataFrame()
        hyp2 = DataFrame()
        
        for filename1 in val_files:
            df_read1 = pd.read_csv(file_val + filename1, index_col=0)
            hyp1 = hyp1.join(df_read1, how="outer")
            
        for filename2 in test_files:
            df_read2 = pd.read_csv(file_test + filename2, index_col=0)
            hyp2 = hyp2.join(df_read2, how="outer")
            
        return hyp1, hyp2
