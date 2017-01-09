from pandas import DataFrame
import numpy as np
import pandas as pd

class pseudo_labeler(object):
    def __init__(self, estimater):
        """
        Define a pseudo-labeler for semi-supervised learning.
        
        -----Parameters-----
        estimater:    Estimater such as XGBClassifier() or RandomForestClassifier(), which have "predict_proba" method.
        """
        
        self.estimater = estimater
    
    def pseudo_labeling(self, X_train, y_train, X_test, max_iter=100, th_confidence=0.95, fit_params=None):
        """
        Extract test data with enough confidence and conduct pseudo-labeling.
        
        -----Parameters-----
        X_train:       Features of training data. Pandas DataFrame is expected
        y_train:       Target of training data. Pandas DataFrame is expected.
        X_test:        Features of test data. Pandas DataFrame is expected.
        max_iter:      Maximun number for iteration of pseudo-labeling. int is expected.
        th_confidence: Threshold of the confidence. float is expected.
        fit_params:    Parameters for "fit" method. dict is expected.
        
        -----Returns-----
        X_conf:        Features of test data with enough confidence.
        y_conf:        Target of test data. Because they have enough confidence, you can treat them as data with response==1.
        """

        continu = 1
        X_conf = DataFrame()
        y_conf = DataFrame()
        
        y_train = DataFrame(y_train) # Although y_train is expected to be a DataFrame, we tends to input a Series for it.

        for iter_ in range(max_iter):
            if continu > 0:
                if iter_ > 0:
                    X = pd.concat([X_train, X_conf], axis=0)
                    y = pd.concat([y_train, y_conf], axis=0)
                    
                if iter_ == 0:
                    X = X_train
                    y = y_train

                print("Processing " + str(iter_+1) + " iteration")

                fit_params = fit_params if fit_params is not None else {}
                self.estimater.fit(X.as_matrix(), y.as_matrix(), **fit_params)

                df_prob = DataFrame(self.estimater.predict_proba(X_test.as_matrix())[:,1], 
                                    index=X_test.index, 
                                    columns=["probability"])

                conf_index = df_prob[df_prob.probability > th_confidence].index
                print(conf_index)

                X_conf_ = X_test[X_test.index.isin(conf_index)]

                y_conf_ = DataFrame(index=X_conf_.index)
                y_conf_.index.names = y_train.index.names
                y_conf_[y_train.columns[0]] = 1
                
                X_conf = pd.concat([X_conf, X_conf_], axis=0)
                y_conf = pd.concat([y_conf, y_conf_], axis=0)
                
                X_test.drop(conf_index, axis=0, inplace=True)

                continu = X_conf_.shape[0]

                print(str(continu) + " samples with enough confidence were found at this iteration.")

        del X, y, X_train, y_train, X_test, X_conf_, y_conf_
        print("Finished!")
        
        return X_conf, y_conf

