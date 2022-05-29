from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pandas as pd

# primero eliminamos todas las columnas que no queremos
class eliminaColumnas(BaseEstimator, TransformerMixin):
    def __init__(self, cols_to_remove=None):
        self.cols_to_remove = cols_to_remove
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        _X = X.copy()
        _X.drop(self.cols_to_remove, axis=1,inplace=True)
        
        return _X

    
class customPipeline(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.train_cols_dummies = None
        self.target_col = "Survived"

    def fit(self, X, y=None):
        # solo se llama para el train
        
        # entonces hacemos el dummies aqui que seria para el train
        print('<fit>------> dummies')
        X = pd.get_dummies(X)
        self.train_cols_dummies = X.columns
        
        # guardamos las columnas a la nube
        print('<fit>------> Saving encoded columns')
        #self.objet_storage.save_object_in_cos(X.columns, 'encoded_columns', self.timestamp)

        return self
        
    
    def transform(self, X, y=None):

        if self.train_cols_dummies is not None:
            # no estamos en train
            # hacemos dummies y comparamos las columnas con las de train
            print('<transform>------> test dummies')
            X = pd.get_dummies(X)
            
            # mismas cols que en train
            print('<transform>------> test igualamos columnas')
            X = X.reindex(labels = self.train_cols_dummies, axis = 1, fill_value = 0)    

        # creación de variable Child de tipo booleana
        print('------> Creating child')
        X['Child'] = 0
        X.loc[X.Age < 16, 'Child'] = 1
        
        return X