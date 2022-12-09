# pipeline
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

import pandas as pd

# scaling
from sklearn.preprocessing import  StandardScaler, MinMaxScaler

# 次元削減
from sklearn.decomposition import PCA

import config

"""
パイプラインの定義

sklearn.pipeline.Pipelineを使用する

基本テンプレート

# BaseEstimatorとTransformerMixinを継承する
class ClassName(BaseEstimator, TransformerMixin):
    def __init__(self, selector=None, *):
        # self.selector　は定義する
        self.selector=selector
    
    # fit関数を定義する（引数: X, y）
    # return: self
    def fit(self, X, y):
        return self
    
    # transform関数を定義する（引数: X）
    # 本来返り値はnp.ndarrayでも良いが、特徴量名を追跡する必要があるので、pd.DataFrame形式を返す
    def transform(self, X):
        return pd.DataFrame()
"""

# 標準化を行うパイプラインクラス
class StandardScalerAsDataFrame(BaseEstimator, TransformerMixin):
    def __init__(self, selector=None):
        self.selector = selector
        self.std = StandardScaler()

    def fit(self, X, y=None):
        self.std.fit(X)
        return self

    def transform(self, X):
        return pd.DataFrame(self.std.transform(X), index=X.index, columns=X.columns)


# 正規化を行うパイプラインクラス
class NormalizationAsDataFrame(BaseEstimator, TransformerMixin):
    def __init__(self, selector=None):
        self.selector = selector
        self.norm = MinMaxScaler()

    def fit(self, X, y=None):
        self.norm.fit(X)
        return self

    def transform(self, X):
        return pd.DataFrame(self.norm.transform(X), index=X.index, columns=X.columns)


# PCAで次元削減を行うクラス
class PCAAsDataFrame(BaseEstimator, TransformerMixin):
    def __init__(self, selector=None):
        self.selector = selector
        self.pca = PCA(n_components=0.95, random_state=config.SEED)

    def fit(self, X, y=None):
        self.pca.fit(X)
        return self

    def transform(self, X):
        return pd.DataFrame(
            self.pca.transform(X),
            index=X.index,
            columns=self.pca.get_feature_names_out(),
        )

    def show_progress(self, X):
        # 主成分負荷量の計算と表示
        loadings = pd.DataFrame(self.pca.components_.T, index=X.columns)
        # 主成分スコアの計算
        score = pd.DataFrame(self.pca.transform(X), index=X.index)
        # 寄与率:各主成分がどれくらいデータを説明できているのかを表す指標
        contribution_ratios = pd.DataFrame(self.pca.explained_variance_ratio_)
        # 累積寄与率:この寄与率を累積して,ある寄与率に達するまでには第何主成分までが必要かを見ることが多い
        cumulative_contribution_ratios = contribution_ratios.cumsum()
        
        print(
            "loadings: {0}, score: {1}, 寄与率: {2}, 累積寄与率: {3} ".format(
                loadings, score, contribution_ratios, cumulative_contribution_ratios
            )
        )