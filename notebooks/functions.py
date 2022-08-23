import os
import random
import pickle

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import graphviz
import pydotplus
from IPython.display import Image
from six import StringIO

# データセット分割
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

# 前処理
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 特徴量選択
from sklearn.feature_selection import GenericUnivariateSelect


# 機械学習アルゴリズム
from sklearn.linear_model import LogisticRegression  # ロジスティック回帰
from sklearn.neighbors import KNeighborsClassifier  # K近傍法
from sklearn.svm import SVC  # サポートベクターマシン
from sklearn.tree import DecisionTreeClassifier, export_graphviz  # 決定木
from sklearn.ensemble import RandomForestClassifier  # ランダムフォレスト
from sklearn.ensemble import AdaBoostClassifier  # AdaBoost
from sklearn.naive_bayes import GaussianNB  # ナイーブ・ベイズ
from sklearn.decomposition import LatentDirichletAllocation as LDA  # 線形判別分析
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA  # 二次判別分析
# 評価指標
from tqdm import tqdm
from sklearn.model_selection import learning_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import warnings

from config import SEED, bcm_names, classifiers


#-----------------------------------------------------------------------------
def make_dir(dir_name: str):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


# seedの固定
def fix_seed(seed: int):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)


# pickleオブジェクトにして保存
def pickle_dump(obj, path):
    with open(path, mode="wb") as f:
        pickle.dump(obj, f)


def pickle_load(path):
    with open(path, mode="rb") as f:
        data = pickle.load(f)
        return data
#-----------------------------------------------------------------------------
def check(df):
    col_list = df.columns.values  # 列名を取得
    row = []
    for col in col_list:
        max_value=None
        min_value=None
        unique = ""
        value_counts = ""
        # 最大・最小の数値を取得
        if df[col].dtype==int or df[col].dtype==float:
            max_value=df[col].max()
            min_value=df[col].min()
        # ユニークな値のカウント
        if df[col].nunique() < 12:
            unique = df[col].unique()
            value_counts = df[col].value_counts().to_dict()
        tmp = (
            col,  # 列名
            df[col].dtypes,  # データタイプ
            df[col].isnull().sum(),  # null数
            df[col].count(),  # データ数 (欠損値除く)
            max_value,
            min_value,
            df[col].nunique(),  # ユニーク値の数 (欠損値除く)
            unique,  # ユニーク値
            value_counts,  # ユニーク値のそれぞれの個数
        )
        row.append(tmp)  # tmpを順次rowに保存
    df = pd.DataFrame(row)  # rowをデータフレームの形式に変換
    df.columns = [
        "feature",
        "dtypes",
        "nan",
        "count",
        "max",
        "min",
        "num_unique",
        "unique",
        "unique_counts",
    ]  # データフレームの列名指定
    # unique_countsの中身確認のために横幅拡張
    d = dict(selector=".col8", props=[("min-width", "200px")])  # name
    return df.style.set_table_styles([d])

# 重複した特徴量のrename関数
def rename_duplicated_columns(df):
    df = df.copy()
    df_columns = df.columns
    new_columns = []
    for item in df_columns:
        counter = 0
        newitem = item
        while newitem in new_columns:
            counter += 1
            newitem = "{}_{}".format(item, counter)
        new_columns.append(newitem)
    df.columns = new_columns
    return df

#-----------------------------------------------------------------------------

# 基本的なスコアの表示（面倒なので関数化した）
def show_scores(y_test: pd.Series, y_pred: pd.Series):
    print("accuracy: ", accuracy_score(y_test, y_pred))
    print("precision: ", precision_score(y_test, y_pred))
    print("recall: ", recall_score(y_test, y_pred))
    print("f1 score: ", f1_score(y_test, y_pred))


# 混合行列のプロット
def plot_confusion_matrix(
    y_test: pd.Series,
    y_pred: pd.Series,
    model_name: str = "confusion matrix",
    display_details: bool = False,
):
    cm = confusion_matrix(y_test, y_pred, normalize="all", labels=[True, False])
    df_cm = pd.DataFrame(data=cm, index=[True, False], columns=[True, False])

    fig = plt.figure()
    sns.heatmap(df_cm, square=True, cbar=True, annot=True, cmap="Blues")
    plt.title(model_name)
    plt.xlabel("Predict label")
    plt.ylabel("True label")
    plt.plot()

    if display_details:
        tn, fp, fn, tp = cm.ravel()
        print("tn: ", tn, "\nfp: ", fp, "\nfn:", fn, "\ntp:", tp)
        show_scores(y_test, y_pred)


# 標準化を行う関数
def transform_std(X_train: pd.DataFrame(), X_test: pd.DataFrame() = None):
    std = StandardScaler()
    std.fit(X_train)
    X_train_std = pd.DataFrame(
        std.transform(X_train), index=X_train.index, columns=X_train.columns
    )
    if X_test is None:
        return X_train_std
    X_test_std = pd.DataFrame(
        std.transform(X_test), index=X_test.index, columns=X_test.columns
    )
    return X_train_std, X_test_std


# 正規化を行う関数
def transform_norm(
    X_train: pd.DataFrame(), X_test: pd.DataFrame() = None
) -> pd.DataFrame():
    mm = MinMaxScaler()
    mm.fit(X_train)
    X_train_norm = pd.DataFrame(
        mm.transform(X_train), index=X_train.index, columns=X_train.columns
    )
    if X_test is None:
        return X_train_norm
    X_test_norm = pd.DataFrame(
        mm.transform(X_test), index=X_test.index, columns=X_test.columns
    )
    return X_train_norm, X_test_norm


# GenericUnivariateSelectについて
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.GenericUnivariateSelect.html#sklearn.feature_selection.GenericUnivariateSelect
def feature_selection(
    X: pd.DataFrame, y: pd.Series, feature_selecton_function: callable, mode: str, param
) -> pd.DataFrame:
    selector = GenericUnivariateSelect(
        feature_selecton_function, mode=mode, param=param
    )
    # 特徴量選択の実施（fit）
    selector.fit(X, y)
    selector.transform(X),
    # 返り値のためのdf作成
    df_result = pd.DataFrame(
        selector.get_support(),
        index=X.columns.values,
        columns=["False: dropped"],
    )
    df_result["score"] = selector.scores_
    df_result["pvalue"] = selector.pvalues_
    return df_result

#-----------------------------------------------------------------------------
def plot_learning_curve(
    estimator,
    title,
    X,
    y,
    axes=None,
    ylim=None,
    cv=None,
    n_jobs=None,
    scoring=None,
    train_sizes=np.linspace(0.1, 1.0, 5),
):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    scoring : str or callable, default=None
        A str (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
        axes[1].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Accuracy score")
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("f1 score")

    # fit_times: Times spent for fitting in seconds
    train_sizes, train_scores, test_scores,  = learning_curve(
        estimator,
        X,
        y,
        scoring='accuracy',
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    train_sizes, train_scores, test_scores = learning_curve(
        estimator,
        X,
        y,
        scoring='f1',
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
    )
    train_f1_mean = np.mean(train_scores, axis=1)
    train_f1_std = np.std(train_scores, axis=1)
    test_f1_mean = np.mean(test_scores, axis=1)
    test_f1_std = np.std(test_scores, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes[0].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes[0].plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    axes[0].plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    axes[0].legend(loc="best")

    # Plot learning curve (f1)
    axes[1].grid()
    axes[1].fill_between(
        train_sizes,
        train_f1_mean - train_f1_std,
        train_f1_mean + train_f1_std,
        alpha=0.1,
        color="r",
    )
    axes[1].fill_between(
        train_sizes,
        test_f1_mean - test_f1_std,
        test_f1_mean + test_f1_std,
        alpha=0.1,
        color="g",
    )
    axes[1].plot(
        train_sizes, train_f1_mean, "o-", color="r", label="Training score (f1)"
    )
    axes[1].plot(
        train_sizes, test_f1_mean, "o-", color="g", label="Cross-validation score (f1)"
    )
    axes[1].legend(loc="best")

    return plt


# 2値分類モデル（Binary Classification Model）の性能を比較する関数
# 比較するbcmはconfig.py参照
# 評価指標はaccuracyとf1
def compare_bcms(
    X: pd.DataFrame(),
    y: pd.Series(),
    bcm_names: list = bcm_names,
    classifiers: list = classifiers,
    sort_column_name: str = "f1_test",
    folds: int = 10,
    test_size: float = 0.25,
    over_sampling_class: callable=None,
    # 標準化・正規化の実行の有無、及びそれを適用するcolumns
    normalization: bool = False,
    standardization: bool = False,
    converted_columns: list() = None,
    plot_cfmatrix: bool = False,
    save_path:str=None,
):
    warnings.filterwarnings("ignore")  # lrで警告が出て視認性が悪いので、いったん非表示
    result = []

    for name, clf in tqdm(zip(bcm_names, classifiers)):  # 指定した複数の分類機を順番に呼び出す
        # print(name)  # モデル名
        # k分割交差検証の実施
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            # print("initsize: ", X_train.shape)
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # 標準化の処理
            if standardization:
                # 特定のカラムへの適用
                if converted_columns:
                    (
                        X_train[converted_columns],
                        X_test[converted_columns],
                    ) = transform_std(
                        X_train[converted_columns], X_test[converted_columns]
                    )
                # df全体への適用
                else:
                    X_train, X_test = transform_std(X_train, X_test)
            
            # 正規化の処理
            if normalization:
                # 特定のカラムへの適用
                if converted_columns:
                    (
                        X_train[converted_columns],
                        X_test[converted_columns],
                    ) = transform_norm(
                        X_train[converted_columns], X_test[converted_columns]
                    )
                # df全体への適用
                else:
                    X_train, X_test = transform_std(X_train, X_test)
            
            # オーバーサンプリング（trainデータのみに適用し、testデータには適用しない）
            if over_sampling_class:
                X_train, y_train = over_sampling_class.fit_resample(X_train, y_train)
            # print("over sampling size: ", X_train.shape)

            # 訓練のスコア
            clf.fit(X_train, y_train)  # 学習
            y_pred_train = clf.predict(X_train)
            acc_train = accuracy_score(y_train, y_pred_train)
            f1_train = f1_score(y_train, y_pred_train)
            #  予測値のスコア
            y_pred = clf.predict(X_test)
            acc_test = accuracy_score(y_test, y_pred)  # 正解率（test）の算出
            f1_test = f1_score(y_test, y_pred)
            result.append([name, acc_train, acc_test, f1_train, f1_test])  # 結果の格納
        # 混合行列の表示
        if plot_cfmatrix:
            plot_confusion_matrix(y_test, y_pred)
            

    # 表示設定
    df_result = pd.DataFrame(
        result, columns=["classifier", "acc_train", "acc_test", "f1_train", "f1_test"]
    )
    df_result_mean = (
        df_result.groupby("classifier")
        .mean()
        .sort_values(sort_column_name, ascending=False)
    )
    warnings.filterwarnings("always")
    
    # 保存設定
    if save_path:
        df_result_mean.to_csv(save_path)
        
    return df_result_mean


