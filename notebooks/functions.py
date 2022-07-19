import os
import random

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
from sklearn.model_selection import KFold

# 前処理
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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


def make_dir(dir_name: str):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


# seedの固定
def fix_seed(seed: int):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)


def check(df):
    col_list = df.columns.values  # 列名を取得
    row = []
    for col in col_list:
        unique = ""
        value_counts = ""
        if df[col].nunique() < 12:
            unique = df[col].unique()
            value_counts = df[col].value_counts().to_dict()
        tmp = (
            col,  # 列名
            df[col].dtypes,  # データタイプ
            df[col].isnull().sum(),  # null数
            df[col].count(),  # データ数 (欠損値除く)
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
        "num_unique",
        "unique",
        "unique_counts",
    ]  # データフレームの列名指定
    # unique_countsの中身確認のために横幅拡張
    d = dict(selector=".col8", props=[("min-width", "200px")])  # name
    return df.style.set_table_styles([d])


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


def plot_learning_curve(
    X: pd.DataFrame(),
    y: pd.DataFrame(),
    estimator: callable,
    cv: int = 10,
    train_sizes: np.arange = np.arange(100, 1001, 100),
):
    # cvにintを渡すと k-foldの「k」を指定できる
    # ↓では3にしているので、3-fold法を使用する。
    train_sizes, train_scores, test_scores = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        train_sizes=train_sizes,
        random_state=SEED,
        shuffle=True,
    )

    """
    print("train_sizes(検証したサンプル数): {}".format(train_sizes))
    print("------------")
    print("train_scores(各サンプル数でのトレーニングスコア): \n{}".format(train_scores))
    print("------------")
    print("test_scores(各サンプル数でのバリデーションスコア): \n{}".format(test_scores))
    """

    # 学習の様子をプロット
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.title("Learning Curve")
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    # Traing score と Test score をプロット
    plt.plot(train_sizes, train_scores_mean, "o-", color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, "o-", color="g", label="Test score")

    # 標準偏差の範囲を色付け
    plt.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        color="r",
        alpha=0.2,
    )
    plt.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        color="g",
        alpha=0.2,
    )

    plt.ylim(0.5, 1.0)
    plt.legend(loc="best")

    plt.show()


def compare_bcms(
    X: pd.DataFrame(),
    y: pd.Series(),
    bcm_names: list = bcm_names,
    classifiers: list = classifiers,
    sort_column_name: str = "f1_test",
    folds: int = 10,
    test_size: float = 0.25,
    over_sampling_class=None,
    # 標準化・正規化の実行の有無、及びそれを適用するcolumns
    normalization: bool = False,
    standardization: bool = False,
    converted_columns: list() = None,
    plot_cfmatrix: bool = False,
):
    warnings.filterwarnings("ignore")  # lrで警告が出て視認性が悪いので、いったん非表示
    result = []

    for name, clf in tqdm(zip(bcm_names, classifiers)):  # 指定した複数の分類機を順番に呼び出す
        # print(name)  # モデル名
        # k分割交差検証の実施
        kf = KFold(n_splits=folds, shuffle=True, random_state=SEED)
        for train_index, test_index in kf.split(X, y):
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
    return df_result_mean
