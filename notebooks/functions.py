import os
import pickle
import dill

import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

# 評価指標
from sklearn.model_selection import learning_curve
from sklearn.metrics import (
    precision_score,
    recall_score,
)
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    roc_auc_score,
    matthews_corrcoef,
    cohen_kappa_score,
    f1_score
)
from sklearn.metrics import confusion_matrix

from config import SEED

# -----------------------------------------------------------------------------
# basic function
def make_dir(dir_name: str):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


# seedの固定
def fix_seed(seed: int):
    # Numpy
    np.random.seed(seed)


# pickleオブジェクトにして保存
def pickle_dump(obj, path):
    with open(path, mode="wb") as f:
        pickle.dump(obj, f)


def pickle_load(path):
    with open(path, mode="rb") as f:
        obj = pickle.load(f)
        return obj

# dillオブジェクトにして保存
def dill_dump(obj, path):
    with open(path, mode="wb") as f:
        dill.dump(obj, f)

def dill_load(path):
    with open(path, mode="rb") as f:
        obj = dill.load(f)
        return obj

# -----------------------------------------------------------------------------
# df status function
def check(df):
    col_list = df.columns.values  # 列名を取得
    row = []
    for col in col_list:
        max_value = None
        min_value = None
        unique = ""
        value_counts = ""
        # 最大・最小の数値を取得
        if df[col].dtype == int or df[col].dtype == float:
            max_value = df[col].max()
            min_value = df[col].min()
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


# -----------------------------------------------------------------------------
# データの読み込み
# 前処理後、各学習の前段階のデータを読み込む
def exists_data_files(file_path: str, file_name: str):
    data_phases = ["train", "val"]

    for data_phase in data_phases:
        X_path = "{0}/{1}/X_{2}.pkl".format(file_path, data_phase, file_name)
        y_path = "{0}/{1}/y_{2}.pkl".format(file_path, data_phase, file_name)
        if not (os.path.exists(X_path) and os.path.exists(y_path)):
            print("pkl file does not exist")
            return False
    return True


def read_preprocessed_df(
    file_path: str = ".",
    file_name: str = "sample",
):
    if exists_data_files(file_path, file_name):
        X_train = pd.read_pickle("{0}/train/X_{1}.pkl".format(file_path, file_name))
        y_train = pd.read_pickle("{0}/train/y_{1}.pkl".format(file_path, file_name))
        X_val = pd.read_pickle("{0}/val/X_{1}.pkl".format(file_path, file_name))
        y_val = pd.read_pickle("{0}/val/y_{1}.pkl".format(file_path, file_name))
        X_train_val = pd.read_pickle(
            "{0}/train_val/X_{1}.pkl".format(file_path, file_name)
        )
        y_train_val = pd.read_pickle(
            "{0}/train_val/y_{1}.pkl".format(file_path, file_name)
        )
        X_test = pd.read_pickle("{0}/test/X_{1}.pkl".format(file_path, file_name))
        y_test = pd.read_pickle("{0}/test/y_{1}.pkl".format(file_path, file_name))
        list_train = [X_train, y_train]
        list_val = [X_val, y_val]
        list_train_val = [X_train_val, y_train_val]
        list_test = [X_test, y_test]
        return list_train, list_val, list_train_val, list_test


# -----------------------------------------------------------------------------
# learning function
# 基本的なスコアの表示（面倒なので関数化した）
def show_scores(y_true: pd.Series, y_pred: pd.Series, save_path: str = None):
    index = ["accuracy", "log_loss", "roc_auc_score", 'matthews_corrcoef']
    data = [
        accuracy_score(y_true, y_pred),
        log_loss(y_true, y_pred),
        roc_auc_score(y_true, y_pred),
        matthews_corrcoef(y_true, y_pred),
    ]
    series = pd.Series(data, index=index)
    display(series)
    if save_path:
        series.to_csv(save_path)


# 混合行列のプロット
def plot_confusion_matrix(
    y_test: pd.Series,
    y_pred: pd.Series,
    model_name: str = "confusion matrix",
    display_details: bool = False,
):
    cm = confusion_matrix(y_test, y_pred, labels=[True, False])
    df_cm = pd.DataFrame(data=cm, index=[True, False], columns=[True, False])

    fig = plt.figure()
    plt.rcParams["font.size"] = 18
    sns.heatmap(df_cm, square=True, cbar=True, annot=True, cmap="Blues")
    plt.title(model_name)
    plt.xlabel("Predict label")
    plt.ylabel("True label")
    plt.plot()

    if display_details:
        tn, fp, fn, tp = cm.ravel()
        print("tn: ", tn, "\nfp: ", fp, "\nfn:", fn, "\ntp:", tp)
        show_scores(y_test, y_pred)


# 2値分類モデル（Binary Classification Model）の性能を比較する関数
# 比較するbcmはconfig.py参照
def compare_bcms(
    X_train: pd.DataFrame(),
    y_train: pd.Series(),
    X_val: pd.DataFrame(),
    y_val: pd.Series(),
    classifiers: list = None,
    plot: bool = False,
    save_path: str = None,
):
    warnings.filterwarnings("ignore")  # lrで警告が出て視認性が悪いので、いったん非表示
    result = []

    for clf in classifiers:  # 指定した複数の分類機を順番に呼び出す
        # 訓練のスコア
        clf.fit(X_train, y_train)  # 学習
        y_train_pred = clf.predict(X_train)
        acc_train = accuracy_score(y_train, y_train_pred)
        f1_train = f1_score(y_train, y_train_pred)
        # マシューズ相関係数について
        # 範囲：[-1, 1]で絶対値が大きいほどよい
        # https://www.datarobot.com/jp/blog/matthews-correlation-coefficient/
        mcc_train = matthews_corrcoef(y_train, y_train_pred)
        ckappa_train = cohen_kappa_score(y_train, y_train_pred)
        auc_train = roc_auc_score(y_train, y_train_pred)
        # 予測値のスコア
        y_val_pred = clf.predict(X_val)
        acc_val = accuracy_score(y_val, y_val_pred)  # 正解率（test）の算出
        f1_val = f1_score(y_val, y_val_pred)
        mcc_val = matthews_corrcoef(y_val, y_val_pred)
        ckappa_val = cohen_kappa_score(y_val, y_val_pred)
        auc_val = roc_auc_score(y_val, y_val_pred)

        result.append(
            [
                clf.__class__.__name__,
                acc_train,
                acc_val,
                f1_train,
                f1_val,
                mcc_train,
                mcc_val,
                ckappa_train,
                ckappa_val,
                auc_train,
                auc_val,
            ]
        )  # 結果の格納

        # 混合行列の表示
        if plot:
            plot_confusion_matrix(
                y_test=y_train,
                y_pred=y_train_pred,
                model_name="train_{0}".format(clf.__class__.__name__),
            )
            plot_confusion_matrix(
                y_test=y_val,
                y_pred=y_val_pred,
                model_name="val_{0}".format(clf.__class__.__name__),
            )
    # 表示設定
    df_result = pd.DataFrame(
        result,
        columns=[
            "classifier",
            "{0}_train".format(accuracy_score.__name__),
            "{0}_val".format(accuracy_score.__name__),
            "{0}_train".format(f1_score.__name__),
            "{0}_val".format(f1_score.__name__),
            "{0}_train".format(matthews_corrcoef.__name__),
            "{0}_val".format(matthews_corrcoef.__name__),
            "{0}_train".format(cohen_kappa_score.__name__),
            "{0}_val".format(cohen_kappa_score.__name__),
            "{0}_train".format(roc_auc_score.__name__),
            "{0}_val".format(roc_auc_score.__name__),
        ],
    )
    df_result_mean = df_result.groupby("classifier").mean()
    # 保存設定
    if save_path:
        df_result_mean.to_csv(save_path)
    return df_result_mean