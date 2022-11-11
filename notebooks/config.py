# models
from sklearn.linear_model import LogisticRegression  # ロジスティック回帰
from sklearn.neighbors import KNeighborsClassifier  # K近傍法
from sklearn.svm import SVC  # サポートベクターマシン
from sklearn.tree import DecisionTreeClassifier, export_graphviz  # 決定木
from sklearn.ensemble import RandomForestClassifier  # ランダムフォレスト
from sklearn.ensemble import AdaBoostClassifier  # AdaBoost
from sklearn.naive_bayes import GaussianNB  # ナイーブ・ベイズ
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA  # 二次判別分析
from sklearn.ensemble import ExtraTreesClassifier
from lightgbm import LGBMClassifier

# ######################################################################################################
# データを保存するための最上位パス
DATA_DIR = "../data"
# ======================================================================================================
## 本番環境の前処理データ
PROCESSED_DIR = DATA_DIR + "/processed"
# ======================================================================================================
## 外部データの保存先の最上位パス
EXTERNAL_DIR = DATA_DIR + "/external"
### metabricの臨床データを保存
EXTERNAl_BRCA_METABRIC_DATA_CLNICAL_DIR = EXTERNAL_DIR + "/brca_metabric_data_clinical"
# ======================================================================================================
## 生データ
RAW_DIR = DATA_DIR + "/raw"
# ......................................................................................................
### metabricの生データ
RAW_BRCA_METABRIC_DIR = RAW_DIR + "/brca_metabric"
### metabricの.gzファイル
RAW_BRCA_METABRIC_TG = RAW_DIR + "/brca_metabric.tar.gz"
# ======================================================================================================
## 加工したデータ保存先の最上位パス
INTERIM_DIR = DATA_DIR + "/interim"
### 1-EDAのデータ保存先の最上位パス
INTERIM_EDA_DIR = INTERIM_DIR + "/EDA"
# ......................................................................................................
### 2-preprocessのデータ保存先の最上位パス
INTERIM_PREPROCESSED_DIR = INTERIM_DIR + "/PREPROCESSED"
#### 再発の保存先のパス
INTERIM_PREPROCESSED_RECURRENCE_DIR = INTERIM_PREPROCESSED_DIR + "/RECURRENCE"
# ......................................................................................................
### 3-create_bcmのデータ保存先の最上位パス
INTERIM_MODELS_DIR = INTERIM_DIR + "/MODELS"
INTERIM_MODELS_RECURRENCE_DIR = INTERIM_MODELS_DIR + "/RECURRENCE"
# ......................................................................................................
### 4-tuningのデータ保存先の最上位パス
INTERIM_TUNING_DIR = INTERIM_DIR + "/TUNING"
#### 再発の保存先のパス
INTERIM_TUNING_RECURRENCE_DIR = INTERIM_TUNING_DIR + "/RECURRENCE"
### その他のデータを保存するためのパス
INTERIM_OTHERS_DIR = INTERIM_DIR + "/OTHERS"
# ######################################################################################################


# ######################################################################################################
# 学習したモデルを保存するためのディレクトリパス
MODELS = "../models"
# ------------------------------------------------------------------------------------------------------
## ノートブックで学習したモデル
MODELS_NOTEBOOK = MODELS + "/notebooks"
# ######################################################################################################


# ######################################################################################################
# 分析結果の最上位パス
REPORT_DIR = "../reports"
# ======================================================================================================
## 画像の保存先の最上位パス
FIGURES_DIR = REPORT_DIR + "/FIGURES"
# ......................................................................................................
### 1-EDAで生成された画像保存先の最上位パス
FIGURES_EDA_DIR = FIGURES_DIR + "/EDA"
#### 再発の保存先のパス
FIGURES_RECURRENCE_DIR = FIGURES_EDA_DIR + "/RECURRENCE"
# ......................................................................................................
### 2-preprocessで生成された画像保存先の最上位パス
FIGURES_PREPROCESS_DIE = FIGURES_DIR + "/PREPROCESS"
#### 再発の保存先のパス
FIGURES_PREPROCESS_RECURRENCE_DIR = FIGURES_PREPROCESS_DIE + "/RECURRENCE"
# ......................................................................................................
### 3-create_bcmで生成された画像保存先の最上位パス
FIGURES_MODELS_DIR = FIGURES_DIR + "/MODELS"
#### 再発の保存先のパス
FIGURES_MODELS_RECURRENCE_DIR = FIGURES_MODELS_DIR + "/RECURRENCE"
# ......................................................................................................
### 4-tuningで生成された画像保存先の最上位パス
FIGURES_TUNING_DIR = FIGURES_DIR + "/TUNING"
#### 再発の保存先のパス
FIGURES_TUNING_RECURRENCE_DIR = FIGURES_TUNING_DIR + "/RECURRENCE"
# ......................................................................................................
### 5-explainで生成された画像保存先の最上位パス
FIGURES_EXPLAIN_DIR = FIGURES_DIR + "/EXPLAIN"
#### 再発のモデル作成で生成れた
FIGURES_EXPLAIN_RECURRENCE_DIR = FIGURES_EXPLAIN_DIR + "/RECURRENCE"
# ......................................................................................................
### 6-lifelinesで生成された画像保存先の最上位パス
FIGURES_LIFELINES_DIR = FIGURES_DIR + "/LIFELINES"
#### 再発の保存先のパス
FIGURES_LIFELINES_RECURRENCE_DIR = FIGURES_LIFELINES_DIR + "/RECURRENCE"
# ......................................................................................................
### その他の画像保存先パス
FIGURES_OTHERS_DIR = FIGURES_DIR + "/OTHERS"
# ======================================================================================================
## 表の保存先の最上位パス
FIGURES_DIR = REPORT_DIR + "/FIGURES"
TABLES_DIR = REPORT_DIR + "/TABLES"
# ......................................................................................................
### 3-create_bcmで生成された表保存先の最上位パス
TABLES_MODELS_DIR = TABLES_DIR + "/MODELS"
#### 再発の保存先のパス
TABLES_MODELS_RECURRENCE_DIR = TABLES_MODELS_DIR + "/RECURRENCE"
# ......................................................................................................
###  5-explainで生成された表保存先の最上位パス
TABLES_EXPLAIN_DIR = TABLES_DIR + "/EXPLAIN"
#### 再発の保存先のパス
TABLES_EXPLAIN_RECURRENCE_DIR = TABLES_EXPLAIN_DIR + "/RECURRENCE"
# ######################################################################################################


# ######################################################################################################
# URL
URL_cBioPortal = "https://cbioportal-datahub.s3.amazonaws.com/brca_metabric.tar.gz"
# ######################################################################################################


# ######################################################################################################
# SEED
SEED = 100
THRESHOLD_YEARS = 5
THRESHOLD_MONTHS = THRESHOLD_YEARS * 12

classifiers = [
    LogisticRegression(max_iter=2000, random_state=SEED),
    KNeighborsClassifier(),
    SVC(
        kernel="linear",
        random_state=SEED,
        class_weight="balanced",
    ),
    SVC(kernel="poly", random_state=SEED, class_weight="balanced"),
    SVC(kernel="rbf", random_state=SEED, class_weight="balanced"),
    SVC(kernel="sigmoid", random_state=SEED, class_weight="balanced"),
    DecisionTreeClassifier(
        min_samples_split=20,
        min_samples_leaf=15,
        random_state=SEED,
        class_weight="balanced",
    ),
    RandomForestClassifier(
        min_samples_split=20,
        min_samples_leaf=15,
        random_state=SEED,
        class_weight="balanced",
    ),
    AdaBoostClassifier(random_state=SEED),
    GaussianNB(),
    GradientBoostingClassifier(random_state=SEED),
    SGDClassifier(random_state=SEED, class_weight="balanced"),
    QDA(),
    LGBMClassifier(class_weight="balanced", random_state=SEED),
    ExtraTreesClassifier(class_weight="balanced", random_state=SEED),
]
# ######################################################################################################