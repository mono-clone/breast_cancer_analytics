# 機械学習モデルインスタンスの定義のためのインポート
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

# ======================================================================================================
# データを保存するための最上位パス
DATA_DIR = "../data"
# ------------------------------------------------------------------------------------------------------
## 外部データの保存先の最上位パス
EXTERNAL_DIR = DATA_DIR + "/external"
### metabricの臨床データを保存
EXTERNAl_BRCA_METABRIC_DATA_CLNICAL_DIR = EXTERNAL_DIR + "/brca_metabric_data_clinical"
# ------------------------------------------------------------------------------------------------------
## 加工したデータ保存先の最上位パス
INTERIM_DIR = DATA_DIR + "/interim"
# ======================================================================================================
### 1.X.X-EDAのデータ保存先の最上位パス
INTERIM_EDA_DIR = INTERIM_DIR + "/EDA"
# ......................................................................................................
### 2.X.X-preprocessのデータ保存先の最上位パス
INTERIM_PREPROCESSED_DIR = INTERIM_DIR + "/PREPROCESSED"
#### 生存期間の保存先のパス
INTERIM_PREPROCESSED_PROGNOSIS_DIR = (
    INTERIM_PREPROCESSED_DIR + "/PROGNOSIS"
)
##### 臨床・遺伝子・混合データの保存先のパス
INTERIM_PREPROCESSED_PROGNOSIS_CLINICAL_DIR = (
    INTERIM_PREPROCESSED_PROGNOSIS_DIR + "/CLINICAL"
)
INTERIM_PREPROCESSED_PROGNOSIS_GENES_DIR = (
    INTERIM_PREPROCESSED_PROGNOSIS_DIR + "/GENES"
)
INTERIM_PREPROCESSED_PROGNOSIS_CROSS_DIR = (
    INTERIM_PREPROCESSED_PROGNOSIS_DIR + "/CROSS"
)

#### 再発の保存先のパス
INTERIM_PREPROCESSED_RECURRENCE_DIR = INTERIM_PREPROCESSED_DIR + "/RECURRENCE"
##### 混合データの保存先のパス
INTERIM_PREPROCESSED_RECURRENCE_CROSS_DIR = (
    INTERIM_PREPROCESSED_RECURRENCE_DIR + "/CROSS"
)
# ......................................................................................................
### 3.X.X-create_bcmのデータ保存先の最上位パス
INTERIM_MODELS_DIR = INTERIM_DIR + "/MODELS"
INTERIM_MODELS_RECURRENCE_DIR = INTERIM_MODELS_DIR + "/RECURRENCE"
INTERIM_MODELS_RECURRENCE_CROSS_DIR = INTERIM_MODELS_RECURRENCE_DIR + "/CROSS"
# ......................................................................................................
### 4.X.X-tuningのデータ保存先の最上位パス
INTERIM_TUNING_DIR = INTERIM_DIR + "/TUNING"
#### 生存期間の保存先のパス
INTERIM_TUNING_PROGNOSIS_DIR = (
    INTERIM_TUNING_DIR + "/PROGNOSIS"
)
##### 混合データの保存先のパス
INTERIM_TUNING_PROGNOSIS_CROSS_DIR = (
    INTERIM_TUNING_PROGNOSIS_DIR + "/CROSS"
)
#### 再発の保存先のパス
INTERIM_TUNING_RECURRENCE_DIR = (
    INTERIM_TUNING_DIR + "/RECURRENCE"
)
##### 混合データの保存先のパス
INTERIM_TUNING_RECURRENCE_CROSS_DIR = (
    INTERIM_TUNING_RECURRENCE_DIR + "/CROSS"
)
# ======================================================================================================
### その他のデータを保存するためのパス
INTERIM_OTHERS_DIR = INTERIM_DIR + "/OTHERS"
# ------------------------------------------------------------------------------------------------------
## 本番環境の前処理データ
PROCESSED_DIR = DATA_DIR + "/processed"
# ------------------------------------------------------------------------------------------------------
## 生データ
RAW_DIR = DATA_DIR + "/raw"
# ......................................................................................................
### metabricの生データ
RAW_BRCA_METABRIC_DIR = RAW_DIR + "/brca_metabric"
### metabricの.gzファイル
RAW_BRCA_METABRIC_TG = RAW_DIR + "/brca_metabric.tar.gz"
# ======================================================================================================


# ======================================================================================================
# 学習したモデルを保存するためのディレクトリパス
MODELS = "../models"
# ------------------------------------------------------------------------------------------------------
## ノートブックで学習したモデル
MODELS_NOTEBOOK = MODELS + "/notebooks"
# ======================================================================================================


# ======================================================================================================
# 分析結果の最上位パス
REPORT_DIR = "../reports"
# ------------------------------------------------------------------------------------------------------
## 画像の保存先の最上位パス
FIGURES_DIR = REPORT_DIR + "/FIGURES"
# ......................................................................................................
### 1.X.X-EDAで生成された画像保存先の最上位パス
FIGURES_EDA_DIR = FIGURES_DIR + "/EDA"
#### 生存期間の保存先のパス
FIGURES_PROGNOSIS_DIR = FIGURES_EDA_DIR + "/PROGNOSIS"
##### 臨床・遺伝子・混合データの保存先のパス
FIGURES_PROGNOSIS_CLINICAL_DIR = FIGURES_PROGNOSIS_DIR + "/CLINICAL"
FIGURES_PROGNOSIS_GENES_DIR = FIGURES_PROGNOSIS_DIR + "/GENES"
FIGURES_PROGNOSIS_CROSS_DIR = FIGURES_PROGNOSIS_DIR + "/CROSS"
#### 再発の保存先のパス
FIGURES_RECURRENCE_DIR = FIGURES_EDA_DIR + "/RECURRENCE"
###### 混合データのの保存先のパス
FIGURES_RECURRENCE_CROSS_DIR = FIGURES_RECURRENCE_DIR + "/CROSS"

# ......................................................................................................
### 2.X.X-preprocessで生成された画像保存先の最上位パス
FIGURES_PREPROCESS_DIE = FIGURES_DIR + "/PREPROCESS"
#### 生存期間の保存先のパス
FIGURES_PREPROCESS_PROGNOSIS_DIR = FIGURES_PREPROCESS_DIE + "/PROGNOSIS"
##### 臨床・遺伝子・混合データの保存先のパス
FIGURES_PREPROCESS_PROGNOSIS_CLINICAL_DIR = (
    FIGURES_PREPROCESS_PROGNOSIS_DIR + "/CLINICAL"
)
FIGURES_PREPROCESS_PROGNOSIS_GENES_DIR = FIGURES_PREPROCESS_PROGNOSIS_DIR + "/GENES"
FIGURES_PREPROCESS_PROGNOSIS_CROSS_DIR = FIGURES_PREPROCESS_PROGNOSIS_DIR + "/CROSS"
#### 再発の保存先のパス
FIGURES_PREPROCESS_RECURRENCE_DIR = FIGURES_PREPROCESS_DIE + "/RECURRENCE"
###### 混合データの保存先のパス
FIGURES_PREPROCESS_RECURRENCE_CROSS_DIR = FIGURES_PREPROCESS_RECURRENCE_DIR + "/CROSS"
# ......................................................................................................
### 3.X.X-create_bcmで生成された画像保存先の最上位パス
FIGURES_MODELS_DIR = FIGURES_DIR + "/MODELS"
#### 生存期間の保存先のパス
FIGURES_MODELS_PROGNOSIS_DIR = FIGURES_MODELS_DIR + "/PROGNOSIS"
##### 臨床・遺伝子・混合データの保存先のパス
FIGURES_MODELS_PROGNOSIS_CLINICAL_DIR = (
    FIGURES_MODELS_PROGNOSIS_DIR + "/CLINICAL"
)
FIGURES_MODELS_PROGNOSIS_GENES_DIR = (
    FIGURES_MODELS_PROGNOSIS_DIR + "/GENES"
)
FIGURES_MODELS_PROGNOSIS_CROSS_DIR = (
    FIGURES_MODELS_PROGNOSIS_DIR + "/CROSS"
)
#### 再発の保存先のパス
FIGURES_MODELS_RECURRENCE_DIR = FIGURES_MODELS_DIR + "/RECURRENCE"
##### 混合データの保存先のパス
FIGURES_MODELS_RECURRENCE_CROSS_DIR = (
    FIGURES_MODELS_RECURRENCE_DIR + "/CROSS"
)
# ......................................................................................................
### 4.X.X-tuningで生成された画像保存先の最上位パス
FIGURES_TUNING_DIR = FIGURES_DIR + "/TUNING"
#### 生存期間の保存先のパス
FIGURES_TUNING_PROGNOSIS_DIR = FIGURES_TUNING_DIR + "/PROGNOSIS"
##### 混合データの保存先のパス
FIGURES_TUNING_PROGNOSIS_CROSS_DIR = (
    FIGURES_TUNING_PROGNOSIS_DIR + "/CROSS"
)
#### 再発の保存先のパス
FIGURES_TUNING_RECURRENCE_DIR = FIGURES_TUNING_DIR + "/RECURRENCE"
##### 混合データの保存先のパス
FIGURES_TUNING_RECURRENCE_CROSS_DIR = (
    FIGURES_TUNING_RECURRENCE_DIR + "/CROSS"
)
# ......................................................................................................
### 5.X.X-explainで生成された画像保存先の最上位パス
FIGURES_EXPLAIN_DIR = FIGURES_DIR + "/EXPLAIN"
#### 生存期間の保存先のパス
FIGURES_EXPLAIN_PROGNOSIS_DIR = FIGURES_EXPLAIN_DIR + "/PROGNOSIS"
##### 混合データの保存先のパス
FIGURES_EXPLAIN_PROGNOSIS_CROSS_DIR = (
    FIGURES_EXPLAIN_PROGNOSIS_DIR + "/CROSS"
)
#### 再発のモデル作成で生成れた
FIGURES_EXPLAIN_RECURRENCE_DIR = FIGURES_EXPLAIN_DIR + "/RECURRENCE"
##### 混合データの保存先のパス
FIGURES_EXPLAIN_RECURRENCE_CROSS_DIR = (
    FIGURES_EXPLAIN_RECURRENCE_DIR + "/CROSS"
)
# ......................................................................................................
### 6.X.X-lifelinesで生成された画像保存先の最上位パス
FIGURES_LIFELINES_DIR = FIGURES_DIR + "/LIFELINES"
#### 生存期間の保存先のパス
FIGURES_LIFELINES_PROGNOSIS_DIR = FIGURES_LIFELINES_DIR + "/PROGNOSIS"
##### 混合データの保存先のパス
FIGURES_LIFELINES_PROGNOSIS_CROSS_DIR = (
    FIGURES_LIFELINES_PROGNOSIS_DIR + "/CROSS"
)
#### 再発の保存先のパス
FIGURES_LIFELINES_RECURRENCE_DIR = FIGURES_LIFELINES_DIR + "/RECURRENCE"
##### 混合データの保存先のパス
FIGURES_LIFELINES_RECURRENCE_CROSS_DIR = (
    FIGURES_LIFELINES_RECURRENCE_DIR + "/CROSS"
)
# ......................................................................................................
### その他の画像保存先パス
FIGURES_OTHERS_DIR = FIGURES_DIR + "/OTHERS"
# ======================================================================================================
## 表の保存先の最上位パス
FIGURES_DIR = REPORT_DIR + "/FIGURES"
TABLES_DIR = REPORT_DIR + "/TABLES"
# ......................................................................................................
### 3.X.X-create_bcmで生成された表保存先の最上位パス
TABLES_MODELS_DIR = TABLES_DIR + "/MODELS"
#### 生存期間のモデル作成で生成れた表
TABLES_MODELS_PROGNOSIS_DIR = TABLES_MODELS_DIR + "/PROGNOSIS"
##### 臨床・遺伝子・混合データの保存先のパス
TABLES_MODELS_PROGNOSIS_CLINICAL_DIR = (
    TABLES_MODELS_PROGNOSIS_DIR + "/CLINICAL"
)
TABLES_MODELS_PROGNOSIS_GENES_DIR = (
    TABLES_MODELS_PROGNOSIS_DIR + "/GENES"
)
TABLES_MODELS_PROGNOSIS_CROSS_DIR = (
    TABLES_MODELS_PROGNOSIS_DIR + "/CROSS"
)
#### 再発の保存先のパス
TABLES_MODELS_RECURRENCE_DIR = TABLES_MODELS_DIR + "/RECURRENCE"
##### 混合データの保存先のパス
TABLES_MODELS_RECURRENCE_CROSS_DIR = (
    TABLES_MODELS_RECURRENCE_DIR + "/CROSS"
)
# ......................................................................................................
###  5.X.X-explainで生成された表保存先の最上位パス
TABLES_EXPLAIN_DIR = TABLES_DIR + "/EXPLAIN"
#### 生存期間のモデル作成で生成れた表
TABLES_EXPLAIN_PROGNOSIS_DIR = TABLES_EXPLAIN_DIR + "/PROGNOSIS"
##### 混合データの保存先のパス
TABLES_EXPLAIN_PROGNOSIS_CROSS_DIR = (
    TABLES_EXPLAIN_PROGNOSIS_DIR + "/CROSS"
)
#### 再発の保存先のパス
TABLES_EXPLAIN_RECURRENCE_DIR = TABLES_EXPLAIN_DIR + "/RECURRENCE"
##### 混合データの保存先のパス
TABLES_EXPLAIN_RECURRENCE_CROSS_DIR = (
    TABLES_EXPLAIN_RECURRENCE_DIR + "/CROSS"
)
# ======================================================================================================


# =====================================================================================================
# URL
URL_cBioPortal = "https://cbioportal-datahub.s3.amazonaws.com/brca_metabric.tar.gz"
# =====================================================================================================


# =====================================================================================================
# SEED
SEED = 100
# microarray_name
SET_NAME_MICROARRAY = (
    "mrna_agilent_microarray",
    "mrna_agilent_microarray_zscores_ref_all_samples",
    "mrna_agilent_microarray_zscores_ref_diploid_samples",
)
INDEX_MICROARRAY = 1
THRESHOLD_YEARS=5
THRESHOLD_MONTHS=THRESHOLD_YEARS*12
# =====================================================================================================


# =====================================================================================================
classifiers = [
    LogisticRegression(max_iter=2000, random_state=SEED),
    KNeighborsClassifier(),
    SVC(kernel="linear", random_state=SEED, class_weight= "balanced",),
    SVC(kernel="poly", random_state=SEED,class_weight= "balanced"),
    SVC(kernel="rbf", random_state=SEED,class_weight= "balanced"),
    SVC(kernel="sigmoid", random_state=SEED,class_weight= "balanced"),
    DecisionTreeClassifier(
        min_samples_split=20, min_samples_leaf=15, random_state=SEED,class_weight= "balanced"
    ),
    RandomForestClassifier(
        min_samples_split=20, min_samples_leaf=15, random_state=SEED,class_weight= "balanced"
    ),
    AdaBoostClassifier(random_state=SEED),
    GaussianNB(),
    GradientBoostingClassifier(random_state=SEED),
    SGDClassifier(random_state=SEED, class_weight= "balanced"),
    QDA(),
]
# =====================================================================================================
