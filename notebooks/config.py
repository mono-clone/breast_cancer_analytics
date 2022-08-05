# 機械学習モデルインスタンスの定義のためのインポート
from sklearn.linear_model import LogisticRegression  # ロジスティック回帰
from sklearn.neighbors import KNeighborsClassifier  # K近傍法
from sklearn.svm import SVC  # サポートベクターマシン
from sklearn.tree import DecisionTreeClassifier, export_graphviz  # 決定木
from sklearn.ensemble import RandomForestClassifier  # ランダムフォレスト
from sklearn.ensemble import AdaBoostClassifier  # AdaBoost
from sklearn.naive_bayes import GaussianNB  # ナイーブ・ベイズ
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA  # 二次判別分析

# ======================================================================================================
# データを保存するためのディレクトリパス
DATA_DIR = "../data"
# ------------------------------------------------------------------------------------------------------
## 外部データ
EXTERNAL_DIR = DATA_DIR + "/external"
### metabricの臨床データ
EXTERNAl_BRCA_METABRIC_DATA_CLNICAL_DIR = EXTERNAL_DIR + "/brca_metabric_data_clinical"
# ------------------------------------------------------------------------------------------------------
## 加工したデータ
INTERIM_DIR = DATA_DIR + "/interim"
# ......................................................................................................
### pickle形式の加工したデータ
INTERIM_PICKLE_DIR = INTERIM_DIR + "/pickle_data"
# ......................................................................................................
#### EDAのpickleデータ
INTERIM_PICKLE_EDA_DIR = INTERIM_PICKLE_DIR + "/EDA"
##### 臨床・遺伝子のEDAのpickleデータ
INTERIM_PICKLE_EDA_CLINICAL_DIR = INTERIM_PICKLE_EDA_DIR + "/clinical"
INTERIM_PICKLE_EDA_GENES_DIR = INTERIM_PICKLE_EDA_DIR + "/genes"
# ......................................................................................................
### preprocessのpickleデータ
INTERIM_PICKLE_PREPROCESSED_DIR = INTERIM_PICKLE_DIR + "/preprocessed"
####　予後の予測のpreprocessのpickleデータ
INTERIM_PICKLE_PREPROCESSED_PROGNOSIS_DIR = (
    INTERIM_PICKLE_PREPROCESSED_DIR + "/PROGNOSIS"
)
##### 臨床・遺伝子の予後の予測のpreprocessのpickleデータ
INTERIM_PICKLE_PREPROCESSED_PROGNOSIS_CLINICAL_DIR = (
    INTERIM_PICKLE_PREPROCESSED_PROGNOSIS_DIR + "/CLINICAL"
)
INTERIM_PICKLE_PREPROCESSED_PROGNOSIS_GENES_DIR = (
    INTERIM_PICKLE_PREPROCESSED_PROGNOSIS_DIR + "/GENES"
)
INTERIM_PICKLE_PREPROCESSED_PROGNOSIS_CROSS_DIR = (
    INTERIM_PICKLE_PREPROCESSED_PROGNOSIS_DIR + "/CROSS"
)

#### 再発の予測のpreprocessのpickleデータ
INTERIM_PICKLE_PREPROCESSED_RFS_DIR = INTERIM_PICKLE_PREPROCESSED_DIR + "/RFS"
##### 臨床・遺伝子の再発の予測のpreprocessのpickleデータ
INTERIM_PICKLE_PREPROCESSED_RFS_CLINICAL_DIR = (
    INTERIM_PICKLE_PREPROCESSED_RFS_DIR + "/CLINICAL"
)
INTERIM_PICKLE_PREPROCESSED_RFS_GENES_DIR = (
    INTERIM_PICKLE_PREPROCESSED_RFS_DIR + "/GENES"
)
# ......................................................................................................
### データのヘッド部分のみ保存するためのディレクトリ
INTERIM_OTHERS_DIR = INTERIM_DIR + "/others"
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
# 分析結果のディレクトリ
REPORT_DIR = "../reports"
# ------------------------------------------------------------------------------------------------------
## 画像（グラフなど）の保存
FIGURES_DIR = REPORT_DIR + "/figures"
# ......................................................................................................
### EDAで生成された画像
FIGURES_EDA_DIR = FIGURES_DIR + "/EDA"
#### 予後のEDAで生成された画像
FIGURES_PROGNOSIS_DIR = FIGURES_EDA_DIR + "/PROGNOSIS"
##### 臨床・遺伝子の予後のEDAで生成された画像
FIGURES_PROGNOSIS_CLINICAL_DIR = FIGURES_PROGNOSIS_DIR + "/CLINICAL"
FIGURES_PROGNOSIS_GENES_DIR = FIGURES_PROGNOSIS_DIR + "/GENES"
FIGURES_PROGNOSIS_CROSS_DIR = FIGURES_PROGNOSIS_DIR + "/CROSS"

# ......................................................................................................
### preprocessで生成された画像
FIGURES_PREPROCESS_DIE = FIGURES_DIR + "/PREPROCESS"
#### 予後のpreprocessで生成された画像
FIGURES_PREPROCESS_PROGNOSIS_DIR = FIGURES_PREPROCESS_DIE + "/PROGNOSIS"
##### 臨床・遺伝子の予後のpreprocessで生成された画像
FIGURES_PREPROCESS_PROGNOSIS_CLINICAL_DIR = (
    FIGURES_PREPROCESS_PROGNOSIS_DIR + "/CLINICAL"
)
FIGURES_PREPROCESS_PROGNOSIS_GENES_DIR = FIGURES_PREPROCESS_PROGNOSIS_DIR + "/GENES"
# ......................................................................................................
### モデル作成で生成された画像
FIGURES_MODELS_DIR = FIGURES_DIR + "/MODELS"
#### 予後のモデル作成で生成れた画像
FIGURES_MODELS_DIR_PROGNOSIS_DIR = FIGURES_MODELS_DIR + "/PROGNOSIS"
##### 臨床・遺伝子の予後のモデル作成生成された画像
FIGURES_MODELS_DIR_PROGNOSIS_CLINICAL_DIR = (
    FIGURES_MODELS_DIR_PROGNOSIS_DIR + "/CLINICAL"
)
FIGURES_MODELS_DIR_PROGNOSIS_GENES_DIR = FIGURES_MODELS_DIR_PROGNOSIS_DIR + "/GENES"
# ......................................................................................................
### その他の画像
FIGURES_OTHERS_DIR = FIGURES_DIR + "/OTHERS"
# ======================================================================================================


# =====================================================================================================
# URL
URL_cBioPortal = "https://cbioportal-datahub.s3.amazonaws.com/brca_metabric.tar.gz"
# =====================================================================================================


# =====================================================================================================
# SEED
SEED = 100
# =====================================================================================================


# =====================================================================================================
# 比較する2値分類器の設定
bcm_names = [
    "Logistic Regression",
    "Nearest Neighbors",
    "Linear SVM",
    "Polynomial SVM",
    "RBF SVM",
    "Sigmoid SVM",
    "Decision Tree",
    "Random Forest",
    "AdaBoost",
    "Naive Bayes",
    # "Linear Discriminant Analysis", # predictメソッドに対応していない
    "Quadratic Discriminant Analysis",
]

classifiers = [
    LogisticRegression(max_iter=2000, random_state=SEED),
    KNeighborsClassifier(),
    SVC(kernel="linear", random_state=SEED),
    SVC(kernel="poly", random_state=SEED),
    SVC(kernel="rbf", random_state=SEED),
    SVC(kernel="sigmoid", random_state=SEED),
    DecisionTreeClassifier(
        min_samples_split=20, min_samples_leaf=15, random_state=SEED
    ),
    RandomForestClassifier(
        min_samples_split=20, min_samples_leaf=15, random_state=SEED
    ),
    AdaBoostClassifier(random_state=SEED),
    GaussianNB(),
    QDA(),
]
# =====================================================================================================
