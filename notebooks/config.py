# 機械学習モデル
from sklearn.linear_model import LogisticRegression  # ロジスティック回帰
from sklearn.neighbors import KNeighborsClassifier  # K近傍法
from sklearn.svm import SVC  # サポートベクターマシン
from sklearn.tree import DecisionTreeClassifier, export_graphviz  # 決定木
from sklearn.ensemble import RandomForestClassifier  # ランダムフォレスト
from sklearn.ensemble import AdaBoostClassifier  # AdaBoost
from sklearn.naive_bayes import GaussianNB  # ナイーブ・ベイズ
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA  # 二次判別分析

# DIRs
# data dirs
DATA_DIR = "../data"
## 外部データ
EXTERNAL_DIR = DATA_DIR + "/external"
### external dirs
EXTERNAl_BRCA_METABRIC_DATA_CLNICAL_DIR = EXTERNAL_DIR + "/brca_metabric_data_clinical"
##加工後データ（実験中）
INTERIM_DIR = DATA_DIR + "/interim"
### interim dirs
INTERIM_BRCA_METABRIC_DIR = INTERIM_DIR + "/brca_metabric"
INTERIM_PICKLE_DIR = INTERIM_DIR + "/pickle_data"
#### EDA
INTERIM_PICKLE_EDA_DIR = INTERIM_PICKLE_DIR + "/EDA_data"
##### 各EDA
INTERIM_PICKLE_EDA_CLINICAL_DIR = INTERIM_PICKLE_EDA_DIR +"/clinical"
INTERIM_PICKLE_EDA_GENES_DIR=INTERIM_PICKLE_EDA_DIR+"/genes"
### preprocess（実験中）
INTERIM_PICKLE_PREPROCESSED_DIR = INTERIM_PICKLE_DIR + "/preprocessed"
INTERIM_PICKLE_PREPROCESSED_OS5YEARS_DIR=INTERIM_PICKLE_PREPROCESSED_DIR+'/OS5YEARS'
INTERIM_PICKLE_PREPROCESSED_OS5YEARS_CLINICAL_DIR=INTERIM_PICKLE_PREPROCESSED_OS5YEARS_DIR+'/CLINICAL'
INTERIM_PICKLE_PREPROCESSED_OS5YEARS_GENES_DIR=INTERIM_PICKLE_PREPROCESSED_OS5YEARS_DIR+'/GENES'
INTERIM_PICKLE_PREPROCESSED_RFS_DIR=INTERIM_PICKLE_PREPROCESSED_DIR+'/RFS'
## 本番環境前処理データ
PROCESSED_DIR = DATA_DIR + "/processed"
## 生データ
RAW_DIR = DATA_DIR + "/raw"
### raw data dirs
RAW_BRCA_METABRIC_DIR = RAW_DIR + "/brca_metabric"
### raw data file path
RAW_BRCA_METABRIC_TG = RAW_DIR + "/brca_metabric.tar.gz"

# save models dir
MODELS="../models"
MODELS_NOTEBOOK=MODELS+"/notebooks"

# analysoutput dirs
REPORT_DIR = "../reports"
## image and plot save dirs
FIGURES_DIR = REPORT_DIR + "/figures"
### 予後の予測タスクの保存先
FIGURES_PROGNOSIS_SURVIVED_DIR = FIGURES_DIR+'/prognosis_survived'
#### 2値分類タスク
FIGURES_PROGNOSIS_SURVIVED_BCM_DIR=FIGURES_PROGNOSIS_SURVIVED_DIR+'/create_bcm_models'
##### 基本のプロット
FIGURES_PROGNOSIS_SURVIVED_BCM_BASIC_DIR=FIGURES_PROGNOSIS_SURVIVED_BCM_DIR+'/basic'
##### オーバーサンプリングの実施結果の保存先
FIGURES_PROGNOSIS_SURVIVED_BCM_OVERSAMPLING_DIR=FIGURES_PROGNOSIS_SURVIVED_BCM_DIR+'/over_sampling'
###### オーバーサンプリング詳細手法：SMOTE
FIGURES_PROGNOSIS_SURVIVED_BCM_OVERSAMPLING_SMOTE_DIR=FIGURES_PROGNOSIS_SURVIVED_BCM_OVERSAMPLING_DIR+'/SMOTE'
#### 正規化の実施結果の保存先
FIGURES_PROGNOSIS_SURVIVED_BCM_NORMALIZATION_DIR=FIGURES_PROGNOSIS_SURVIVED_BCM_DIR+'/normalization'
### EDA画像
SEABORN_DIR = FIGURES_DIR + "/EDA"
### 木モデルの画像
TREE_DIR = FIGURES_DIR + "/decision_tree"# need to modify name
## reports
PANDAS_PROFILING_REPORT_DIR = REPORT_DIR + "/pandas-profiling_report"
SWEETVIZ_REPORT_DIR = REPORT_DIR + "/sweetviz_report"


# URL
URL_cBioPortal='https://cbioportal-datahub.s3.amazonaws.com/brca_metabric.tar.gz'

# SEED
SEED=100


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
    DecisionTreeClassifier(min_samples_split=20, min_samples_leaf=15,random_state=SEED),
    RandomForestClassifier(min_samples_split=20, min_samples_leaf=15,random_state=SEED),
    AdaBoostClassifier(random_state=SEED),
    GaussianNB(),
    QDA(),
]

