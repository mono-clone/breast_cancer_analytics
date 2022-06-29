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
