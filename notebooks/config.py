# DIRs
# basic dirs
DATA_DIR = "../data"
EXTERNAL_DIR = DATA_DIR + "/external"
INTERIM_DIR = DATA_DIR + "/interim"
PROCESSED_DIR = DATA_DIR + "/processed"
RAW_DIR = DATA_DIR + "/raw"

# raw data dirs
RAW_BRCA_METABRIC_DIR = RAW_DIR + "/brca_metabric"
# raw data file path
RAW_BRCA_METABRIC_TG = RAW_DIR + "/brca_metabric.tar.gz"

# external dirs
EXTERNAl_BRCA_METABRIC_DATA_CLNICAL_DIR = EXTERNAL_DIR + "/brca_metabric_data_clinical"

# interim dirs
INTERIM_BRCA_METABRIC_DIR = INTERIM_DIR + "/brca_metabric"
INTERIM_PICKLE_DIR = INTERIM_DIR + "/pickle_data"
INTERIM_PICKLE_EDA_DIR = INTERIM_PICKLE_DIR + "/EDA_data"
INTERIM_PICKLE_PREPROCESSED_DIR = INTERIM_PICKLE_DIR + "/preprocessed"

# save models dir
MODELS="../models"
MODELS_NOTEBOOK=MODELS+"/notebooks"

# analysoutput dirs
REPORT_DIR = "../reports"
# image and plot save dirs
FIGURES_DIR = REPORT_DIR + "/figures"
   # SEABORN_DIR = FIGURES_DIR + "/EDA"9
TREE_DIR = FIGURES_DIR + "/decision_tree"# need to modify name
# 予後の予測タスクの保存先
FIGURES_PROGNOSIS_SURVIVED_DIR = FIGURES_DIR+'/prognosis_survived'
# 2値分類タスク
FIGURES_PROGNOSIS_SURVIVED_BCM_DIR=FIGURES_PROGNOSIS_SURVIVED_DIR+'/create_bcm_models'
# オーバーサンプリングの実施結果の保存先
FIGURES_PROGNOSIS_SURVIVED_BCM_OVERSAMPLING_DIR=FIGURES_PROGNOSIS_SURVIVED_BCM_DIR+'/over_sampling'
# 手法：SMOTE
FIGURES_PROGNOSIS_SURVIVED_BCM_OVERSAMPLING_SMOTE_DIR=FIGURES_PROGNOSIS_SURVIVED_BCM_OVERSAMPLING_DIR+'/SMOTE'


# reports
PANDAS_PROFILING_REPORT_DIR = REPORT_DIR + "/pandas-profiling_report"
SWEETVIZ_REPORT_DIR = REPORT_DIR + "/sweetviz_report"


# SEED
SEED=100
