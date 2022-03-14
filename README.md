breast_cancer_analytics
==============================

修士論文共同研究

# 環境構築
共同で研究するときには、それぞれの環境で使用しているライブラリ等のバージョンに起因するエラーを防ぐために開発環境を統一することが推奨されます。 
PythonやAnacondaのようなデータ解析ツールは2022年現在、DS界隈の標準と言っても過言ではありません。でもMATLABやExcel、RにJuliaと様々な物があるので過言かもしれません。  
何にせよ、今回の分析では201X年以降入学された皆さんが講義で使い慣れたであろうPythonをベースとしたパッケージであるAnacondaを中心に扱っていきます。  
Anacondaではいわゆる**仮想環境**を作成することができ、この仮想環境を作るための設計図である*hogehoge*.yml（hogehogeは任意のファイル名です）をAnaconda環境で読み込むことでOSやハードウェアの垣根を超えて同じ環境を作成することができます。  
当リポジトリをcloneしたらまずは仮想環境を構築してみましょう。  
尚、PyCaretというライブラリを利用するに当たり、M1 Macを利用している方は依存ライブラリのインストールにエラーが発生します。
PyCaretを利用する際は別途pycaret_requirement.txtからPyCaretの依存ライブラリを**pip install**してください。

## 仮想環境構築注意点
### データ分析環境
- Anaconda仮想環境のファイル
    - breast_cancer_analytics.yml  

### モデル構築環境
- pycaretインストール（M1 Macでは、conda経由でpycaretライブラリのインストールにエラーが発生します。以下のファイルをもとにpip経由でインストールしてください）
    - pycaret_requirements.txt
### コマンド
- 仮想環境インストール
    - conda env create -n *任意の仮想環境名* -f *仮想環境ファイル名*.yml  
    例) conda env create -n breast-cancer-analytics -f breast_cancer_analytics.yml
- 仮想環境エクスポート
    - conda env export > *hogehoge*.yml  
    例) conda env export > breast_cancer_analytics.yml

# プロジェクト構成
プロジェクトの構成をディレクトリで個人の思うままに管理すると不満に思う人も出てくるでしょう。
しかし、1人1人の思想を反映するとディレクトリ構成を決めるだけでも大変です。
後々必要に迫られディレクトリ構成の変更を迫られたりするかもしれません。  
そこで、広く使用されている構成を真似るために**Cookiecutter**というライブラリを用いました（https://github.com/cookiecutter/cookiecutter）。  
CookiecutterはOSに依存しないクロスプラットフォームで使用できるパッケージ構成管理テンプレートを展開するPythonライブラリです。
PythonだけでなくJacaScriptやRubyなど様々な言語で使用できますが、詳しくは公式のドキュメントを読んでください。
兎にも角にもCookiecutterの形式に則り、今回の分析を実施していきます。
下記のツリー状の図はCookiecutterで展開したディレクトリに配置するべきファイル等の説明です。
これらを参考に、ファイルを利用する際は注意してください。

---

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
