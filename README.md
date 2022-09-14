# breast_cancer_analytics

修士論文共同研究

言語：Python

# 環境構築

共同で研究するときには、それぞれの環境で使用しているライブラリ等のバージョンに起因するエラーを防ぐために開発環境を統一することが推奨されます。
Python や Anaconda のようなデータ解析ツールは 2022 年現在、DS 界隈の標準と言っても過言ではありません
（MATLAB や Excel、R に Julia と様々な物があるのでやっぱり過言かもしれません）。  
今回の分析では 201X 年以降入学された皆さんが講義で使い慣れたであろう
Python をベースとしたパッケージである Anaconda を中心に扱っていきます。  
Anaconda ではいわゆる**仮想環境**を作成することができ、
この仮想環境を作るための設計図である*hogehoge*.yml（hogehoge は任意のファイル名です）を
Anaconda 環境で読み込むことで、同一のosであれば同じ環境を再現することができます。  
また、pip 等と比較してパッケージ管理が容易であり、不要になったライブラリは*conda uninstall*にて依存ライブラリ共々綺麗に消すことができます。
しかし、異なるos間では環境の再現が困難です。  
そこで、Docker で環境のコンテナ化をし、異なるos上にコンテナ環境を構築し、その環境上でAnaconda仮想環境を構築するシステムを構築しました。  
このシステムによって異なるosでも同一の実験環境を整えることが可能になります。

## Docker

以下のコマンドで環境を構築できます。  
不足しているライブラリは適宜 docker 環境内で`conda install (or pip install)`してください。

```
# PLEASE RUN THESE COMMANDS UNDER THIS PROJECT REPOSITORY (./breast-cancer-analytics)

# build docker image
# warning: please cheack Dockerifile 1st command 'FROM ~'
# please use 'FROM ubuntu:latest' if you use Intel chip
# please use 'FROM --platform=linux/amd64 ubuntu:latest' if you use m1 chip
# don't forget "." in this command
docker build --tag breast-cancer-analytics .

# run docker container
# please run this command in this dir
docker run --mount type=bind,source="$(pwd)",target=/breast_cancer_analytics -p 8888:8888 -it breast-cancer-analytics:latest

# relaunch docker container
# please check <container id> by this command 'docker ps --all'
docker restart <container id>
# rejoin relaunched docker container
docker container exec -it <container name> bash

# build conda env
conda env create -f=conda_env.yml
# relaunch docker terminal(shell)

# launch jupyter labs
# access 'http://localhost:8888' on your browser with token in terminal
jupyter-lab --ip 0.0.0.0 --allow-root

# export conda env (RUN UNDER ./breast-cancer-analytics)
# 新しくライブラリ等インストールしたら、このコマンドを実行し、仮想環境に記録すること
conda env export > conda_env.yml

```

# git・github によるソースコード管理

複数人が携わる開発環境では、それぞれの開発部分が重複したり、バラバラに開発するせいで収集がつかなくなることがあります。  
これは事前にルールを決めてもうっかり忘れが発生したり、ルールの管理・認知コストがかかってしまい、完全になくすことはできません。  
そこで、ソースコードを管理する上でスタンダードなツールである git・github を使用します。

## 理解する必要のある概念

以下のコマンドや概念については理解してください。  
git や github の基本的な概念になります。　　
Google で調べれば出てきます。

### コマンド

- git add .
- git commit -m 'comment'
- git push origin _your remote branch_
- git pull origin main

### 概念

- local と remote
- branch
- merge と conflict

## コマンドテンプレート

### github へ変更を送信するコマンド

```
git add .

git commit -m 'comment'

git push origin <remote_branch_name>
```

### remote の内容を local に反映するコマンド

```
git pull origin <remote_branch_name>
```

## 各種概念

### branch（自身の開発領域）の概念

共同開発の際は、様々な人が開発をするため、その内容が様々に派生していきます。  
１プロジェクトを全員で共有して同時に開発を進めると、旧バージョンで存在した機能が補修中となったために該当機能に依存した他の機能が停止してしまうことも起こりえます。
バグがあった場合、全員の進捗を戻す必要もあるでしょう。
そのため、個々人の開発領域を分けることで、上記の問題点を改善・低減することが重要になります。
*git branch*コマンドはこの開発領域の区分けを実現する機能です。  
**かならず自身の branch（開発領域）を設定してください。**

#### ブランチに関するコマンド

##### ブランチの作成

```
git branch <branch_name>
```

#### ブランチの移動

```
git checkout <branch_name>
```

# ノートブック規則

ノートブックで解析をすすめる上での規則を以下に記します。  
視認性向上のため、なるべく厳守してください。  
また、ノートブックにも定義した内容や、処理内容を、少なくともメモなどとして残してください。

## 整形のためのライブラリ

他の人がコードを参考にしたり、レビューするときに、閲覧者の理解の妨げにならないようコードを整形することが求められます。
この時、個人個人でフォーマッタ方法が異なると意味がありません。  
そこで今回は以下のツールを用いてコードの整形を実施します。  
拡張機能を利用し、保存時に自動で整形されるようにすると良いでしょう。

- フォーマッタ: black
- コードチェッカー: flake8

## 基本命名規則

コードを記述する際、様々な対象を命名するが、その名付けに規則を設けることでチーム内での理解が促進されます。  
[PEP8](https://pep8-ja.readthedocs.io/ja/latest/)に則ったルールであるので、他の場面でも意識することを推奨します。  
参考: https://qiita.com/naomi7325/items/4eb1d2a40277361e898b

| 対象       | ルール                                 | 例            |
| ---------- | -------------------------------------- | ------------- |
| パッケージ | 全小文字。アンダースコア（\_）非推奨。 | numpy, pandas |
| モジュール | 全小文字。アンダースコア可。           | sys, os       |
| クラス     | 最初大文字＋大文字区切り               | MyClass       |
| 例外       | 最初大文字＋大文字区切り               | MyError       |
| 型変数     | 最初大文字＋大文字区切り               | MyType        |
| メソッド   | 全小文字＋アンダースコア区切り         | my_method     |
| 関数       | 全小文字＋アンダースコア区切り         | my_function   |
| 変数       | 全小文字＋アンダースコア区切り         | my_variable   |
| 定数       | 全大文字＋アンダースコア区切り         | MY_CONST      |

## ノートブックファイル名規則

jupyter notebook は全て./notebooks 以下に置くこと。

### 命名規則

基本的な命名規則は以下だが、臨機応変に変更してください。  
（カテゴリ番号）.（作成番号）\-（処理カテゴリ名）\_（処理具体内容・対象）.ipynb  
例. 0.0-download_data.ipynb

#### カテゴリ番号の区分について

**大区分** 
0. 分析以前の処理。データの取得など。
1. EDA
2. 前処理
3. モデル構築

**小区分**
1. X.0.X: 予後の予測
2. X.1.X: 再発の予測

### ノートブック内セル規則

#### 関数名

関数が何を行うのかがわかるような名前をつけてください。  
また、基本命名規則に従い、全て小文字+アンダースコア区切りとしてください。

#### 関数内容

入出力がわかる関数共通使用のために、関数アノテーションを実施してください。  
また、その関数が何を行うのか簡単なコメントを残すようにしてください。  
例．

```
def function_compare(val1: int, val2: int) -> bool:
    return val1 > val2
```

# プロジェクト構成

プロジェクトの構成をディレクトリで個人の思うままに管理すると不満に思う人も出てくるでしょう。
しかし、1 人 1 人の思想を反映するとディレクトリ構成を決めるだけでも大変です。
後々必要に迫られディレクトリ構成の変更を迫られたりするかもしれません。  
そこで、広く使用されている構成を真似るために**Cookiecutter**というライブラリを用いました（https://github.com/cookiecutter/cookiecutter）。  
Cookiecutter は OS に依存しないクロスプラットフォームで使用できるパッケージ構成管理テンプレートを展開する Python ライブラリです。
Python だけでなく JacaScript や Ruby など様々な言語で使用できますが、詳しくは公式のドキュメントを読んでください。
兎にも角にも Cookiecutter の形式に則り、今回の分析を実施していきます。
下記のツリー状の図は Cookiecutter で展開したディレクトリに配置するべきファイル等の説明です。
これらを参考に、ファイル・ディレクトリを操作する際は注意してください。

---

## Project Organization

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

---

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
