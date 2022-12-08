apt-get update
apt-get upgrade -y
apt-get install -y \
    vim \
    wget \
    bash \
    make
apt-get clean

# install miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh -O Miniconda.sh
bash Miniconda.sh -b
rm -rf Miniconda.sh

# prepend the new path
export PATH="/root/miniconda3/bin:$PATH"

# install packages
pip install --upgrade pip
conda update -n base -c defaults conda -y
conda update --all -y

conda install -y numpy pandas matplotlib seaborn
conda install -y flake8 black
conda install -y jupyter jupyterlab
conda install -y scikit-learn
conda install -c conda-forge -y imbalanced-learn scikit-survival lifelines optuna shap eli5

wget https://anaconda.org/conda-forge/nodejs/18.12.1/download/linux-aarch64/nodejs-18.12.1-he850d6a_0.tar.bz2
conda install nodejs-18.12.1-he850d6a_0.tar.bz2
rm nodejs-18.12.1-he850d6a_0.tar.bz2
