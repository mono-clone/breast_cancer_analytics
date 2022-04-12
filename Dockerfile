# 参考：https://qiita.com/ku_a_i/items/8c7d91f6e13a5091f51c

# docker pull continuumio/anaconda3
# docker run --name breast_cancer_analytics -v $PWD:/breast_cancer_analytics -p 8888:8888 --rm -it continuumio/anaconda3:latest
# conda create -n breast-cancer-analytics python=3.8
# conda activate breast-cancer-analytics
# conda config --add conda-forge
# conda install jupyter jupyterlab 
# jupyter lab --ip 0.0.0.0 --allow-root /breast_cancer_analytics
