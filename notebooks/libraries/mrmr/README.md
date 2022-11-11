<p align="center">
<img src="./docs/img/mrmr_logo_white_bck.png" alt="drawing" width="450"/>
</p>

## What is mRMR

*mRMR*, which stands for "minimum Redundancy - Maximum Relevance", is a feature selection algorithm.

## Why is it unique

The peculiarity of *mRMR* is that it is a **minimal-optimal** feature selection algorithm. <br/>
This means it is designed to find the smallest relevant subset of features for a given Machine Learning task.

Selecting the minimum number of useful features is desirable for many reasons:
- memory consumption,
- time required,
- performance,
- explainability of results.

This is why a minimal-optimal method such as *mrmr* is often preferable.

On the contrary, the majority of other methods (for instance, Boruta or Positive-Feature-Importance) are classified as **all-relevant**, 
since they identify all the features that have some kind of relationship with the target variable.

## When to use mRMR

Due to its efficiency, *mRMR* is ideal for practical ML applications, 
where it is necessary to perform feature selection frequently and automatically, 
in a relatively small amount of time.

For instance, in **2019**, **Uber** engineers published a paper describing how they implemented 
*mRMR* in their marketing machine learning platform [Maximum Relevance and Minimum Redundancy Feature Selection Methods for a Marketing Machine Learning Platform](https://eng.uber.com/research/maximum-relevance-and-minimum-redundancy-feature-selection-methods-for-a-marketing-machine-learning-platform/).

## How to install this package

You can install this package in your environment via pip:

<pre>
pip install mrmr_selection
</pre>

And then import it in Python through:

<pre>
import mrmr
</pre>

## How to use this package

This package is designed to do *mMRM* selection through different tools, depending on your needs and constraints.

Currently, the following tools are supported (others will be added):
- **Pandas** (in-memory)
- **Spark**
- **Google BigQuery**

The package has a module for each supported tool. Each module has *at least* these 2 functions:
- `mrmr_classif`, for feature selection when the target variable is categorical (binary or multiclass).
- `mrmr_regression`, for feature selection when the target variable is numeric.

Let's see some examples.

#### 1. Pandas example
You have a Pandas DataFrame (`X`) and a Series which is your target variable (`y`).
You want to select the best `K` features to make predictions on `y`.

```python
# create some pandas data
from sklearn.datasets import make_classification
X, y = make_classification(n_samples = 1000, n_features = 50, n_informative = 10, n_redundant = 40)
X = pd.DataFrame(X)
y = pd.Series(y)

# select top 10 features using mRMR
from mrmr import mrmr_classif
selected_features = mrmr_classif(X=X, y=y, K=10)
```

Note: the output of mrmr_classif is a list containing K selected features. This is a **ranking**, therefore, if you want to make a further selection, take the first elements of this list.

#### 2. Spark example

```python
# create some spark data
import pyspark
session = pyspark.sql.SparkSession(pyspark.context.SparkContext())
data = [(1.0, 1.0, 1.0, 7.0, 1.5, -2.3), 
        (2.0, float('NaN'), 2.0, 7.0, 8.5, 6.7), 
        (2.0, float('NaN'), 3.0, 7.0, -2.3, 4.4),
        (3.0, 4.0, 3.0, 7.0, 0.0, 0.0),
        (4.0, 5.0, 4.0, 7.0, 12.1, -5.2)]
columns = ["target", "some_null", "feature", "constant", "other_feature", "another_feature"]
df_spark = session.createDataFrame(data=data, schema=columns)

# select top 2 features using mRMR
import mrmr
selected_features = mrmr.spark.mrmr_regression(df=df_spark, target_column="target", K=2)
```

#### 3. Google BigQuery example

```python
# initialize BigQuery client
from google.cloud.bigquery import Client
bq_client = Client(credentials=your_credentials)

# select top 20 features using mRMR
import mrmr
selected_features = mrmr.bigquery.mrmr_regression(
    bq_client=bq_client,
    table_id='bigquery-public-data.covid19_open_data.covid19_open_data',
    target_column='new_deceased',
    K=20
)
```


## Reference

For an easy-going introduction to *mRMR*, read my article on **Towards Data Science**: [“MRMR” Explained Exactly How You Wished Someone Explained to You](https://towardsdatascience.com/mrmr-explained-exactly-how-you-wished-someone-explained-to-you-9cf4ed27458b).

Also, this article describes an example of *mRMR* used on the world famous **MNIST** dataset: [Feature Selection: How To Throw Away 95% of Your Data and Get 95% Accuracy](https://towardsdatascience.com/feature-selection-how-to-throw-away-95-of-your-data-and-get-95-accuracy-ad41ca016877).

*mRMR* was born in **2003**, this is the original paper: [Minimum Redundancy Feature Selection From Microarray Gene Expression Data](https://www.researchgate.net/publication/4033100_Minimum_Redundancy_Feature_Selection_From_Microarray_Gene_Expression_Data).

Since then, it has been used in many practical applications, due to its simplicity and effectiveness.
For instance, in **2019**, **Uber** engineers published a paper describing how they implemented MRMR in their marketing machine learning platform [Maximum Relevance and Minimum Redundancy Feature Selection Methods for a Marketing Machine Learning Platform](https://eng.uber.com/research/maximum-relevance-and-minimum-redundancy-feature-selection-methods-for-a-marketing-machine-learning-platform/).
