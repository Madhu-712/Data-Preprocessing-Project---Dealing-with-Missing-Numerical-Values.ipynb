# Data-Preprocessing-Project---Dealing-with-Missing-Numerical-Values.ipynb

Data Preprocessing Project - Dealing with Missing Numerical Values
In this project, I discuss various data preprocessing techniques to handle missing numerical values. The contents of this project are categorized into various sections which are listed in table of contents as follows:-

Table of Contents
Introduction

Source dataset

Dealing with missing numerical values

Drop missing values with dropna()

Fill missing values with a test statistic

Fill missing values with Imputer

Build a prediction model

KNN Imputation

Check with ASSERT statement

References

1. Introduction
Over the last few decades, Machine Learning (ML) has gained immense popularity in solving real world business problems. It has emerged as a technology tool for companies looking to boost productivity and profit. ML practitioners source real world data and write algorithms to solve business problems. The success of the ML algorithms depends on the quality of the data. The data must be free from errors and discrepancies. It must adhere to a specific standard so that ML algorithms can accept them. But, this does not happen in reality.

In reality, the data has its own limitations. The data is dirty. It is incomplete, noisy and inconsistent. Incomplete data means it has missing values and lacks certain attributes. The data may be noisy as it contains errors and outliers and hence does not produce desired results. Lastly, the data may be inconsistent as it contains discrepancies in data or duplicate data.

So, ML practitioners must take actions to transform raw data into standardized data that is suitable for ML algorithms. It involves cleaning, transforming and standardizing data to remove all the inadequacies and irregularities in the data. These actions are collectively known as Data Preprocessing.

2. Source dataset
I have used wiki4HE.csv data set for this project. I have downloaded this data set from the UCI Machine Learning Repository. The data set describes survey results of faculty members from two Spanish universities on teaching uses of Wikipedia.

The dataset contains 53 attributes and 913 instances. Out of the 53 attributes, 4 are of numeric data types and 49 are of text or character data types.

The data set can be found at the following url-

https://archive.ics.uci.edu/ml/datasets/wiki4HE

3. Dealing with missing numerical values
It is a very common scenario that when looking at a real world data, a data scientist may come across missing values. These missing values could be due to error prone data entry process, wrong data collection methods, certain values not applicable, particular fields left blank in a survey or the respondent decline to answer. Whatever may be the reason for the missing value, the data scientist must find ways to handle these missing values. He knows that missing values need to be handled carefully, because they give wrong results if we simply ignore them. He must answer whether he should delete these missing values or replace them with a suitable statistic. The first step in dealing with missing values properly is to identify them.

The initial inspection of the data help us to detect whether there are missing values in the data set. It can be done by Exploratory Data Analysis. So, it is always important that a data scientist always perform Exploratory Data Analysis (EDA) to identify missing values correctly.

# Import required libraries

import numpy as np

import pandas as pd
# Import the dataset

dataset = "C:/project_datasets/wiki4HE.csv"

df = pd.read_csv(dataset, sep = ';')
Exploratory Data Analysis (EDA)
Below is the list of commands to identity missing values with EDA.

df.head()
This will output the first five rows of the dataset. It will give us quick view on the presence of ‘NaN’ or ‘?’ ‘-1’ or ’0’ or blank spaces “” in the dataset. If required, we can view more number of rows by specifying the number of rows inside the parenthesis.

# View the first 5 rows of the dataset

print(df.head())
   AGE  GENDER DOMAIN  PhD YEARSEXP  UNIVERSITY UOC_POSITION OTHER_POSITION  \
0   40       0      2    1       14           1            2              ?   
1   42       0      5    1       18           1            2              ?   
2   37       0      4    1       13           1            3              ?   
3   40       0      4    0       13           1            3              ?   
4   51       0      6    0        8           1            3              ?   

  OTHERSTATUS USERWIKI ...  BI2 Inc1 Inc2 Inc3 Inc4 Exp1 Exp2 Exp3 Exp4 Exp5  
0           ?        0 ...    3    5    5    5    5    4    4    4    1    2  
1           ?        0 ...    2    4    4    3    4    2    2    4    2    4  
2           ?        0 ...    1    5    3    5    5    2    2    2    1    3  
3           ?        0 ...    3    3    4    4    3    4    4    3    3    4  
4           ?        1 ...    5    5    5    4    4    5    5    5    4    4  

[5 rows x 53 columns]
Interpretation

We can see that there are lots of missing values in the dataset. The columns OTHER_POSITION and OTHERSTATUS contain missing values.

The column GENDER contain zeros. It might be because of Male is encoded as 1 and Female is encoded as 0.

We need to explore the dataset further to confirm which columns contain the missing values.

df.info()
This command is quite useful in detecting the missing values in the dataset. It will tell us the total number of non - null observations present including the total number of entries. Once number of entries isn’t equal to number of non - null observations, we know there are missing values in the dataset.

# View the summary of the dataframe

print(df.info())
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 913 entries, 0 to 912
Data columns (total 53 columns):
AGE               913 non-null int64
GENDER            913 non-null int64
DOMAIN            913 non-null object
PhD               913 non-null int64
YEARSEXP          913 non-null object
UNIVERSITY        913 non-null int64
UOC_POSITION      913 non-null object
OTHER_POSITION    913 non-null object
OTHERSTATUS       913 non-null object
USERWIKI          913 non-null object
PU1               913 non-null object
PU2               913 non-null object
PU3               913 non-null object
PEU1              913 non-null object
PEU2              913 non-null object
PEU3              913 non-null object
ENJ1              913 non-null object
ENJ2              913 non-null object
Qu1               913 non-null object
Qu2               913 non-null object
Qu3               913 non-null object
Qu4               913 non-null object
Qu5               913 non-null object
Vis1              913 non-null object
Vis2              913 non-null object
Vis3              913 non-null object
Im1               913 non-null object
Im2               913 non-null object
Im3               913 non-null object
SA1               913 non-null object
SA2               913 non-null object
SA3               913 non-null object
Use1              913 non-null object
Use2              913 non-null object
Use3              913 non-null object
Use4              913 non-null object
Use5              913 non-null object
Pf1               913 non-null object
Pf2               913 non-null object
Pf3               913 non-null object
JR1               913 non-null object
JR2               913 non-null object
BI1               913 non-null object
BI2               913 non-null object
Inc1              913 non-null object
Inc2              913 non-null object
Inc3              913 non-null object
Inc4              913 non-null object
Exp1              913 non-null object
Exp2              913 non-null object
Exp3              913 non-null object
Exp4              913 non-null object
Exp5              913 non-null object
dtypes: int64(4), object(49)
memory usage: 378.1+ KB
None
Interpretation

The above command shows that there are no missing values in the data set. But, this is not true. The dataset contains missing values.It may be because of missing values are encoded in different ways.

Encode missing numerical values
Missing values are encoded in different ways. They can appear as ‘NaN’, ‘NA’, ‘?’, zero ‘0’, ‘xx’, minus one ‘-1’ or a blank space “ ”. We need to use various pandas methods to deal with missing values. But, pandas always recognize missing values as ‘NaN’. So, it is essential that we should first convert all the ‘?’, zeros ‘0’, ‘xx’, minus ones ‘-1’ or blank spaces “ ” to ‘NaN’. If the missing values isn’t identified as ‘NaN’, then we have to first convert or replace such non ‘NaN’ entry with a ‘NaN’.

Convert '?' to ‘NaN’
df[df == '?'] = np.nan

# Convert '?' to 'NaN'

df[df == '?'] = np.nan
# View the first 5 rows of the dataset again

print(df.head())
   AGE  GENDER DOMAIN  PhD YEARSEXP  UNIVERSITY UOC_POSITION OTHER_POSITION  \
0   40       0      2    1       14           1            2            NaN   
1   42       0      5    1       18           1            2            NaN   
2   37       0      4    1       13           1            3            NaN   
3   40       0      4    0       13           1            3            NaN   
4   51       0      6    0        8           1            3            NaN   

  OTHERSTATUS USERWIKI ...  BI2 Inc1 Inc2 Inc3 Inc4 Exp1 Exp2 Exp3 Exp4 Exp5  
0         NaN        0 ...    3    5    5    5    5    4    4    4    1    2  
1         NaN        0 ...    2    4    4    3    4    2    2    4    2    4  
2         NaN        0 ...    1    5    3    5    5    2    2    2    1    3  
3         NaN        0 ...    3    3    4    4    3    4    4    3    3    4  
4         NaN        1 ...    5    5    5    4    4    5    5    5    4    4  

[5 rows x 53 columns]
# View the summary of the dataframe again

print(df.info())
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 913 entries, 0 to 912
Data columns (total 53 columns):
AGE               913 non-null int64
GENDER            913 non-null int64
DOMAIN            911 non-null object
PhD               913 non-null int64
YEARSEXP          890 non-null object
UNIVERSITY        913 non-null int64
UOC_POSITION      800 non-null object
OTHER_POSITION    652 non-null object
OTHERSTATUS       373 non-null object
USERWIKI          909 non-null object
PU1               906 non-null object
PU2               902 non-null object
PU3               908 non-null object
PEU1              909 non-null object
PEU2              899 non-null object
PEU3              816 non-null object
ENJ1              906 non-null object
ENJ2              896 non-null object
Qu1               906 non-null object
Qu2               903 non-null object
Qu3               898 non-null object
Qu4               891 non-null object
Qu5               884 non-null object
Vis1              841 non-null object
Vis2              796 non-null object
Vis3              905 non-null object
Im1               891 non-null object
Im2               893 non-null object
Im3               856 non-null object
SA1               902 non-null object
SA2               901 non-null object
SA3               902 non-null object
Use1              899 non-null object
Use2              896 non-null object
Use3              904 non-null object
Use4              890 non-null object
Use5              898 non-null object
Pf1               902 non-null object
Pf2               907 non-null object
Pf3               899 non-null object
JR1               886 non-null object
JR2               860 non-null object
BI1               881 non-null object
BI2               870 non-null object
Inc1              878 non-null object
Inc2              878 non-null object
Inc3              876 non-null object
Inc4              871 non-null object
Exp1              900 non-null object
Exp2              902 non-null object
Exp3              900 non-null object
Exp4              899 non-null object
Exp5              900 non-null object
dtypes: int64(4), object(49)
memory usage: 378.1+ KB
None
Interpretation

Now, we can see that there are lots of columns containing missing values. We should view the column names of the dataframe.

# View the column names of dataframe

print(df.columns)
Index(['AGE', 'GENDER', 'DOMAIN', 'PhD', 'YEARSEXP', 'UNIVERSITY',
       'UOC_POSITION', 'OTHER_POSITION', 'OTHERSTATUS', 'USERWIKI', 'PU1',
       'PU2', 'PU3', 'PEU1', 'PEU2', 'PEU3', 'ENJ1', 'ENJ2', 'Qu1', 'Qu2',
       'Qu3', 'Qu4', 'Qu5', 'Vis1', 'Vis2', 'Vis3', 'Im1', 'Im2', 'Im3', 'SA1',
       'SA2', 'SA3', 'Use1', 'Use2', 'Use3', 'Use4', 'Use5', 'Pf1', 'Pf2',
       'Pf3', 'JR1', 'JR2', 'BI1', 'BI2', 'Inc1', 'Inc2', 'Inc3', 'Inc4',
       'Exp1', 'Exp2', 'Exp3', 'Exp4', 'Exp5'],
      dtype='object')
df.describe()
This will display summary statistics of all observed features and labels. The most important statistic is the minimum value. If we see -1 or 0 in our observations, then we can suspect missing value.

# View the descriptive statistics of the dataframe

print(df.describe())
              AGE      GENDER         PhD  UNIVERSITY
count  913.000000  913.000000  913.000000  913.000000
mean    42.246440    0.424973    0.464403    1.123768
std      8.058418    0.494610    0.499005    0.329497
min     23.000000    0.000000    0.000000    1.000000
25%     36.000000    0.000000    0.000000    1.000000
50%     42.000000    0.000000    0.000000    1.000000
75%     47.000000    1.000000    1.000000    1.000000
max     69.000000    1.000000    1.000000    2.000000
Interpretation

We can see there are four columns of integer data types - AGE, GENDER, PhD and UNIVERSITY.

In the AGE column, the maximum and minimum values are 69 and 23. The median value is 42 and the count is 913. We do not suspect any missing value in this column.

Similar, explanation goes for the PhD and UNIVERSITY columns.

The GENDER column has only two possible values 0 and 1. This is reasonable because 0 is for female and 1 is for male.

So, we do not find any missing values in the above four columns.

df.isnull()
The above command checks whether each cell in a dataframe contains missing values or not. If the cell contains missing value, it returns True otherwise it returns False.

df.isnull.sum()
The above command returns the total number of missing values in each column in the data set.

# View missing values in each column in the dataset

print(df.isnull().sum())
AGE                 0
GENDER              0
DOMAIN              2
PhD                 0
YEARSEXP           23
UNIVERSITY          0
UOC_POSITION      113
OTHER_POSITION    261
OTHERSTATUS       540
USERWIKI            4
PU1                 7
PU2                11
PU3                 5
PEU1                4
PEU2               14
PEU3               97
ENJ1                7
ENJ2               17
Qu1                 7
Qu2                10
Qu3                15
Qu4                22
Qu5                29
Vis1               72
Vis2              117
Vis3                8
Im1                22
Im2                20
Im3                57
SA1                11
SA2                12
SA3                11
Use1               14
Use2               17
Use3                9
Use4               23
Use5               15
Pf1                11
Pf2                 6
Pf3                14
JR1                27
JR2                53
BI1                32
BI2                43
Inc1               35
Inc2               35
Inc3               37
Inc4               42
Exp1               13
Exp2               11
Exp3               13
Exp4               14
Exp5               13
dtype: int64
Interpretation

We can see that there is a YEARSEXP column which contain 23 missing values. In the data set description, it is given that this column denotes number of years of university teaching experience and its data type is numeric. But, the df.info() command shows that it is of object data type. So, we need to change its data type.

Similarly, the last five columns Exp1, Exp2, Exp3, Exp4 and Exp5 denote the number of years of experience. They contain 13, 11, 13, 14 and 13 missing values respectively. They have numeric data types. But, the df.info() command shows that they are of object data types. So, we need to change their data types as well.

All the other columns are of text data types.

So, we need to subset these columns from the above dataset.

# Subset the dataframe df with above columns

df_sub = df[['YEARSEXP', 'Exp1', 'Exp2', 'Exp3', 'Exp4', 'Exp5']]
# Check the data types of columns of df_sub

print(df_sub.dtypes)
YEARSEXP    object
Exp1        object
Exp2        object
Exp3        object
Exp4        object
Exp5        object
dtype: object
Interpretation

We can see that the data type of columns of the dataframe of df_sub is object. We should convert it into integer data type.

# Convert columns of df_sub into integer data types

df_sub = df_sub.apply(pd.to_numeric)
print(df_sub.dtypes)
YEARSEXP    float64
Exp1        float64
Exp2        float64
Exp3        float64
Exp4        float64
Exp5        float64
dtype: object
Interpretation

We can see that all the columns of df_sub dataframe are converted to float64 numeric data types.

# View the summary of the dataframe df_sub

print(df_sub.info())
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 913 entries, 0 to 912
Data columns (total 6 columns):
YEARSEXP    890 non-null float64
Exp1        900 non-null float64
Exp2        902 non-null float64
Exp3        900 non-null float64
Exp4        899 non-null float64
Exp5        900 non-null float64
dtypes: float64(6)
memory usage: 42.9 KB
None
isna() and notna() functions to detect ‘NA’ values
Pandas provides isna() and notna() functions to detect ‘NA’ values. These are also methods on Series and DataFrame objects.

Examples of isna() and notna() commands

detect ‘NA’ values in the dataframe

df.isna()

detect ‘NA’ values in a particular column in the dataframe

pd.isna(df[col_name])

df[col_name].notna()

# View the number of missing values in each column of dataframe df_sub

print(df_sub.isnull().sum())
YEARSEXP    23
Exp1        13
Exp2        11
Exp3        13
Exp4        14
Exp5        13
dtype: int64
Interpretation

We can see that columns YEARSEXP, Exp1, Exp2, Exp3, Exp4 and Exp5 contain 23, 13, 11, 13, 14 and 13 missing values respectively.

Handle missing values
There are several methods to handle missing values. Each method has its own advantages and disadvantages. The choice of the method is subjective and depends on the nature of data and the missing values. The summary of the options available for handling missing values is given below:-

• Drop missing values with dropna()

• Fill missing values with a test statistic

• Fill missing values with Imputer

• Build a Prediction Model

• KNN Imputation

I have discussed each method in below sections:-

4. Drop missing values with dropna()
This is the easiest method to handle missing values. In this method, we drop labels or columns from a data set which refer to missing values.

drop labels or rows from a data set containing missing values

df.dropna (axis = 0)

drop columns from a data set containing missing values

df.dropna(axis = 1)

This is the Pandas dataframe dropna() method. An equivalent dropna() method is available for Series with same functionality.

To drop a specific column from the dataframe, we can use drop() method of Pandas dataframe.

drop col_name column from Pandas dataframe
df.drop(‘col_name’, axis = 1)

A note about axis parameter

Axis value may contain (0 or ‘index’) or (1 or ‘columns’). Its default value is 0.

We set axis = 0 or ‘index’ to drop rows which contain missing values.

We set axis = 1 or ‘columns’ to drop columns which contain missing values.

After dropping the missing values, we can again check for missing values and the dimensions of the dataframe.

again check the missing values in each column

df.isnull.sum()

again check the dimensions of the dataset

df.shape

But, this method has one disadvantage. It involves the risk of losing useful information. Suppose there are lots of missing values in our dataset. If drop them, we may end up throwing away valuable information along with the missing data. It is a very grave mistake as it involves losing key information. So, it is only advised when there are only few missing values in our dataset.

So, it's better to develop an imputation strategy so that we can impute missing values with the mean or the median of the row or column containing the missing values.

# Copy the dataframe df_sub

df1 = df_sub.copy()
# View the number of missing values in each column of dataframe df1

print(df1.isnull().sum())
YEARSEXP    23
Exp1        13
Exp2        11
Exp3        13
Exp4        14
Exp5        13
dtype: int64
Interpretation

The column Exp2 contain least number of missing values. So, I will drop that column from df1.

# Drop column Exp2 from df1

df1 = df1.drop('Exp2', axis = 1)
# View the first five rows of dataframe df1

print(df1.head())
   YEARSEXP  Exp1  Exp3  Exp4  Exp5
0      14.0   4.0   4.0   1.0   2.0
1      18.0   2.0   4.0   2.0   4.0
2      13.0   2.0   2.0   1.0   3.0
3      13.0   4.0   3.0   3.0   4.0
4       8.0   5.0   5.0   4.0   4.0
# View the summary of dataframe df1

print(df1.info())
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 913 entries, 0 to 912
Data columns (total 5 columns):
YEARSEXP    890 non-null float64
Exp1        900 non-null float64
Exp3        900 non-null float64
Exp4        899 non-null float64
Exp5        900 non-null float64
dtypes: float64(5)
memory usage: 35.7 KB
None
Conclusion

I have dropped the Exp2 column from the dataframe df1 with df1.drop() command.

5. Fill missing values with a test statistic
In this method, we fill the missing values with a test statistic like mean, median or mode of the particular feature the missing value belongs to. One can also specify a forward-fill or back-fill to propagate the next values backward or previous value forward.

Filling missing values with a test statistic like median

median = df['col_name'].median()

df['col_name'].fillna(value = median, inplace = True )

We can also use replace() in place of fillna()

df[‘col_name’].replace(to_replace = NaN, value = median, inplace = True)

If we choose this method, then we should compute the median value on the training set and use it to fill the missing values in the training set. Then we should save the median value that we have computed. Later, we will replace missing values in the test set with the median value to evaluate the system.

# Copy the df1 dataframe

df2 = df1.copy()
# View the number of missing values in each column of dataframe df2

print(df2.isnull().sum())
YEARSEXP    23
Exp1        13
Exp3        13
Exp4        14
Exp5        13
dtype: int64
Interpretation

We can see that the YEARSEXP column contain 23 missing values. I will fill missing values in YEARSEXP column with median of YEARSEXP column.

# Fill missing values in YEARSEXP column with median of YEARSEXP column.

median = df2['YEARSEXP'].median()

df2['YEARSEXP'].fillna(value = median, inplace = True)
# View the summary of df2 dataframe 

print(df2.info())
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 913 entries, 0 to 912
Data columns (total 5 columns):
YEARSEXP    913 non-null float64
Exp1        900 non-null float64
Exp3        900 non-null float64
Exp4        899 non-null float64
Exp5        900 non-null float64
dtypes: float64(5)
memory usage: 35.7 KB
None
# Again view the number of missing values in each column of dataframe df2

print(df2.isnull().sum())
YEARSEXP     0
Exp1        13
Exp3        13
Exp4        14
Exp5        13
dtype: int64
Interpretation

I have fill all the missing values of YEARSEXP column with the median value of YEARSEXP column. Now, this column has no missing values.

6. Fill missing values with Imputer
Scikit-Learn provides Imputer class to deal with the missing values. In this method, we replace the missing value with the mean value of the entire feature column. This can be done as shown in the following code:

from sklearn.preprocessing import Imputer

imp = Imputer(missing_values='NaN',  strategy='mean', axis=0)

imputed_data = imp.fit_transform(df)

imputed_data

Here, I have replaced each ‘NaN’ value with the corresponding mean value. The mean value is separately calculated for each feature column. If instead of axis = 0, we set axis = 1, then mean values are calculated for each row.

Other options for strategy parameter are ‘median’ or ‘most_frequent’. The ‘most_frequent’ parameter replaces the missing values with the most frequent value. It is useful for imputing categorical feature values.

# Fill missing values with Imputer

from sklearn.preprocessing import Imputer

imp = Imputer(missing_values='NaN',  strategy='mean', axis=0)

df2 = imp.fit_transform(df2)

print(df2)
[[14.  4.  4.  1.  2.]
 [18.  2.  4.  2.  4.]
 [13.  2.  2.  1.  3.]
 ...
 [ 9.  5.  5.  4.  1.]
 [10.  4.  2.  1.  1.]
 [12.  2.  3.  1.  1.]]
# Imputer convert the dataframe df2 into a numpy array.

# So, we need to convert it back into the dataframe df2.

columnnames = ['YEARSEXP', 'Exp1', 'Exp3', 'Exp4', 'Exp5']

df2 = pd.DataFrame(df2, columns = columnnames)
# View the first 5 rows of imputed dataframe df2

print(df2.head())
   YEARSEXP  Exp1  Exp3  Exp4  Exp5
0      14.0   4.0   4.0   1.0   2.0
1      18.0   2.0   4.0   2.0   4.0
2      13.0   2.0   2.0   1.0   3.0
3      13.0   4.0   3.0   3.0   4.0
4       8.0   5.0   5.0   4.0   4.0
# View the summary of the imputed dataframe df2

print(df2.info())
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 913 entries, 0 to 912
Data columns (total 5 columns):
YEARSEXP    913 non-null float64
Exp1        913 non-null float64
Exp3        913 non-null float64
Exp4        913 non-null float64
Exp5        913 non-null float64
dtypes: float64(5)
memory usage: 35.7 KB
None
# Agian check that there are no missing values in df2

print(df2.isnull().sum())
YEARSEXP    0
Exp1        0
Exp3        0
Exp4        0
Exp5        0
dtype: int64
Interpretation

We can see that there are no missing numerical values in the columns of dataframe df2.

7. Build a prediction model
We can build a prediction model to handle missing values. In this method, we divide our data set into two sets – training set and test set. Training set does not contain any missing values and test set contains missing values. The variable containing missing values can be treated as a target variable. Next, we create a model to predict target variable and use it to populate missing values of test data set.

8. KNN Imputation
In this method, the missing values of an attribute are imputed using the given number of attributes that are mostly similar to the attribute whose values are missing. The similarity of attributes is determined using a distance function.

The above two mmethods are more sophisticated methods to deal with missing numerical values. Hence, I will not go into much detail.

9. Check with ASSERT statement
Finally, we can check for missing values programmatically. If we drop or fill missing values, we expect no missing values. We can write an assert statement to verify this. So, we can use an assert statement to programmatically check that no missing or unexpected ‘0’ value is present. This gives confidence that our code is running properly. Assert statement will return nothing if the value being tested is true and will throw an AssertionError if the value is false.

Asserts

• assert 1 == 1 (return Nothing if the value is True)

• assert 1 == 2 (return AssertionError if the value is False)

assert that there are no missing values in the dataframe

assert pd.notnull(df).all().all()

assert that there are no missing values for a particular column in dataframe

assert df.column_name.notnull().all()

assert all values are greater than 0

assert (df >=0).all().all()

assert no entry in a column is equal to 0

assert (df['column_name']!=0).all().all()

# assert that there are no missing values in the dataframe df2

assert pd.notnull(df2).all().all()


# When I run the above command, it returns nothing. Hence the assert statement is true. 

# So, there are no missing values in dataframe df2.
# assert that there are no missing values for a particular column in the dataframe

assert df2['YEARSEXP'].notnull().all()

assert df2['Exp1'].notnull().all()

assert df2['Exp3'].notnull().all()

assert df2['Exp4'].notnull().all()

assert df2['Exp5'].notnull().all()
Interpretation

When I run the above commands, it returns nothing. Hence the assert statements are true. Hence, there are no missing values in df2 dataframe.

This concludes our discussion on missing values.
