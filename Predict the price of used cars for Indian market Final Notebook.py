#!/usr/bin/env python
# coding: utf-8

# In[95]:


# Import required libraries
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# To ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Remove the limit from the number of displayed columns and rows. It helps to see the entire dataframe while printing it
pd.set_option("display.max_columns", None)


# In[96]:


data= pd.read_csv('used_cars.csv')
data


# In[97]:


# View first 5 rows
data.head()


# In[99]:


# View last 5 rows
data.tail()


# **Observations and Insights:
# 
# 1) We have total of 7253 observations in 14 columns. 2) In both head and tail operations new price columns seems to have no value. 3) Name column consist of Brand and name of the car, we might need to saperate as we can group the brand to impute some missing values in the dataset.

# In[100]:


# Check the datatypes of each column.
data.info()


# In[101]:


# Check total number of missing values of each column.

data.isnull().sum()


# **Observations and Insights:
# 
# 1) In the given dataset we have 5 columns which are object and the rest are either float or int which are numbers. 2) For the further analysis we might need to convert some data columns into some variables e.g., transmission is autoamtic and manual so we can give automatic as 0 and manual as 1 and use them in analysis. 3) For given dataset we have 6247 null or not available observations for new price column, for the price column there are 1234 null observations which are high in numbers, so during analysis we won't be able to remove those and those need to be treated either by mean, median or some sort of statistics. 4) We have no null values in name, location, year, KM driven, fuel type, transmission and owner type columns.

# We can observe that S.No. has no null values. Also the number of unique values are equal to the number of observations. So, S.No. looks like an index for the data entry and such a column would not be useful in providing any predictive power for our analysis. Hence, it can be dropped.

# In[102]:


# Remove S.No. column from data.
data.drop(columns=['S.No.'], inplace= True)
data.head()


# In[103]:


# Explore basic summary statistics of numeric variables.

data.describe().T


# **Observations and Insights:
# 
# 1) We only have year and kilometers driven columns where there is no null value. other column has some misisng values specially new price column. 2) We have cars manufactured from 1996 to 2019 and 75 percentage range from 1996 to 2016. 3) Kilometers driven column definately has some outliers as the maximum value is 6500000 which could be data entry error. 4) Mileage column has some 0 value as mileage won't be 0 for any given car. 5) Column New_price has more than half missing value, so to impute this we need better approach other than mean, median. 6) For the price column as 75 percent value lies within 0.44 to 9.95, as the maximum value is 160 meaning outliers due to data entry error.

# In[104]:


# Explore basic summary statistics of categorical variables. Hint: Use the argument include = ['object'] 

# cat_col= list(data.select_dtypes("object").columns) #make a list of all categorical columns
# data_col = pd.DataFrame(cat_col) # convet into dataframe
# data_col.describe() # not working


data.describe(include='object').T


# In[105]:


cat_cols = list(data.select_dtypes(include = ['object']).columns)

for column in cat_cols:
    
    print("For column:", column)
    
    print(data[column].value_counts())
    
    print('-'*50)


# **Observations and Insights:
# 1) There are 2041 unique names with Mahindra XUV500 W8 2WD on top of the list with frequency of 55. 2) Data collected in 11 different cities with Mumbai on top with 949 number of observations followed by Hydrabad, Coimbatore and Kochi etc. 3) Given dataset has 3852 observations recorded for Disel cars in fuel type column followed by pertol with 3325 observations. 4) We have 2 different types of transmission namely manuel with 5204 observations and automatic with 2049 observations. 5) Record of 5952 of first hand cars are collected in the dataset.

# In[106]:


# Sort the dataset in 'descending' order using the feature 'Kilometers_Driven'
data.sort_values(['Kilometers_Driven'], ascending= False).head(10)


# **Observations and Insights:
# 1) For the observation number 2328 we can see that 2017 BMW model drove 6500000 and still costs 6500000 for 2nd owner, so this is clearly data entry error. 2) The lastest is the model higher is the price even kilometers driven are too much.
# 
# 
# 
# In the first row, a car manufactured as recently as 2017 having been driven 6500000 km is almost impossible. It can be considered as data entry error and so we can remove this value/entry from data.

# In[107]:


# Removing the 'row' at index 2328 from the data. Hint: use the argument inplace=True
data.drop((2328), inplace=True)
data.shape


# In[108]:


# Sort the dataset in 'ascending' order using the feature 'Mileage'
data.sort_values(['Mileage'], ascending= True).head(82)


# Observations:
# Mileage of cars can not be 0, so we should treat 0's as missing values. We will do it in the Feature Engineering part.

# Univariate Analysis
# Univariate analysis is used to explore each variable in a data set, separately. It looks at the range of values, as well as the central tendency of the values. It can be done for both numerical and categorical variables.
# 
# 
# 1. Univariate Analysis - Numerical Data
# Histograms and box plots help to visualize and describe numerical data. We use box plot and histogram to analyse the numerical columns.

# In[109]:


# Let us write a function that will help us create a boxplot and histogram for any input numerical variable.
# This function takes the numerical column as the input and returns the boxplots and histograms for the variable.

def histogram_boxplot(feature, figsize = (15, 10), bins = None):
    
    """ Boxplot and histogram combined
    
    feature: 1-d feature array
    
    figsize: size of fig (default (9, 8))
    
    bins: number of bins (default None / auto)
    
    """
    f2, (ax_box2, ax_hist2) = plt.subplots(nrows = 2, # Number of rows of the subplot grid = 2
                                           sharex = True, # X-axis will be shared among all subplots
                                           gridspec_kw = {"height_ratios": (.25, .75)}, 
                                           figsize = figsize 
                                           ) # Creating the 2 subplots
    
    sns.boxplot(feature, ax = ax_box2, showmeans = True, color = 'violet') # Boxplot will be created and a symbol will indicate the mean value of the column
    
    sns.distplot(feature, kde = F, ax = ax_hist2, bins = bins, palette = "winter") if bins else sns.distplot(feature, kde = False, ax = ax_hist2) # For histogram
    
    ax_hist2.axvline(np.mean(feature), color = 'green', linestyle = '--') # Add mean to the histogram
    
    ax_hist2.axvline(np.median(feature), color = 'black', linestyle = '-') # Add median to the histogram


# In[110]:


# Plot histogram and box-plot for 'Kilometers_Driven'
histogram_boxplot(data['Kilometers_Driven'])


# In[112]:


# Log transformation of the feature 'Kilometers_Driven'
sns.distplot(np.log(data["Kilometers_Driven"]), axlabel = "Log(Kilometers_Driven)")


# **Observations and Insights: SO the data before log transformation is right skewed as there are many outliers present in the dataset, as after log transformation we reduce the skewness and made data look more normally distributed.

# In[113]:


# We can add a transformed kilometers_driven feature in data
data["kilometers_driven_log"] = np.log(data["Kilometers_Driven"])
(data.head())
# print(data.shape)


# Note: Like Kilometers_Driven, the distribution of Price is also highly skewed, we can use log transformation on this column to see if that helps normalize the distribution. And add the transformed variable into the dataset. You can name the variable as 'price_log'.

# In[114]:


# Plot histogram and box-plot for 'Price'

histogram_boxplot(data['Price'])


# In[115]:


# Log transformation of the feature 'Price'

sns.distplot(np.log(data['Price']), axlabel= "Price_log")


# In[116]:


# We can Add a transformed Price feature in data

data['Price_Log']= np.log(data['Price'])
data.head()


# *Observations and Insights for all the plots: As we applied log transformation to the column, it looks like the data in price_log is normally distributed and we can drop the original column in further analysis.

# 2. Univariate analysis - Categorical Data

# In[117]:


# Let us write a function that will help us create barplots that indicate the percentage for each category.
# This function takes the categorical column as the input and returns the barplots for the variable.

def perc_on_bar(z):
    '''
    plot
    feature: categorical feature
    the function won't work if a column is passed in hue parameter
    '''

    total = len(data[z]) # Length of the column
    
    plt.figure(figsize = (15, 5))
    
    ax = sns.countplot(data[z], palette = 'Paired', order = data[z].value_counts().index)
    
    for p in ax.patches:
        
        percentage = '{:.1f}%'.format(100 * p.get_height() / total) # Percentage of each class of the category
        
        x = p.get_x() + p.get_width() / 2 - 0.05 # Width of the plot
        
        y = p.get_y() + p.get_height()           # Hieght of the plot
        
        ax.annotate(percentage, (x, y), size = 12) # Annotate the percantage 
    
    plt.show() # Show the plot


# In[118]:


# Bar Plot for 'Location'

perc_on_bar('Location')


# In[119]:


# Bar Plot for 'Year'
perc_on_bar('Year')


# In[120]:


# Bar Plot for 'Fuel_Type'

perc_on_bar('Fuel_Type')


# In[121]:


# Bar Plot for 'Transmission'

perc_on_bar('Transmission')


# In[122]:


# Bar Plot for 'Owner_Type'

perc_on_bar('Owner_Type')


# **Observations and Insights from all plots:
# 
# 1) 13.1 percent data is collected in Mumbai which is the highest followed by Hydrabad with 12.1 and Coimbator, Kochi and pune with 10.6 each. Ahembdabad city is with least amount of data collected.
# 2) Almost 50 percent recorded cars are from year 2013 to 2016 and amlost 1 percent record from year 1996 to 2004 with no cars from year 1997. 3) 99 percent of the cars are driven either by disel or pertol, the 1 percent is driven by CNG or LPG and no automatic cars were recorded. 4) 72 percent of cars has Manual transmission and 28 are automatic. 5) There is only 18 percent of the owners are second or later, most people prefer to buy first hand car. But the problem we are solving is showing trend that switch from new buy to pre owned.

# In[123]:


data.head()


# ## **Bivariate Analysis**

# ## **Scatter Plot**

# In[125]:


# Let us plot pair plot for the variables 'year' and 'price_log'

plt.figure(figsize=(8,8))
sns.scatterplot(x = 'Price_Log', y = 'Year', data=data)


# In[126]:


# Scatter plot for the variable 'kilometers_driven_log' and 'Price_Log_Transformation'

plt.figure(figsize=(8,8))
sns.scatterplot(y='kilometers_driven_log', x= 'Price_Log', data=data)

#data.plot(y='kilometers_driven_log', x= 'Price_Log_Transformation')


# In[127]:


# scatter plot for variables 'Mileage' and 'Price_Log_Transformation'

plt.figure(figsize=(8,8))
sns.scatterplot(x='Price_Log', y= 'Mileage', data=data)


# In[128]:


# scatter plot for the variables 'Engine' and 'Price_Log_Transformation'

plt.figure(figsize=(10,10))
sns.scatterplot(x='Price_Log', y= 'Engine', data=data)


# In[129]:


# scatter plot for the variables 'Power' and 'Price_Log_Transformation'

plt.figure(figsize=(10,10))
sns.scatterplot(x='Price_Log', y='Power', data=data)


# In[130]:


# scatter plot for the variables 'Seats' and 'Price_Log_Transformation'

plt.figure(figsize=(10,10))
sns.scatterplot(x='Price_Log', y='Seats', data=data)


# **Observations and Insights from all plots:
# 
# 1) Price_log is low positively correlated with year and low negatively correlated with kilometers_driven_log. This information is found after we convert the actual columns with log. 2) There are outliers with mileage as 0 and needs to be taken care during feature engineering. But there is some relation between these variables. 3) Seats and price of car seems no relation. 4) Power and price is having positive correlation, ploynomial curve would best fit the data points in this case. 5) There is low positive correlation between engine size and price log. 6) Every graph has some outliers and can be treated or removed in further analysis.

# ### **2. Heat Map**

# In[132]:


# We can include the log transformation values and drop the original skewed data columns
plt.figure(figsize = (12, 7))

sns.heatmap(data.drop(['Kilometers_Driven', 'Price'],axis = 1).corr(), annot = True, vmin = -1, vmax = 1)

plt.show()


# **Observations and Insights:
# 1) Price_Log is highly positively correlated with Power, Engine, new price and year. and negatively correlated with milegae and kilometers driven.
# 2) Milegae decreses with increase in Engine size, power, number of seats and the price, so these variable are in negative correlation with mileage. We can see that newer the car better is the mileage meaning in low positive correlation.
# 3) With increase in power and price, Engine capacity increases. So the variable are highly positively correlated with Engine capacity but mileage would highly impacted with higher engine capacity.
# 4) The car with high power comes with high price.

# ### **3. Box Plot**

# In[134]:


# Let us write a function that will help us create boxplot w.r.t Price for any input categorical variable.
# This function takes the categorical column as the input and returns the boxplots for the variable.
def boxplot(z):
    
    plt.figure(figsize = (12, 5)) # Setting size of boxplot
    
    sns.boxplot(x = z, y = data['Price']) # Defining x and y
    
    plt.show()
    
    plt.figure(figsize = (12, 5))
    
    plt.title('Without Outliers')
    
    sns.boxplot(x = z, y = data['Price'], showfliers = False) # Turning off the outliers
    
    plt.show()


# In[135]:


# Box Plot: Price vs Location
boxplot(data['Location'])


# In[136]:


# Box Plot: Price vs Fuel_Type
boxplot(data['Fuel_Type'])


# In[137]:


# Box Plot: Price vs Transmission

boxplot(data['Transmission'])


# In[138]:


# Box Plot: Price vs Owner_Type

boxplot(data['Owner_Type'])


# **Observations and Insights for all plots:
# 
# 1) Coimbator and Banglore seem to have more expensive cars with median aorung 800000 in both the cities. 2) Pune, Chennai, Jaipur and Kolkata seems to be in similar price range of used cars. 3) Price of Electric cars looks higher than any other fuel type, most of the cars are disel based fuel with almost 75 percent fall within 2 lakhs to 17 lakhs. 4) As the price of automatic cars are higher than manual, explains why people choose manual transmission over automatic. 5) 75 percent of first hand car price is between 50 thousand to 10 lakhs with the average value of 6.5 lakhs, some cars are more expensive than others.

# ## **Feature Engineering**

# In[139]:


data['Brand']= data['Name'].apply(lambda x: x.split(" ")[0].lower())
data['Brand'].value_counts()
print(data['Brand'].nunique())


# In[140]:


data['Model']= data['Name'].apply(lambda x: x.split(" ")[1].lower())
data['Model'].value_counts()


# In[141]:


data.groupby(['Brand'])["Price"].mean().sort_values(ascending=False)


# ### **Missing value treatment**

# In[142]:


# Now check the missing values of each column. Hint: Use isnull() method

data.isnull().sum()


# **Observations and Insights:
# We have lot's of missing value specially for New_price column. We can use different methods like mean, median, or groupby to impute this missing values. We also have few columns which don't have any null values. So we can use those columns to fill the missing values in other columns.

# In[143]:


# Checking missing values in the column 'Seats'

data['Seats'].isnull().sum()


# In[144]:


# Impute missing values in Seats,you can use fillna method in pandas

data['Seats']= data.groupby(["Brand", "Model"])["Seats"].transform(lambda x: x.fillna(x.median()))


# In[145]:


# Now check total number of missing values of the seat column to verify if they are imputed or not. Hint: Use isnull() method

data['Seats'].isnull().sum()


# In[146]:


data['Seats']= data['Seats'].fillna(5.0)


# In[147]:


print(data['Seats'].nunique())

print(data['Seats'].isnull().sum())


# **Missing values for Mileage**

# In[148]:


# Now check missing values of each column. Hint: Use isnull() method

data['Mileage'].isnull().sum()


# In[149]:


# Impute missing Mileage. For example, use can use median or any other methods.

data['Mileage'].fillna(data['Mileage'].median(), inplace=True)


# In[150]:


# Now check total number of missing values of the Mileage column to verify if they are imputed or not. Hint: Use isnull() method

data['Mileage'].isnull().sum()


# In[151]:


data.isnull().sum()


# **Missing Values for Engine**

# In[152]:


#check missing values

data['Engine'].isnull().sum() 


# In[153]:


#replace the missing values with meadian

data['Engine'].fillna(data['Engine'].median(), inplace=True)


# In[154]:


#check if we successfully replaced the values or not

data['Engine'].isnull().sum() 


# **Missing values for Power**

# In[155]:


#check missing values

data['Power'].isnull().sum()


# In[156]:


#replace the missing values with meadian

data['Power'].fillna(data['Power'].median(), inplace=True)


# In[157]:


#check if we successfully replaced the values or not

data['Power'].isnull().sum()


# **Missing Value for New Price**

# In[158]:


#check missing values

data['New_price'].isnull().sum()


# In[159]:


#drop the column

data.drop(columns= ['New_price'], inplace=True)
data.isnull().sum()


# In[160]:


data.dropna(inplace=True)


# In[161]:


data.head()


# **Observations for missing values after imputing: Used median method to fill the na values for all the columns where we had missing values.

# In[162]:


data.isnull().sum()


# In[163]:


data.info()


# ### **Saving the Data**

# In[164]:


# Assume df_cleaned is the pre-processed data frame in your code, then
data.to_csv("cars_data_updated.csv", index = False)


# ## **Milestone 2**

# ## **Model Building**
# 
# 1. What we want to predict is the "Price". We will use the normalized version 'price_log' for modeling.
# 2. Before we proceed to the model, we'll have to encode categorical features. We will drop categorical features like Name. 
# 3. We'll split the data into train and test, to be able to evaluate the model that we build on the train data.
# 4. Build Regression models using train data.
# 5. Evaluate the model performance.

# **Note:** Please load the data frame that was saved in Milestone 1 here before separating the data, and then proceed to the next step in Milestone 2.

# In[1]:


# Library to split data
from sklearn.model_selection import train_test_split

# Import libraries for building linear regression model
from statsmodels.formula.api import ols
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

# Importing libraries for scaling the data
from sklearn.preprocessing import MinMaxScaler

# To ignore warnings
import warnings
warnings.filterwarnings('ignore')


# ### **Load the data**

# In[2]:


import pandas as pd
import numpy as np

cars_data = pd.read_csv("cars_data_updated.csv")
cars_data


# In[3]:


cars_data.info()


# In[4]:


cars_data.describe().T


# In[186]:


a= cars_data['Mileage'].sort_values(ascending=True)
pd.DataFrame(a.head(68))


# In[187]:


cars_data.iloc[262]


# In[188]:


cars_data.iloc[67]


# In[5]:


cars_data.describe(include='object').T


# ### **Split the Data**

# <li>Step1: Seperating the indepdent variables (X) and the dependent variable (y). 
# <li>Step2: Encode the categorical variables in X using pd.dummies.
# <li>Step3: Split the data into train and test using train_test_split.

# **Think about it:** Why we should drop 'Name','Price','price_log','Kilometers_Driven' from X before splitting?

# In[57]:


# Step-1
X = cars_data.drop(['Name','Price','Price_Log','Kilometers_Driven'], axis = 1)
X = pd.get_dummies(X, drop_first = True)
y = cars_data[["Price", "Price_Log"]]


# In[58]:


X


# In[7]:


pd.DataFrame(X.columns)


# In[8]:


pd.DataFrame(y.columns)


# In[9]:


# Step-2 Use pd.get_dummies(drop_first = True)
# X = pd.get_dummies(X, drop_first = True)


# In[59]:


# Step-3 Splitting data into training and test set:
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size = 0.3, random_state = 1)

print(X_train.shape, X_test.shape)


# In[60]:


X_train


# In[61]:


X_test


# In[62]:


# Let us write a function for calculating r2_score and RMSE on train and test data
# This function takes model as an input on which we have trained particular algorithm
# The categorical column as the input and returns the boxplots and histograms for the variable

def get_model_score(model, flag = True):
    '''
    model : regressor to predict values of X

    '''
    # Defining an empty list to store train and test results
    score_list = [] 
    
    pred_train = model.predict(X_train)
    
    pred_train_ = np.exp(pred_train)
    
    pred_test = model.predict(X_test)
    
    pred_test_ = np.exp(pred_test)
    
    train_r2 = metrics.r2_score(y_train['Price'], pred_train_)
    
    test_r2 = metrics.r2_score(y_test['Price'], pred_test_)
    
    train_rmse = metrics.mean_squared_error(y_train['Price'], pred_train_, squared = False)
    
    test_rmse = metrics.mean_squared_error(y_test['Price'], pred_test_, squared = False)
    
    # Adding all scores in the list
    score_list.extend((train_r2, test_r2, train_rmse, test_rmse))
    
    # If the flag is set to True then only the following print statements will be dispayed, the default value is True
    if flag == True: 
        
        print("R-sqaure on training set : ", metrics.r2_score(y_train['Price'], pred_train_))
        
        print("R-square on test set : ", metrics.r2_score(y_test['Price'], pred_test_))
        
        print("RMSE on training set : ", np.sqrt(metrics.mean_squared_error(y_train['Price'], pred_train_)))
        
        print("RMSE on test set : ", np.sqrt(metrics.mean_squared_error(y_test['Price'], pred_test_)))
    
    # Returning the list with train and test scores
    return score_list


# In[14]:


#Check the VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor

def checking_vif(train):
    vif= pd.DataFrame()
    vif['feature']= train.columns
    
    # Vif of each feature
    vif["VIF"] = [variance_inflation_factor(train.values, i) for i in range (len(train.columns))]
    return vif


# <hr>

# For Regression Problems, some of the algorithms used are :<br>
# 
# **1) Linear Regression** <br>
# **2) Ridge / Lasso Regression** <br>
# **3) Decision Trees** <br>
# **4) Random Forest** <br>

# ### **Fitting a linear model**

# Linear Regression can be implemented using: <br>
# 
# **1) Sklearn:** https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html <br>
# **2) Statsmodels:** https://www.statsmodels.org/stable/regression.html

# In[63]:


# Import Linear Regression from sklearn
from sklearn.linear_model import LinearRegression
import sklearn.metrics as metrics


# In[64]:


# Create a linear regression model
lr = LinearRegression()


# In[65]:


cars_data['Price_Log'].isnull().sum()


# In[66]:


# Fit linear regression model
lr.fit(X_train, y_train['Price_Log'])


# In[56]:


X.columns.value_counts()


# In[54]:


cars_data.isnull().sum()


# In[67]:


# Get score of the model
LR_score = get_model_score(lr)


# **Observations from results: 
# 1) R_Squared is measure of fit for linear regression models. And it measures the strengeth of the relationship between your model and the dependent variable. 
# 2) As we can see that r-square on test set is higher than train set. Model is performing well on both training and test dataset. 
# 3) RMSE is the square root of the  mean of squared differences between the actual outout and prediction, so lower the RMSE better is the performance of the model. For Linear regreesion we have lower RMSE for tarin and test set, so the model is performing well.

# **Important variables of Linear Regression**

# Building a model using statsmodels.

# In[21]:


# Import Statsmodels 
import statsmodels.api as sm

# Statsmodel api does not add a constant by default. We need to add it explicitly
x_train = sm.add_constant(X_train)

# Add constant to test data
x_test = sm.add_constant(X_test)

def build_ols_model(train):
    
    # Create the model
    olsmodel = sm.OLS(y_train["Price_Log"], train)
    
    return olsmodel.fit()


# Fit linear model on new dataset
olsmodel1 = build_ols_model(x_train)

print(olsmodel1.summary())


# In[22]:


# Retrive Coeff values, p-values and store them in the dataframe
olsmod = pd.DataFrame(olsmodel1.params, columns = ['coef'])

olsmod['pval'] = olsmodel1.pvalues

print(olsmod['pval'])

print(olsmod.shape)


# In[23]:


# Filter by significant p-value (pval <= 0.05) and sort descending by Odds ratio

olsmod = olsmod.sort_values(by = "pval", ascending = False)

pval_filter = olsmod['pval']<= 0.05

olsmod[pval_filter]


# In[24]:


# We are looking overall significant varaible

pval_filter = olsmod['pval']<= 0.05
mp_vars = olsmod[pval_filter].index.tolist()

# We are going to get overall varaibles (un-one-hot encoded varables) from categorical varaibles
sig_var = []
for col in mp_vars:
    if '' in col:
        first_part = col.split('_')[0]
        for c in cars_data.columns:
            if first_part in c and c not in sig_var :
                sig_var.append(c)

                
start = '\033[1m'
end = '\033[95m'
print(start+ 'Most overall significant varaibles of LINEAR REGRESSION  are ' +end,':\n', sig_var)


# **Build Ridge / Lasso Regression similar to Linear Regression:**<br>
# 
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html

# In[25]:


# Import Ridge/ Lasso Regression from sklearn
from sklearn.linear_model import Ridge


# In[26]:


# Create a Ridge regression model
ridge = Ridge()


# In[27]:


# Fit Ridge regression model
ridge.fit(X_train, y_train['Price_Log'])


# In[28]:


# Get score of the model
print('Score for Ridge regression is:', get_model_score(ridge))


# In[29]:


# Import Lasso Regression from sklearn
from sklearn.linear_model import Lasso


# In[30]:


# create model
lasso=Lasso()


# In[31]:


# fit the model
lasso.fit(X_train, y_train['Price_Log'])


# In[32]:


# get the score
print('Score for Lasso regression is:', get_model_score(lasso))


# In[166]:


ridge.n_features_in_


# **Observations from results:
# 1) So the ridge regression model is performing similar to linear regression as the r-square and RMSE for ridge regression for train set are lower than test set. 
# 2) Meaning that ridge could be best model to use as final model.
# 3) Lasso in other case is performing bad, in this model we are adding unnecessary features. This would be interesting if we can remove some of the unnecessary features from the model.

# ### **Decision Tree** 
# 
# https://scikit-learn.org/stable/auto_examples/tree/plot_tree_regression.html

# In[33]:


# Import Decision tree for Regression from sklearn
from sklearn.tree import DecisionTreeRegressor


# In[34]:


# Create a decision tree regression model, use random_state = 1
dtree = DecisionTreeRegressor(random_state=1)


# In[35]:


# Fit decision tree regression model
dtree.fit(X_train, y_train['Price_Log'])


# In[36]:


# Get score of the model
Dtree_model = get_model_score(dtree)


# **Observations from results: 
# 1) Desicion tree is performing well on training data and testing data, but the r-square on training data is almost 100 percent so the model could be overfitting.
# 2) Even the RMSE for the training data is far less than test data so model is performing well but need to check for overfitting issue.
# 3) There is noise in the dataset.

# Print the importance of features in the tree building. The importance of a feature is computed as the (normalized) total reduction of the criterion brought by that feature. It is also known as the Gini importance.
# 

# In[37]:


print(pd.DataFrame(dtree.feature_importances_, columns = ["Imp"], 
                   index = X_train.columns).sort_values(by = 'Imp', ascending = False))
imp_features= pd.DataFrame(dtree.feature_importances_, columns = ["Imp"], 
                   index = X_train.columns).sort_values(by = 'Imp', ascending = False)


# **Observations and insights: 
# 1) The most important feature for predicting price is the power, as the split of data using power led to the maximum reduction of RSS, following year and engine.
# 2) SO the decision node of the tree is power. 
# 3) Here most of the information is gained form power, year and engine variable respectively.

# ### **Random Forest**
# 
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html

# In[38]:


# Import Randomforest for Regression from sklearn
from sklearn.ensemble import RandomForestRegressor


# In[39]:


# Create a Randomforest regression model 
r_forest = RandomForestRegressor()


# In[40]:


# Fit Randomforest regression model
r_forest.fit(X_train, y_train['Price_Log'])


# In[41]:


# Get score of the model
get_model_score(r_forest)


# **Observations and insights: 
# 1) Random forest model works well on train and test data. As we have higher r square for training set, this model is overfitting.
# 2) R square for train is higher than test and rmse for train is lower than test.

# **Feature Importance**

# In[42]:


# Print important features similar to decision trees
print(pd.DataFrame(r_forest.feature_importances_, columns = ["Imp"],
                   index = X_train.columns).sort_values(by = 'Imp', ascending = False))


# **Observations and insights:
# 1) The most important feature for predicting price is the power, as the data follows bootstrap samples, followed by year and engine. 
# 2) After tarining various models and taking average, power is the best feature varible for predicting price.

# ### **Hyperparameter Tuning: Decision Tree**

# In[43]:


from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve,recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer,mean_squared_error, r2_score, mean_absolute_error

# Choose the type of estimator 
dtree_tuned = DecisionTreeRegressor(random_state = 1)

# Grid of parameters to choose from
# Check documentation for all the parametrs that the model takes and play with those
parameters = {'criterion':['squared_error', 'friedman_mse'],
    'max_depth':np.arange(1,5),
    'min_samples_split':[2, 4, 6, 8],
    'min_samples_leaf':[2, 5, 7]}

# Type of scoring used to compare parameter combinations
scorer = metrics.make_scorer(r2_score)

# Run the grid search
grid_obj = GridSearchCV(dtree_tuned, parameters, scoring = scorer, cv = 10)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the model to the best combination of parameters
dtree_tuned = grid_obj.best_estimator_

# Fit the best algorithm to the data
dtree_tuned.fit(X_train, y_train['Price_Log'])


# In[44]:


# Get score of the dtree_tuned
get_model_score(dtree_tuned)


# In[45]:


grid_obj.best_params_


# **Observations and insights: 
# 1) Model is performing well on both training and test data. 
# 2) If we increase the maximum depth params then the model start overfitting.

# **Feature Importance**

# In[46]:


# Print important features of tuned decision tree similar to decision trees
print(pd.DataFrame(dtree_tuned.feature_importances_, columns=['IMP'], 
            index= X_train.columns).sort_values(by= 'IMP', ascending = False))


# **Observations and insights: 
# 1) Even after tuning power in the most important feature followed by year and engine.
# 2) In decision tree other variables were having some importance byt during tuning the importance cut down to only three feature variables.
# 3) The rest of the variables have no impact in this model.

# ### **Hyperparameter Tuning: Random Forest**

# In[47]:


# Choose the type of Regressor
Rforest_tuned= RandomForestRegressor(random_state=1)

# Define the parameters for Grid to choose from
# Check documentation for all the parametrs that the model takes and play with those
parameters = {'n_estimators': [110, 120], 
            'max_depth' : [5, 7], 
            'max_features': [0.8, 1]}


# Type of scoring used to compare parameter combinations
scorer = make_scorer(r2_score)

# Run the grid search
grid_obj = GridSearchCV(Rforest_tuned, parameters, scoring=scorer, cv=5)

grid_obj = grid_obj.fit(X_train, y_train)

# Set the model to the best combination of parameters
rf_tuned_regressor = grid_obj.best_estimator_


# Fit the best algorithm to the data
Rforest_tuned.fit(X_train, y_train['Price_Log'])


# In[48]:


# Get score of the model
get_model_score(Rforest_tuned)


# In[49]:


grid_obj.best_params_


# **Observations and insights: 
# 1) Model is slightly overfitting as r square on train data is higher than test data, but it does very good on test and capturing 85 percent of variation.

# **Feature Importance**

# In[50]:


# Print important features of tuned decision tree similar to decision trees
print(pd.DataFrame(Rforest_tuned.feature_importances_, columns=['IMP'], 
            index= X_train.columns).sort_values(by='IMP', ascending=False))


# **Observations and insights: 
# 1) After tuning random forest power and year are the most important feature variables. 
# 2) Variables like Engine, Mileage and kilometers_driven_log are important but significantly less than that of power or year. 

# **Observations: 
# 1) Decision tree, Random forest and tuned decision tree and random forest give good performance in terms of RMSE and R-squared when compared to linear, ridge and lasso regression.
# 2) The normal Decision tree is performing better than tuned decision tree.
# 3) Surprisingly no tuned affect the tuned random forest as it give exact the same results as normal random forest.
# 4) Decision tree is the best model thus far followed by random forest.

# ### **KNN**

# In[71]:


# KNN
# import libraries
from sklearn import neighbors
from math import sqrt
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[84]:


#create a model
KNN = neighbors.KNeighborsRegressor(n_neighbors=20)


# In[85]:


#fit it
KNN.fit(X_train, y_train['Price_Log'])


# In[86]:


#get model score
get_model_score(KNN)


# In[94]:


from sklearn.model_selection import GridSearchCV
params = {'n_neighbors':np.arange(1,30)}

knn = neighbors.KNeighborsRegressor()

model = GridSearchCV(knn, params, cv=5)
model.fit(X_train,y_train['Price_Log'])
model.best_params_


# In[91]:


New_KNN = neighbors.KNeighborsRegressor(n_neighbors=4)
New_KNN.fit(X_train, y_train['Price_Log'])

get_model_score(New_KNN)


# In[92]:


# Defining list of models you have trained
models = [lr, ridge, lasso, dtree, r_forest, dtree_tuned, Rforest_tuned, KNN, New_KNN]

# Defining empty lists to add train and test results
r2_train = []
r2_test = []
rmse_train = []
rmse_test = []

# Looping through all the models to get the rmse and r2 scores
for model in models:
    
    # Accuracy score
    j = get_model_score(model, False)
    
    r2_train.append(j[0])
    
    r2_test.append(j[1])
    
    rmse_train.append(j[2])
    
    rmse_test.append(j[3])


# In[93]:


comparison_frame = pd.DataFrame({'Model':['Linear Regression', 'Ridge', 'Lasso', 'Decision Tree', 'Random Forest',
                                          'Hyperparameter Tuning: Decision Tree', 'Hyperparameter Tuning: Random Forest', 'K-Nearest Neighbors'
                                         , 'New KNN'], 
                                          'Train_r2': r2_train,'Test_r2': r2_test,
                                          'Train_RMSE': rmse_train,'Test_RMSE': rmse_test}) 
comparison_frame


# ### **Insights**
# 
# **Refined insights**:
# - What are the most meaningful insights from the data relevant to the problem?
# 1)	After the Exploratory data analysis and feature engineering we have numerical as well as categorical features in the dataset with 6018 observations in 16 columns. 
# 2)	There are still 68 observations in the dataset with 0 mileage, so further data processing includes removing older cars with 0 mileage.
# 3)	We used function to get dummy variables from the categorical dataset which converts categorical features into numbers.
# 4)	The processed dataset has now 263 features with 6018 columns. Here we removed original columns for Kilometer_Driven and Price. As the dataset in these columns get scaled and we transformed using log function.
# 5)	Power, Engine and Year are the most significant variables in the dataset which are responsible to capture most variance in the dataset.
# 
# 
# **Comparison of various techniques and their relative performance**:
# - How do different techniques perform? Which one is performing relatively better? Is there scope to improve the performance further?
# 1)	For given dataset we created 8 different model. These models differ from one another on the basis of algorithms used, parameters given and unique tendency to relate to the feature variables.
# 2)	Ridge regression is the best model followed by Linear regression. In ridge regression the model is capable of capturing 94 percent of variation. 
# 3)	Decision tree overfits the data, this means that this model is unpredictable and cannot use for prediction. There is noise in the model.
# 4)	Tuned hyperparameters for decision tree and random forest gives best values by changing in parameters, so those models can underfit, overfit or could be best models.  
# 5)	There is considerable scope for improving the models, keeping in mind that mileage won’t be 0. So, after removing those 68 observations we can have better model. 
# 
# **Proposal for the final solution design**:
# - What model do you propose to be adopted? Why is this the best solution to adopt?
# 1)	Ridge regression is the model to go forward with as it captures greater variation in relative features and balanced error. 
# 2)	This model captures multicollinearity issue as well which can be seen in linear regression. 
# 3)	Ridge regression is most consistent across both train and test sets. 
# 

# ### **THANK YOU**
