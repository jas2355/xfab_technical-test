#!/usr/bin/env python
# coding: utf-8

# In[97]:


# Python at Jupyter Notebook
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


# In[115]:


#read csv file
data = pd.read_csv('C:/Users/User/Downloads/xfab technical test/Tool_Sensor_Data.csv')
data.head()
print(data.shape)


# In[116]:


# 1.1 Data Preprocessing
# 1.1.1 Data Cleaning
# check missing values
print(data.isna().sum())


# In[117]:


# drop the columns with missing values
data = data.dropna(axis=1, how='all')

# test again the total missing values
print(data.isna().sum())
print(data.shape)


# In[118]:


# find columns that contain more than 80% of zero
zero_columns = data.columns[(data == 0).mean() > 0.8]
print(zero_columns)


# In[119]:


# Drop columns containing all zeros
data.drop(zero_columns, axis=1, inplace=True)
data.head()
print(data.shape)


# In[120]:


# Remove duplicate rows
data.drop_duplicates(inplace=True)
print(data.shape)


# In[121]:


# remove specific columns
data.drop(columns=['EventType', 'EventName', 'EventId','RunStartTime'], inplace=True)
print(data.shape)


# In[122]:


# Drop rows with NaN values
data.dropna(inplace=True)
print(data.shape)


# In[123]:


# check missing values again
print('Total missing values:', data.isna().sum().sum())


# In[124]:


# check whether there are any blank space in the dataset
blank_spaces = data.isnull().values.any()

if blank_spaces:
    print("Blank spaces exist.")
else:
    print("No blank spaces found.")


# In[125]:


# Select rows where values in the specified columns are non-negative
tool_sensor_data_cleaned = data.loc[(data[['YiAwOaAhwskZcEfg','tMqophNywoUtXsGZAeVHBvtFjuyM']] >= 0).all(axis=1)]
print(tool_sensor_data_cleaned.shape)


# In[126]:


tool_sensor_data_cleaned.head()


# In[127]:


# Convert to datetime format
tool_sensor_data_cleaned['TimeStamp'] = pd.to_datetime(tool_sensor_data_cleaned['TimeStamp'], format='%d/%m/%Y %H:%M')

# Extract date
tool_sensor_data_cleaned['new_timestamp'] = tool_sensor_data_cleaned['TimeStamp'].dt.date

# Convert 'Date' column to datetime format
tool_sensor_data_cleaned['new_timestamp'] = pd.to_datetime(tool_sensor_data_cleaned['new_timestamp'])

# Drop the original 'timestamp' column
tool_sensor_data_cleaned.drop(columns=['TimeStamp'], inplace=True)


# In[128]:


tool_sensor_data_cleaned.head()


# In[129]:


tool_sensor_data_cleaned.info()


# In[256]:


# save the cleaned data in csv file for future use
tool_sensor_data_cleaned.to_csv('tool_sensor_data_cleaned.csv', index=False)


# In[136]:


# 1.2 EDA
#Separate the numerical and categorical variables
# Identify numeric data
numeric_data = tool_sensor_data_cleaned.select_dtypes(include=['float64', 'int64'])

# Identify category data
category_data = tool_sensor_data_cleaned.select_dtypes(include=['object'])


# In[131]:


# 1.2.1 Standardization
# Scale only the numeric columns
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)

# Convert scaled_data back to DataFrame
scaled_df = pd.DataFrame(scaled_data, columns=numeric_data.columns)


# In[132]:


# 1.2.2 Summary Statistics
selected_sensors = ['OunhHslCRwIRilo','BmpcKiosIw','SwpYipezsdueC','ArsbiQzICA','ETcatZBXS']

# create summary stats with selected sensor column
sum_stats = scaled_df[selected_sensors].describe()
sum_stats.loc['range'] = sum_stats.loc['max'] - sum_stats.loc['min']
sum_stats.loc['IQR'] = sum_stats.loc['75%'] - sum_stats.loc['25%']
sum_stats


# In[133]:


# 1.2.3 Histograms
# Generate a histogram for each selected column
for column in selected_sensors:
    plt.figure(figsize=(6, 4))
    numeric_data[column].hist(bins=20)
    plt.title(f'Histogram for {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show() 


# In[242]:


# 1.2.4 Boxplot - to detect outliers
# Generate a box plot for each selected column
import seaborn as sns

for column in selected_sensors:
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=numeric_data, y=column)
    plt.title(f'Box Plot for {column}')
    plt.ylabel('Values')
    plt.show()


# In[55]:


# 1.2.5 Correlation
# correlation between different sensors
correlation_matrix = numeric_data[selected_sensors].corr()
correlation_matrix


# In[234]:


# 1.2.6 Heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Sensors')
plt.show()


# In[145]:


#calculate the average sensor readings and group by day for line graph
sensor_daily_avg = tool_sensor_data_cleaned.groupby('new_timestamp')[selected_sensors].mean().reset_index()
sensor_daily_avg


# In[146]:


# 1.2.7 Line Graph
for column in selected_sensors:
    plt.figure(figsize=(6, 4))
    plt.plot(sensor_daily_avg['new_timestamp'], sensor_daily_avg[column], marker='o', linestyle='-', label=column)
    plt.title(f'Line Graph for {column}')
    plt.xlabel('Date')
    plt.ylabel(column)
    plt.show()


# In[150]:


for column in sensor_daily_avg.columns:
    if column != 'new_timestamp':
        plt.plot(sensor_daily_avg['new_timestamp'], sensor_daily_avg[column], label=column)

plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Line Graph of Multiple Sensors')
plt.legend()
plt.show()


# In[ ]:




