#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


# In[2]:


# Load the CSV data
csv_file_path = 'C:\\Users\\91983\\Desktop\\assignment\\weather_data.csv'
weather_data = pd.read_csv(csv_file_path)


# In[3]:


# Display the first few rows of the data
print(weather_data.head())


# In[4]:


# Choose the features (X) and target variable (y)
features = weather_data[['Data.Precipitation', 'Date.Month','Date.Week of','Date.Year','Data.Wind.Speed','Data.Temperature.Avg Temp','Data.Temperature.Max Temp','Data.Temperature.Min Temp','Data.Wind.Direction']] 
target = weather_data['Data.Temperature.Avg Temp']  # Assuming 'Weather' is the target variable, change it accordingly


# In[5]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)


# In[6]:


# Create a linear regression model
model = LinearRegression()


# In[7]:


# Train the model
model.fit(X_train, y_train)


# In[8]:


# Make predictions on the test set
predictions = model.predict(X_test)


# In[9]:


# Display the actual vs predicted values
result_df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
print(result_df)


# In[10]:


# Plotting the results (for simplicity, considering only one feature)
plt.scatter(X_test.iloc[:,0], y_test, color='black', label='Actual')
plt.scatter(X_test.iloc[:,1], predictions, color='blue', label='Predicted')
plt.xlabel('Date.Year')
plt.ylabel('Data.Wind.Speed')
plt.legend()
plt.show()


# In[11]:


# Plotting the actual values
plt.bar(X_test['Date.Year'], y_test, color='purple', label='Actual', alpha=0.1)

# Plotting the predicted values
plt.bar(X_test['Date.Year'], predictions, color='yellow', label='Predicted', alpha=0.5)

plt.xlabel('Date.Year')
plt.ylabel('Data.Wind.Direction')
plt.title('Actual vs Predicted Wind Direction')
plt.legend()
plt.show()


# In[ ]:




