#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
df = pd.read_csv("C:/Users/gaura/Downloads/Gravity_csv_V202211/Gravity_V202211.csv")


# In[2]:


# List of countries you want to keep
countries_of_interest = ['USA', 'SAU', 'ARE', 'IRQ', 'QAT', 'IND', 'BHR', 'MAR', 'JOR', 'ISR', 'OMN']

# Filter the dataset to include only the desired countries
filtered_df = df[df['iso3_d'].isin(countries_of_interest) & df['iso3_o'].isin(countries_of_interest) & (df['year']==2016)]


# In[32]:


filtered_df


# In[3]:


import os

# Define the path to your desired directory
desired_directory = "C:/Users/gaura/Downloads/412 final ppt"

# Change the current working directory to your desired directory
os.chdir(desired_directory)


# In[4]:


filtered_df1 = filtered_df.copy()

mefta_countries = ['BHR', 'MAR', 'JOR', 'ISR', 'OMN', 'USA']

# Create a new column 'New_Column' based on the condition
filtered_df1['FTA'] = filtered_df1.apply(lambda row: 1 if row['iso3_d'] in mefta_countries and row['iso3_o'] in mefta_countries else 0, axis=1)


# In[5]:


filtered_df1.to_csv('init.csv', index=False)


# In[8]:


pip install gegravity


# In[1]:


import gegravity as ge
import pandas as pd 
# Increase number of columns printed for a pandas DataFrame
pd.set_option("display.max_columns", None)
pd.set_option('display.width', 1000)
import gme as gme


# In[4]:


gravity_data_location = "init.csv"
grav_data = pd.read_csv(gravity_data_location)
# grav_data_cleaned = grav_data.dropna()

# grav_data.to_csv('cleaned_data.csv', index=False)

# print(grav_data.head())


# In[5]:


grav_data['trade'] = (grav_data['tradeflow_imf_o'] + grav_data['tradeflow_imf_o'] )/2


# In[6]:


import numpy as np
grav_data['logdist'] = np.log(grav_data['dist'])


# In[10]:


# grav_data['international'] = (grav_data['iso3_o'] == grav_data['iso3_d']).astype(int)
grav_data['intl'] = np.where(grav_data['iso3_o'] == grav_data['iso3_d'], 0, 1)


# In[7]:


gme_data = gme.EstimationData(grav_data, # Dataset
                              imp_var_name="iso3_d", # Importer column name
                              exp_var_name="iso3_o", # Exporter column name
                              year_var_name = "year",  # Year column name
                              trade_var_name="trade")  # Trade column name


# In[8]:


gme_model = gme.EstimationModel(gme_data, 
                                lhs_var="trade",                               
                                rhs_var=[ "contig", "gdp_o", "gdp_d",
                                         "logdist", "FTA"],
                                fixed_effects=[["iso3_o"],["iso3_d"]])     # Fixed effects to use


# In[9]:


gme_model.estimate()


# In[10]:


(gme_model.results_dict['all']).summary()


# In[11]:


ge_model = ge.OneSectorGE(gme_model,                   # gme gravity model
                       year = "2016",               # Year to use for model
                       expend_var_name = "gdp_d",       # Expenditure column name
                       output_var_name = "gdp_o",       # Output column name
                       reference_importer = "IND",  # Reference importer
                       sigma = 5)                   # Elasticity of substitution


# In[12]:


rescale_eval = ge_model.check_omr_rescale(omr_rescale_range=3)
print(rescale_eval)


# In[13]:


ge_model.build_baseline(omr_rescale=0.1)
# Examine the solutions for the baselin multilateral resistances
print(ge_model.baseline_mr.head(12))


# In[14]:


exp_data = ge_model.baseline_data.copy()
exp_data["FTA"] = 1


# In[15]:


ge_model.define_experiment(exp_data)
# Examine the baseline and counterfactual trade costs
# print(ge_model.bilateral_costs.head(20))


# In[16]:


ge_model.simulate()


# In[17]:


country_results = ge_model.country_results

print(country_results[['factory gate price change (percent)', 'GDP change (percent)',
                       'foreign exports change (percent)']])


# In[18]:


# The bilateral trade results
bilateral_results = ge_model.bilateral_trade_results

print(bilateral_results)


# In[19]:


agg_trade = ge_model.aggregate_trade_results


# country multilateral resistance (MR) terms
mr_terms = ge_model.country_mr_terms
# Get the solver diaganoistics, which is a dictionary containing many types of solver diagnostic info
solver_diagnostics = ge_model.solver_diagnostics

agg_trade


# In[20]:


mr_terms = ge_model.country_mr_terms
print(mr_terms)


# In[22]:


ge_model.export_results(directory="C:/Users/gaura/Downloads/412 final ppt",name="412final", include_levels = True)


# In[27]:


ge_model.bilateral_costs


# In[ ]:




