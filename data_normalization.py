# importing packages 
import pandas as pd 

# create data 
df = pd.DataFrame([
    [180000, 110, 18.9, 1400], 
    [360000, 905, 23.4, 1800], 
    [230000, 230, 14.0, 1300], 
    [60000, 450, 13.5, 1500]], 

    columns = ['Col A', 'Col B', 'Col C', 'Col D'])

# view data
display(df)


import matplotlib.pyplot as plt 

df.plot(kind= 'bar')


# copy the data 
df_max_scaled = df.copy()

# apply normalization techniques 
for column in df_max_scaled.columns:
    df_max_scaled[column] = df_max_scaled[column] / df_max_scaled[column].abs().max()

# view normalized data 
display(df_max_scaled)

import matplotlib.pyplot as plt 
df_max_scaled.plot(kind='bar')

