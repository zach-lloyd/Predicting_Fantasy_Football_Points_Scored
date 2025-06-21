import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(context='notebook', style='whitegrid')

df = pd.read_csv('combine_data/leviathan_plus_combine.csv')
# Replace NaNs in numeric columns with 0's. This is necessary 
# for this step because I want to be able to visualize how much data is
# missing. If I leave them as NaNs, they will just be ignored for purposes 
# of the histogram.
num_cols = df.select_dtypes('number').columns          
df[num_cols] = df[num_cols].fillna(0) 
# I replaced '40Yard' with the applicable names of the other features to
# produce the other histograms.
fig, ax = plt.subplots(figsize=(8,5))
sns.histplot(data=df, x='40Yard', bins=50, kde=True, ax=ax)
ax.set_title('Distribution of 40 Yard Dash Times')
plt.show()
