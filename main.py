# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from ls import LeastSquares

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
        
df = pd.read_csv("phase1_training_data.csv")
#print(df)
df_CA = df[df["country_id"] == "CA"]
deaths_CA = df_CA["deaths"].to_numpy()
cases_CA = df_CA["cases"].to_numpy()/10
N = len(deaths_CA)

ls = LeastSquares()
#ls.fit(y=deaths_CA)

plt.plot()
dDeaths = deaths_CA[1:] - deaths_CA[:-1]
dCases = cases_CA[1:] - cases_CA[:-1]
plt.scatter(x=list(range(len(dDeaths))), y=dDeaths, c="red")
plt.scatter(x=list(range(len(dCases))), y=dCases, c="green")
plt.show()

Z = np.concat()
"""
#print(dfCA)
CAdeaths = np.empty(len(dfCA["deaths"]-1))



plt.plot()
plt.scatter(x=list(range(len(dDeaths))), y=dDeaths)
plt.show()
"""

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session