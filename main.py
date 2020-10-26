# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from ls import LeastSquares
#from utils import test_and_plot3
from sklearn.metrics import mean_squared_error

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
        
df = pd.read_csv("phase2_training_data.csv")
#print(df)


countries = list(dict.fromkeys(df["country_id"]))
errors = {}

#Calculate absolute and squared error
def get_error(ypred):
    yreal = [23, 26, 11, 16, 28, 23, 5, 14, 27, 10, 35]
    #yreal = [25, 25, 25, 25, 25]
    e = 0
    ab = 0
    sq = 0
    return mean_squared_error(ypred, yreal)

preds = {}

StartDate = pd.Timestamp(year=2020, month=9, day=1, hour=0)
#df["date"] = pd.to_datetime(df["date"])

#Optimizing hyperparameter - country.
#Fit model based on each country in dataset, and use that to predict. Lowest error = best country for prediction
for country in countries:
    
    
    df_CA = df[df["country_id"] == country]
    
    #df_CA = df[df["date"] > StartDate]
    #df_CA = df_CA[245:]

    deaths_CA = df_CA["deaths"].to_numpy()
    cases_CA = df_CA["cases"].to_numpy()
    
    ls = LeastSquares()
    #ls.fit(y=deaths_CA)

    dDeaths = deaths_CA[1:] - deaths_CA[:-1]
    dCases = cases_CA[1:] - cases_CA[:-1]
    N = len(dDeaths)

    #auto_end = 11 and auto_start 26 gives best (so far) error = 15
    auto_start = -26
    auto_end = -11
    if N+auto_start < 0:
        continue
    Z = np.empty((N+auto_start, auto_end - auto_start))
    for i in range(-auto_start, N):
        Z[i+auto_start] = dCases[i+auto_start:i+auto_end]
    y = dDeaths[-auto_start:]

    Zpredict = np.empty((5, auto_end - auto_start))
    for i in range(N, N+5):
        Zpredict[i-N] = dCases[i+auto_start:i+auto_end]
        #if len(dCases[i+auto_start:i+auto_end]) >= 11:
        #    Zpredict[i-N] = dCases[i+auto_start:i+auto_end]
        #else:
            
    ls = LeastSquares()
    ls.fit(Z, y)
    ypred = ls.predict(Zpredict)
    preds[country] = ypred

"""
    e = get_error(ypred)
    print(country, e)
    errors[country] = e

minerror = min(errors.values())
for n in errors.keys():
    if errors[n] == minerror:
        print(n)


print(min(errors.values()))
"""
print(preds["PY"])
arr = preds["PY"]

sum = 9922
for val in arr:
    sum += val
    print(sum)

"""
#Figure out the minimum absolute, squared difference in error and the country it belongs to
min_abs = min([i[0] for i in errors.values()])
min_sq = min([i[1] for i in errors.values()])

country_abs = None
country_sq = None

for n in errors.keys():
    if errors[n][0] == min_abs:
        country_abs = n
    if errors[n][1] == min_sq:
        country_sq = n

print("Mininum absolute difference in error: {}, {}".format(country_abs, min_abs))
print("Mininum squared difference in error: {}, {}".format(country_sq, min_sq))
"""
#output = pd.DataFrame(errors)
#output.to_csv(r"C:/Users/Ethan Cho/Desktop/CPSC340/Midterm2020Fall/phase2>")
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session