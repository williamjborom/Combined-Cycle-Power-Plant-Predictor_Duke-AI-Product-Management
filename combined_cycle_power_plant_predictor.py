import wget
import numpy as np
import os
from pathlib import Path

#Download Data
if not os.path.exists("CCPP_data.csv"):
    URL = "https://storfage.googleapis.com/aipi_datasets/CCPP_data.csv"
    data = wget.download(URL)
else:
    print("Data already downloaded")

#Read data to numpy array
data = np.loadtxt('CCPP_data.csv', delimiter=',', skiprows=1)
