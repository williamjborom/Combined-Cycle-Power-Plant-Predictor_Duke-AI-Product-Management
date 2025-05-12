import wget

URL = "https://storage.googleapis.com/aipi_datasets/CCPP_data.csv"
data = wget.download(URL)
print(data)