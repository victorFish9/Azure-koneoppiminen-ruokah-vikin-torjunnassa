import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


try:
    ml_client = MLClient.from_config(credential=DefaultAzureCredential())
    print("Yhteys työtilaan (MLClient) muodostettu.")
except Exception as ex:
    print(f"VIRHE: Yhteyttä työtilaan ei saatu. {ex}")
    raise


try:
    train_data_asset = ml_client.data.get(name="ruokahavikki-train-table", version="1")
    test_data_asset = ml_client.data.get(name="ruokahavikki-test-table", version="1")
    print("Data-assetit 'ruokahavikki-train-table' ja 'ruokahavikki-test-table' löydetty.")
except Exception as ex:
    print(f"VIRHE: Data-assetteja ei löydetty nimellä. Tarkista nimet Azure ML Studiosta. {ex}")
    raise

try:
    print(f"Ladataan opetusdataa polusta: {train_data_asset.path}...")
    train_df = pd.read_csv(train_data_asset.path)
    
    print(f"Ladataan testidataa polusta: {test_data_asset.path}...")
    test_df = pd.read_csv(test_data_asset.path)


    print("Datan lataus Pandas DataFrame -muotoon onnistui.")
    print(f"Opetusdata ladattu: {train_df.shape[0]} riviä")
    print(f"Testidata ladattu: {test_df.shape[0]} riviä")

except Exception as ex:
    print(f"VIRHE: Datan lukeminen CSV-polusta epäonnistui. {ex}")
    print("Varmista, että compute-instanssillasi on tarvittavat kirjastot (esim. 'pip install pandas azureml-fsspec')")
    raise

