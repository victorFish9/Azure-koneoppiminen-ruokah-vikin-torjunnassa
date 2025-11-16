import pandas as pd
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
import os

ml_client = MLClient.from_config(credential=DefaultAzureCredential())
data_asset = ml_client.data.get("ruokahavikki-synteettinen-data", version="1")
df = pd.read_csv(data_asset.path)

print("data ladattu, alkuperäiset tietotyypit: ")
print(df.info())

print("\nKorjataan 'data' ja 'best_before_date' sarakkeiden tietotyypit..")
df['date'] = pd.to_datetime(df['date'])
df['best_before_date'] = pd.to_datetime(df['best_before_date'])

print("\nTarkistus korjausten jälkeen: ")
df.info()

output_folder = './data'
os.makedirs(output_folder, exist_ok=True)
cleaned_file_path = os.path.join(output_folder, 'synteettinen_myyntidata_puhdistettu.csv')

df.to_csv(cleaned_file_path, index=False, encoding="utf-8")
print(f"\nPuhdistettu data tallennettu väliaikaisesti polkuun: {cleaned_file_path}")

my_cleaned_data= Data(
    name="ruokahavikki-data-puhdistettu",
    version="1",
    description="Puhdistettu versio: Tietotyypit korjattu (date, best_before_date).",
    path=cleaned_file_path,
    type=AssetTypes.URI_FILE
)

ml_client.data.create_or_update(my_cleaned_data)

print("Puhdistettu data on nyt rekisteröity Azureen nimellä 'ruokahavikki-data-puhdistettu'.")
