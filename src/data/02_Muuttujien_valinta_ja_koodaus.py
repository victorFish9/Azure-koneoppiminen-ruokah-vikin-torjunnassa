import pandas as pd
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
import os


ml_client = MLClient.from_config(credential=DefaultAzureCredential())
output_folder = './data'
os.makedirs(output_folder, exist_ok=True)


print("Ladataan 'ruokahavikki-data-puhdistettu' Azuresta...")
data_asset = ml_client.data.get("ruokahavikki-data-puhdistettu", version="1")
df = pd.read_csv(data_asset.path)

df['date'] = pd.to_datetime(df['date'])
df['best_before_date'] = pd.to_datetime(df['best_before_date'])
print(f"Data ladattu. Rivit: {df.shape[0]}, Sarakkeet: {df.shape[1]}")

print("\nLuodaan uusia piirteitä...")

df['viikonpaiva'] = df['date'].dt.dayofweek
df['kuukausi'] = df['date'].dt.month
df['onko_viikonloppu'] = df['viikonpaiva'].apply(lambda x: 1 if x >= 5 else 0)

df['jaljella_oleva_aika'] = (df['best_before_date'] - df['date']).dt.days

print("Uudet piirteet luotu: viikonpaiva, kuukausi, onko_viikonloppu, jaljella_oleva_aika")

print("\nLuodaan kategoriset teksti-muuttujat numeroiksi (One-Hot Encoding)...")

koodattavat_sarakkeet = ['store_id', 'category', 'unit_size']
df_encoded = pd.get_dummies(df, columns=koodattavat_sarakkeet, drop_first=False)

poistettavat = ['date', 'best_before_date', 'sku']

df_final = df_encoded.drop(columns=poistettavat, errors='ignore')

print(f"Datan muokkaus valmis! Uudet mitat: Rivit {df_final.shape[0]}, Sarakkeet {df_final.shape[1]}")
print("Ensimmäiset 5 saraketta:", df_final.columns.tolist()[:5])

final_file_path = os.path.join(output_folder, 'synteettinen_data_features.csv')
df_final.to_csv(final_file_path, index=False, encoding="utf-8")

my_final_data = Data(
    name="ruokahavikki-data-valmis",
    version="1",
    description="Malli-valmis data: Puhdistettu, uudet piirteet luotu, koodattu numeroiksi.",
    path=final_file_path,
    type=AssetTypes.URI_FILE
)

ml_client.data.create_or_update(my_final_data)
print("\nVALMIS! Data rekisteröity Azureen nimellä: 'ruokahavikki-data-valmis'")
