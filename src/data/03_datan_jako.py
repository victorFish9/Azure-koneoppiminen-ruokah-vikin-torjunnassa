import pandas as pd
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
import os
import yaml
import shutil


ml_client = MLClient.from_config(credential=DefaultAzureCredential())
output_folder = './data'
os.makedirs(output_folder, exist_ok=True)


print("Ladataan 'ruokahavikki-data-valmis' (versio 3)...")
data_asset = ml_client.data.get("ruokahavikki-data-valmis", version="3")
df = pd.read_csv(data_asset.path)

df_original = pd.read_csv(ml_client.data.get("ruokahavikki-data-puhdistettu", version="1").path)
df['temp_date'] = pd.to_datetime(df_original['date'])

katkaisupaiva = "2024-11-01"
print(f"\nJaetaan data aikaperusteisesti. Katkaisupäivä: {katkaisupaiva}")

train_df = df[df['temp_date'] < katkaisupaiva].copy()
test_df = df[df['temp_date'] >= katkaisupaiva].copy()

train_df = train_df.drop(columns=['temp_date'])
test_df = test_df.drop(columns=['temp_date'])

print(f"Train-setin koko: {train_df.shape[0]} riviä ({len(train_df)/len(df):.1%})")
print(f"Test-setin koko:  {test_df.shape[0]} riviä ({len(test_df)/len(df):.1%})")


train_csv_path = os.path.join(output_folder, 'train_data.csv')
test_csv_path = os.path.join(output_folder, 'test_data.csv')
train_df.to_csv(train_csv_path, index=False)
test_df.to_csv(test_csv_path, index=False)


def create_mltable_folder(csv_path, folder_name):
    folder_path = os.path.join(output_folder, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    
    
    csv_filename = os.path.basename(csv_path)
    shutil.copyfile(csv_path, os.path.join(folder_path, csv_filename))
    
    
    mltable_def = {
        "paths": [{"file": f"./{csv_filename}"}],
        "transformations": [
            {"read_delimited": {"delimiter": ",", "header": "all_files_same_headers", "encoding": "utf8"}}
        ]
    }
    with open(os.path.join(folder_path, "MLTable"), "w") as f:
        yaml.dump(mltable_def, f)
        
    return folder_path

print("\nLuodaan MLTable-rakenteet...")
train_mltable_path = create_mltable_folder(train_csv_path, 'train_mltable')
test_mltable_path = create_mltable_folder(test_csv_path, 'test_mltable')


train_data = Data(
    name="ruokahavikki-train-table", 
    version="1",
    description="Opetusdata (MLTable) ennen 1.11.2024",
    path=train_mltable_path,
    type=AssetTypes.MLTABLE          
)

test_data = Data(
    name="ruokahavikki-test-table",  
    version="1",
    description="Testidata (MLTable) 1.11.2024 jälkeen",
    path=test_mltable_path,
    type=AssetTypes.MLTABLE          
)

ml_client.data.create_or_update(train_data)
ml_client.data.create_or_update(test_data)

print("\nSuorittu! Uudet MLTable-resurssit rekisteröity:")
print("- ruokahavikki-train-table")
print("- ruokahavikki-test-table")
