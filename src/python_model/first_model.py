import mltable
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

train_data_asset = mltable.load("azureml:ruokahavikki-train-table:1")
df_train = train_data_asset.to_pandas_dataframe()

test_data_asset = mltable.load("azureml:ruokahavikki-test-table:1")
df_test = test_data_asset.to_pandas_dataframe()

print("Opetusdatan koko:", df_train.shape)
print("Testidatan koko:", df_test.shape)

target = "waste_qty"

X_train = df_train.drop([target], axis=1)
Y_train = df_train[target]

X_test = df_test.drop([target], axis=1)
Y_test = df_test[target]

print("Koulutusmuuttujat: ", X_train.columns.tolist())

lr_model = LinearRegression()

lr_model.fit(X_train, Y_train)

lr_pred = lr_model.predict(X_test)

dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, Y_train)
dt_pred = dt_model.predict(X_test)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, Y_train)
rf_pred = rf_model.predict(X_test)


def evaluate_model(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"--- {name} ---")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}\n")

evaluate_model("Lineaarinen Regressio", Y_test, lr_pred)
evaluate_model("Päätöspuu", Y_test, dt_pred)
evaluate_model("Random Forest", Y_test, rf_pred)
