import requests
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv("USD_TRY Geçmiş Verileri.csv")

# Tarih sütununun tarih formatına çevrilmesi
df["Tarih"] = pd.to_datetime(df["Tarih"], format="%d.%m.%Y")

# Son 5 yıllık verilerin seçilmesi
end_date = pd.Timestamp("today")
start_date = end_date - pd.DateOffset(years=5)
df_last_five_years = df.loc[(df["Tarih"] >= start_date) & (df["Tarih"] <= end_date)]

# Günlük verilerin gösterilmesi
# print("Günlük dolar kuru verileri:")
# print(df_last_five_years)

# Aylık verilerin gösterilmesi
df_last_five_years_monthly = df_last_five_years.resample("M", on="Tarih").mean().reset_index()
# print("\nAylık dolar kuru verileri:")
# print(df_last_five_years_monthly)
print(df_last_five_years_monthly.columns)


# Son dört yılın aylık döviz kuru verilerini seçme
df_last_four_years_monthly = df_last_five_years_monthly.tail(48).copy()

# "Tarih" sütununu indeks olarak ayarlama
df_last_four_years_monthly.set_index("Tarih", inplace=True)


print(df_last_four_years_monthly)


print(df_last_four_years_monthly.columns)

from sklearn.ensemble import RandomForestRegressor

# "Tarih" sütununu yeniden biçimlendirme
df_last_four_years_monthly["Tarih"] = pd.to_datetime(df_last_four_years_monthly.index)

# Aylık döviz kuru verilerinde kullanılacak özellikleri seçme
X = df_last_four_years_monthly.index.month.values.reshape(-1, 1)

# Aylık döviz kuru verilerini seçme
y = df_last_four_years_monthly["Simdi"].values

# Random Forest Regressor Modeli oluşturma
reg = RandomForestRegressor().fit(X, y)

# Beşinci yılın aylık döviz kuru tahminlerini yapma
y_pred = reg.predict([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12]])

print(y_pred)

# Son yılın verilerini seçme
df_last_year = df_last_four_years_monthly.loc["2022-04-04":"2023-04-04"]

# Aylık döviz kuru verilerinde kullanılacak özellikleri seçme
X_last_year = df_last_year.index.month.values.reshape(-1, 1)

# Aylık döviz kuru verilerini seçme
y_last_year = df_last_year["Simdi"].values

# Beşinci yılın aylık döviz kuru tahminlerini yapma
y_pred_last_year = reg.predict(X_last_year)

# Gerçek ve tahmini değerleri karşılaştırma
for i in range(len(y_last_year)):
    print("Gerçek değer:", y_last_year[i], "Tahmini değer:", y_pred_last_year[i])


from sklearn.metrics import mean_squared_error

# Gerçek 5. yıl verilerini kaydetme
y_true = df_last_year["Simdi"].values

# MSE hesaplayarak doğruluk puanını bulma
mse = mean_squared_error(y_true, y_pred)
accuracy = 1 - mse / np.var(y_true)

print("Doğruluk Puanı:", accuracy)
