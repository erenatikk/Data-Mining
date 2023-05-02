import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# Veri setini yükleme
df = pd.read_csv("USD_TRY Geçmiş Verileri.csv")

# Tarih sütununun tarih formatına çevrilmesi
df["Tarih"] = pd.to_datetime(df["Tarih"], format="%d.%m.%Y")

# Son 5 yıllık verilerin seçilmesi
end_date = pd.Timestamp("today")
start_date = end_date - pd.DateOffset(years=5)
df_last_five_years = df.loc[(df["Tarih"] >= start_date) & (df["Tarih"] <= end_date)]

# Aylık verilerin gösterilmesi
df_last_five_years_monthly = df_last_five_years.resample("M", on="Tarih").mean().reset_index()

# Son dört yılın aylık döviz kuru verilerini seçme
df_last_four_years_monthly = df_last_five_years_monthly.tail(48).copy()

# "Tarih" sütununu indeks olarak ayarlama
df_last_four_years_monthly.set_index("Tarih", inplace=True)

# LSTM modeli için girdi verilerinin hazırlanması
def prepare_data(data, lags=1):
    X, y = [], []
    for row in range(len(data)-lags-1):
        a = data[row:(row+lags), 0]
        X.append(a)
        y.append(data[row + lags, 0])
    return np.array(X), np.array(y)

# LSTM modeli oluşturma ve eğitme
lags = 12
data = df_last_four_years_monthly[['Simdi']].values
train_data = data[:len(data)-12]
test_data = data[len(data)-12:]

scaler = MinMaxScaler(feature_range=(0, 1))
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

X_train, y_train = prepare_data(train_data, lags)
X_test, y_test = prepare_data(test_data, lags)

X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))


model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(1, lags)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=2)

# Tahminlerin yapılması
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

train_predict = scaler.inverse_transform(train_predict)
y_train = scaler.inverse_transform([y_train])
test_predict = scaler.inverse_transform(test_predict)
y_test = scaler.inverse_transform([y_test])

# Gerçek ve tahmini değerleri karşılaştırma
for i in range(len(y_test[0])):
    print("Gerçek değer:", y_test[0][i], "Tahmini değer:", test_predict[i][0])

# MSE hesaplayarak doğruluk puanını bulma
mse = mean_squared_error(y_test[0], test_predict[:,0])
accuracy = 1 - mse / np.var(y_test[0])

print(accuracy)
