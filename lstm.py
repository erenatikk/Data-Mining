import pandas as pd
import numpy as np

# Veri setini okuyalım
df = pd.read_csv("USD_TRY Geçmiş Verileri full.csv")

# "Simdi" sütununu "Close" olarak değiştirelim
df.rename(columns={"Simdi": "Close"}, inplace=True)

# Tarih sütununu indeks olarak ayarlayalım
df.set_index("Tarih", inplace=True)

# Veri setini gösterelim
print(df.head())



from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Veri setini numpy dizisine dönüştürelim
data = df.values

# Veriyi normalize edelim
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Eğitim ve test verilerini ayıralım
train_size = int(len(scaled_data) * 0.8)
test_size = len(scaled_data) - train_size
train_data, test_data = scaled_data[0:train_size, :], scaled_data[train_size:len(scaled_data), :]

# Verileri LSTM modeli için uygun hale getirelim
def create_dataset(dataset, time_step=1):
    X, y = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        X.append(a)
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Verilerin boyutlarını LSTM modeli için yeniden şekillendirelim
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# LSTM modelini oluşturalım
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))

# Modeli derleyelim ve eğitelim
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=64, verbose=1)

mse = model.evaluate(X_test, y_test, verbose=0)
print('Test set MSE:', mse)