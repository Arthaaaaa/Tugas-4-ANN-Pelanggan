# model_training.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. Load Dataset
df = pd.read_excel('Dataset.xlsx')

# 2. Ambil input dan output
X = df[['Tanggal']].values  # fitur input: Tanggal
y = df['Penjualan (pcs)'].values  # target output: Penjualan

# 3. Normalisasi
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
y = y  # output kita biarkan dulu (opsional: bisa juga dinormalisasi)

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. Bangun Model ANN
model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(16, activation='relu'),
    Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])


# 6. Train model
history = model.fit(X_train, y_train, epochs=200, validation_data=(X_test, y_test))

# 7. Save model
model.save('model_pelanggan.h5')

# 8. Visualisasi training
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

print("Training Selesai!")
