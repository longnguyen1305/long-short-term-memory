import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# ----------------------------------------------------------------------- Load data
df = pd.read_csv('data/monthly_milk_production.csv',
                 index_col='Date',
                 parse_dates=True)

# ----------------------------------------------------------------------- Preprocessing
# Cross validation
cut = 112
train = df.iloc[:cut]
test = df.iloc[cut:]

# Normalize
scaler = MinMaxScaler()
scaler.fit(train)
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)

# ----------------------------------------------------------------------- Train model
n_features = 1
n_input = 12
generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)

# Define model
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(n_input, n_features)))
model.add(Dense(1))

# Compile model
model.compile(optimizer='adam', loss='mse')

# Train model
model.summary()
model.fit(generator, epochs=5)

# ----------------------------------------------------------------------- Test model
# Create test data generator
test_generator = TimeseriesGenerator(scaled_test, scaled_test, length=n_input, batch_size=1)
# Predict
test_predictions = model.predict(test_generator)
# Inverse transform predictions
test_predictions_original = scaler.inverse_transform(test_predictions)

# ----------------------------------------------------------------------- Visuallization
# Prepare test data for plotting
test_actual = df.iloc[cut + n_input:].values

# Plot actual vs predicted
plt.figure(figsize=(10, 5))
plt.plot(test_actual, label="Actual", linestyle="dashed")
plt.plot(test_predictions_original, label="Predicted")
plt.xlabel("Time")
plt.ylabel("Milk Production")
plt.title("LSTM Predictions vs Actual Values")
plt.legend()
plt.show()
