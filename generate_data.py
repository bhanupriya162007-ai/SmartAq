import pandas as pd
import numpy as np
import os

# Generate sample data for SmartAQI

np.random.seed(42)

total_records = 800

# Generate traffic and environmental values
traffic = np.random.randint(100, 1501, total_records)

pm25 = 8 + (traffic / 65) + np.random.normal(0, 8, total_records)
pm10 = 18 + (traffic / 35) + np.random.normal(0, 12, total_records)

pm25 = np.clip(pm25, 5, 200)
pm10 = np.clip(pm10, 10, 300)

noise = 45 + (traffic / 40) + np.random.normal(0, 5, total_records)
noise = np.clip(noise, 35, 120)

temperature = np.random.uniform(18, 42, total_records)
humidity = np.random.uniform(25, 95, total_records)

# Create a synthetic AQI-like target
aqi = (
    1.8 * pm25
    + 0.65 * pm10
    + 0.025 * traffic
    + 0.12 * noise
    + 0.08 * temperature
    - 0.025 * humidity
    + np.random.normal(0, 8, total_records)
)

aqi = np.clip(aqi, 0, 500)

# Create a dataset
data = pd.DataFrame({
    "Traffic_Density": traffic,
    "PM25": pm25,
    "PM10": pm10,
    "Noise_Level": noise,
    "Temperature": temperature,
    "Humidity": humidity,
    "AQI": aqi
})

# Save the dataset
os.makedirs("data", exist_ok=True)

data.to_csv("data/smart_aqi_dataset.csv", index=False)

print("SmartAQI dataset created successfully!")
print("Total records:", len(data))
print("\nFirst five records:")
print(data.head())
print("\nDataset saved in data/smart_aqi_dataset.csv")