import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==================================================
# LOAD DATA
# ==================================================

df = pd.read_csv("data/interim/flowcast_features.csv")

# Create output folder if it does not exist
os.makedirs("reports/figures", exist_ok=True)

# ==================================================
# SET PLOT STYLE
# ==================================================

sns.set_theme(style="whitegrid")

# ==================================================
# 1. CONGESTION DISTRIBUTION
# ==================================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="congestion_level",
    order=["Free-flow", "Moderate", "Heavy", "Severe"]
)

plt.title("Traffic Congestion Distribution")
plt.xlabel("Congestion Level")
plt.ylabel("Number of Records")
plt.tight_layout()

plt.savefig(
    "reports/figures/congestion_distribution.png",
    dpi=300
)

plt.close()

# ==================================================
# 2. TRAFFIC VOLUME DISTRIBUTION
# ==================================================

plt.figure(figsize=(8, 5))

sns.histplot(
    df["traffic_volume"],
    bins=50,
    kde=True
)

plt.title("Traffic Volume Distribution")
plt.xlabel("Traffic Volume")
plt.ylabel("Frequency")
plt.tight_layout()

plt.savefig(
    "reports/figures/traffic_volume_distribution.png",
    dpi=300
)

plt.close()

# ==================================================
# 3. AVERAGE SPEED DISTRIBUTION
# ==================================================

plt.figure(figsize=(8, 5))

sns.histplot(
    df["avg_speed"],
    bins=40,
    kde=True
)

plt.title("Average Speed Distribution")
plt.xlabel("Average Speed")
plt.ylabel("Frequency")
plt.tight_layout()

plt.savefig(
    "reports/figures/average_speed_distribution.png",
    dpi=300
)

plt.close()

# ==================================================
# 4. WEATHER DISTRIBUTION
# ==================================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="weather_condition",
    order=df["weather_condition"].value_counts().index
)

plt.title("Weather Condition Distribution")
plt.xlabel("Weather Condition")
plt.ylabel("Number of Records")
plt.xticks(rotation=20)
plt.tight_layout()

plt.savefig(
    "reports/figures/weather_distribution.png",
    dpi=300
)

plt.close()

# ==================================================
# 5. TRAFFIC VOLUME BY HOUR
# ==================================================

hourly_traffic = (
    df.groupby("hour")["traffic_volume"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(9, 5))

sns.lineplot(
    data=hourly_traffic,
    x="hour",
    y="traffic_volume",
    marker="o"
)

plt.title("Average Traffic Volume by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Average Traffic Volume")
plt.xticks(range(0, 24))
plt.tight_layout()

plt.savefig(
    "reports/figures/traffic_volume_by_hour.png",
    dpi=300
)

plt.close()

# ==================================================
# 6. TRAFFIC VOLUME BY DAY OF WEEK
# ==================================================

daily_traffic = (
    df.groupby("day_of_week")["traffic_volume"]
    .mean()
    .reset_index()
)

day_names = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

daily_traffic["day_name"] = daily_traffic["day_of_week"].map(
    dict(enumerate(day_names))
)

plt.figure(figsize=(9, 5))

sns.barplot(
    data=daily_traffic,
    x="day_name",
    y="traffic_volume"
)

plt.title("Average Traffic Volume by Day of Week")
plt.xlabel("Day of Week")
plt.ylabel("Average Traffic Volume")
plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig(
    "reports/figures/traffic_volume_by_day.png",
    dpi=300
)

plt.close()

# ==================================================
# COMPLETION MESSAGE
# ==================================================

print("==================================================")
print("       FLOWCAST EDA VISUALIZATION COMPLETED")
print("==================================================")

print("\nGraphs saved in:")
print("reports/figures/")

print("\nGenerated files:")

print("1. congestion_distribution.png")
print("2. traffic_volume_distribution.png")
print("3. average_speed_distribution.png")
print("4. weather_distribution.png")
print("5. traffic_volume_by_hour.png")
print("6. traffic_volume_by_day.png")

print("\nAll EDA graphs generated successfully.")