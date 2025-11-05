# GPT generated, because not important
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Load data
df = pd.read_csv("data.csv")

# Add a year to the Date column (use current year; change if needed)
year = datetime.now().year
df["Date"] = pd.to_datetime(df["Date"].astype(str) + f"/{year}",
                            format="%d/%m/%Y")

# Sort and set index
df = df.sort_values("Date").set_index("Date")

# 36-hour rolling average
df["Dose_36h_avg"] = df["Dose"].rolling("36H", min_periods=1).mean()

# Plot
plt.figure(figsize=(10, 5))
plt.plot(df.index, df["Dose"], label="Dose", alpha=0.3)
plt.plot(df.index, df["Dose_36h_avg"], label="Theoretical average pg/mg")

for x, y in zip(df.index, df["Dose_36h_avg"]):
    plt.text(x, y, f"{y:.1f}", fontsize=8,
             ha='center', va='bottom', rotation=45)

plt.xlabel("Date")
plt.ylabel("Dose")
plt.legend()
plt.tight_layout()
plt.savefig("graph.jpg", dpi=100)
plt.show()
