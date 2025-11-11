import pandas as pd
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from datetime import datetime

# Load data
df = pd.read_csv("data.csv")

year = datetime.now().year
df["Date"] = pd.to_datetime(df["Date"].astype(str) + f"/{year}",
                            format="%d/%m/%Y")

# Assume dose taken at 08:00 each day (change if needed)
df["Date"] = df["Date"] + pd.Timedelta(hours=8)

# pg/ml -> mg-ish
df["Dose"] = df["Dose"] / 30.0

df = df.sort_values("Date").set_index("Date")

# --- build HOURLY index for entire span + 23h tail ---
full_index = pd.date_range(
    start=df.index.min(),
    end=df.index.max() + pd.Timedelta(hours=23),
    freq="H",
)

# doses only at dosing times, 0 otherwise
dose_hourly = df["Dose"].reindex(full_index, fill_value=0.0)

# --- 10-hour half-life model ---
half_life_h = 10
k = np.log(2) / half_life_h

dt_hours = dose_hourly.index.to_series().diff(
).dt.total_seconds().div(3600).fillna(0)

amount = []
x = 0.0
for dose, dt in zip(dose_hourly, dt_hours):
    x = x * np.exp(-k * dt) + dose
    amount.append(x)

df_sim = pd.DataFrame({
    "Dose": dose_hourly,
    "Amount": amount,
}, index=full_index)

# Plot
plt.figure(figsize=(20, 5))

# simulated concentration curve
plt.plot(df_sim.index, df_sim["Amount"],
         label=f"Amount ({half_life_h}h t½)")

plt.axhline(y=6, color='red', linestyle='--', linewidth=1)
plt.text(plt.xlim()[1], 6, '200pg/ml (90%)',
         color='red', va='bottom', ha='right')

goal = 150/30
plt.axhline(y=goal, color='blue', linestyle='-', linewidth=1)
plt.text(plt.xlim()[1], goal, 'Goal\'ish (74%)',
         color='lightgrey', va='bottom', ha='right')

plt.axhline(y=3, color='orange', linestyle='--', linewidth=1)
plt.text(plt.xlim()[1], 3, '100pg/ml (58%)',
         color='orange', va='bottom', ha='right')

plt.axhline(y=1.5, color='green', linestyle='--', linewidth=1)
plt.text(plt.xlim()[1], 1.5, '50pg/ml (24%)',
         color='green', va='bottom', ha='right')

# ---- NEW: all intercepts of Amount with goal'ish ----
y = df_sim["Amount"].values
x = df_sim.index.values
diff = y - goal

cross_idx = np.where(np.sign(diff[:-1]) * np.sign(diff[1:]) < 0)[0]

xs_int = []
ys_int = []

if len(cross_idx) > 0:
    for idx in cross_idx:
        x0, x1 = x[idx], x[idx + 1]
        y0, y1 = y[idx], y[idx + 1]
        t = (goal - y0) / (y1 - y0)  # linear interpolation
        xi = x0 + (x1 - x0) * t
        yi = goal

        xs_int.append(xi)
        ys_int.append(yi)

    # plot all intersection points
    plt.scatter(xs_int, ys_int, s=60, marker='o',
                edgecolors='black', facecolors='none',
                label="Goal'ish intercepts")

    # label each interval between neighboring intercepts (x-axis length)
    fem = True
    for i in range(len(xs_int) - 1):
        dx = xs_int[i+1] - xs_int[i]
        dx_arr = np.array(dx)

        if np.issubdtype(dx_arr.dtype, np.timedelta64):
            length_hours = dx / np.timedelta64(1, 'h')
        else:
            # assume numeric in hours
            length_hours = dx

        x_mid = xs_int[i] + dx / 2
        y_mid = goal

        plt.text(x_mid, y_mid, f"{length_hours:.1f} h",
                 ha="center", va="bottom", fontsize=8, color=("pink" if fem else "lightblue"))
        fem = not fem
# ---- END NEW ----


plt.xlabel("Time")
plt.ylabel("Dose / Amount (same units)")
plt.legend()
plt.tight_layout()
# plt.savefig("graph.jpg", dpi=100)
plt.savefig("graph.png", dpi=100, transparent=True)
# plt.show()

print("[ OK ] Saved plot to current dir")
