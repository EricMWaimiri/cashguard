import numpy as np
import pandas as pd

np.random.seed(42)

# Synthetic orders: 3000 rows. The fraud label is generated from simple rules plus noise.
n = 3000
orders = pd.DataFrame({
    "amount": np.random.gamma(2.5, 60, n).round(2),
    "account_age_days": np.random.exponential(300, n).round(0),
    "ship_bill_mismatch": np.random.binomial(1, 0.15, n),
    "hour_of_day": np.random.randint(0, 24, n),
    "orders_last_hour": np.random.poisson(0.5, n),
})

score = (
    0.004 * orders["amount"]
    - 0.006 * orders["account_age_days"]
    + 1.8 * orders["ship_bill_mismatch"]
    + 0.7 * orders["orders_last_hour"]
    + 0.8 * orders["hour_of_day"].isin([0, 1, 2, 3, 4]).astype(int)
    - 1.5
)
prob = 1 / (1 + np.exp(-score))
orders["is_fraud"] = np.random.binomial(1, prob)
orders.to_csv("orders.csv", index=False)

print("orders:", orders.shape, "fraud rate:", round(orders["is_fraud"].mean(), 3))