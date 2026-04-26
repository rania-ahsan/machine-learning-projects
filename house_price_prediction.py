import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
# Dataset
data = {
    "Area": [1000, 1500, 2000, 2500, 3000, 3500],
    "Price": [200000, 250000, 300000, 350000, 400000, 450000]
}
df = pd.DataFrame(data)

# Features and target
X = df[["Area"]]
y = df["Price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# Plot
plt.scatter(df["Area"], df["Price"])
plt.plot(df["Area"], model.predict(df[["Area"]]), color="red")
plt.show()
