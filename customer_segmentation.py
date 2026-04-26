import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# Dataset
data = {
    "Annual_Income": [15, 16, 17, 18, 40, 42, 43, 44, 80, 82, 85, 88],
    "Spending_Score": [39, 40, 42, 43, 60, 62, 65, 66, 90, 92, 94, 95]
}
df = pd.DataFrame(data)

# Model
kmeans = KMeans(n_clusters=3, random_state=42)
df["Cluster"] = kmeans.fit_predict(df)

# Plot clusters
plt.scatter(df["Annual_Income"], df["Spending_Score"], c=df["Cluster"], cmap="viridis")
plt.xlabel("Income")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation")
plt.show()
