import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("heuristic_coefficient_results.csv", index_col=0)

plt.figure(figsize=(12, 6))
sns.heatmap(df, annot=True, fmt=".4f", cmap="YlGnBu", 
            cbar_kws={'label': 'Area under curve'})
plt.title("Performance under Different Coefficients")
plt.ylabel("Coefficient")
plt.xlabel("City")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()