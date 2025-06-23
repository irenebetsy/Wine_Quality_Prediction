from Libraries import sns,plt,LabelEncoder,pd,np,stats,os,datetime
from Data import data_file_path

# Load your DataFrame (replace with your actual dataset path if needed)
df = pd.read_csv(data_file_path, encoding='latin-1')

# Ensure eda_results folder exists
eda_folder = "SQL_Data\EDA_Results"
os.makedirs(eda_folder, exist_ok=True)

# Get timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

print("Generating count plot for 'type'...")
sns.set(rc={'figure.figsize': (20, 8.27)})
sns.countplot(x='type', data=df)
plt.title("Wine Type Count")
plt.savefig(f"{eda_folder}/countplot_type_{timestamp}.png")
plt.show()

# Label encode 'type'
label_encoder =LabelEncoder()
df['type'] = label_encoder.fit_transform(df['type'])

print("Scatter plot: alcohol vs fixed acidity...")
plt.figure(figsize=(10, 7))
plt.scatter(x="alcohol", y="fixed acidity", data=df, marker='o', color='purple')
plt.xlabel("alcohol", fontsize=15)
plt.ylabel("fixed_acidity", fontsize=15)
plt.title("Alcohol vs Fixed Acidity")
plt.savefig(f"{eda_folder}/scatter_alcohol_fixed_acidity_{timestamp}.png")
plt.show()

print("Generating correlation heatmap...")
correlation_matrix = df.corr()
dataplot = sns.heatmap(correlation_matrix, cmap="YlGnBu", annot=True)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{eda_folder}/correlation_heatmap_{timestamp}.png")
plt.show()

# Save correlation matrix as CSV
correlation_matrix.to_csv(f"{eda_folder}/correlation_matrix_{timestamp}.csv")

print("Generating boxplot for all features...")
sns.set()
plt.figure(figsize=(20, 10))
sns.boxplot(data=df, palette="Set3")
plt.title("Boxplot for All Features")
plt.savefig(f"{eda_folder}/boxplot_all_features_{timestamp}.png")
plt.show()

print("Detecting outliers using Z-score...")
z = np.abs(stats.zscore(df))
print("Outlier indices where z > 3:", np.where(z > 3))

# Remove outliers
df1 = df[(z < 3).all(axis=1)]
print(f"Original shape: {df.shape}")
print(f"Shape after outlier removal: {df1.shape}")

print("Mapping 'quality' to Low, Medium, High...")
quality_mapping = {3: "Low", 4: "Low", 5: "Medium", 6: "Medium", 7: "Medium", 8: "High", 9: "High"}
df["quality"] = df["quality"].map(quality_mapping)
print(df.quality.value_counts())

print("Encoding 'quality' labels into numbers...")
mapping_quality = {"Low": 0, "Medium": 1, "High": 2}
df["quality"] = df["quality"].map(mapping_quality)

# Final head print
print("Sample of cleaned and mapped DataFrame:")
print(df.head())