from sklearn.ensemble import RandomForestClassifier
from data_extractor import PatientData
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
data = PatientData()


features = data.feature_names

# Train a RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(data.X, data.y)

# Get feature importances
importances = clf.feature_importances_

# Create a DataFrame for better visualization
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print(importance_df)

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Importance')
plt.title('Feature Importances')
plt.savefig('feature_assessment.png')