import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from evaluate import plot_confusion_matrix, plot_roc_curve

import matplotlib.pyplot as plt


# Load preprocessed data
df = pd.read_csv('../data/rfm_churn_data.csv')

# Features and target
X = df[['Recency', 'Frequency', 'Monetary']]
y = df['Churned']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model
import os
os.makedirs('../model', exist_ok=True)
joblib.dump(model, '../model/random_forest_churn.pkl')
print("✅ Model saved to ../model/random_forest_churn.pkl")

# Predict
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluation
print("Classification Report:\n", classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_prob))

# Plot evaluation metrics


# Save confusion matrix
fig1 = plot_confusion_matrix(y_test, y_pred)
fig1.savefig('../output/confusion_matrix.png')
plt.close(fig1)

# Save ROC curve
fig2 = plot_roc_curve(y_test, y_prob)
fig2.savefig('../output/roc_curve.png')
plt.close(fig2)
# Save predictions with business-friendly output
output_df = X_test.copy()
output_df['TrueLabel'] = y_test.values
output_df['PredictedLabel'] = y_pred
output_df['ChurnProbability'] = y_prob.round(2)

# Risk segmentation and suggested actions
def assign_segment(prob):
    if prob > 0.75:
        return "High Risk"
    elif prob > 0.4:
        return "Medium Risk"
    else:
        return "Low Risk"

def suggest_action(segment):
    if segment == "High Risk":
        return "Send 20% voucher"
    elif segment == "Medium Risk":
        return "Send retention email"
    else:
        return "Promote loyalty club"

output_df['Segment'] = output_df['ChurnProbability'].apply(assign_segment)
output_df['SuggestedAction'] = output_df['Segment'].apply(suggest_action)

# Simulate CustomerID for report
output_df['CustomerID'] = ['CUST' + str(i) for i in range(1001, 1001 + len(output_df))]

# Reorder columns
report_df = output_df[['CustomerID', 'Recency', 'Frequency', 'Monetary',
                       'ChurnProbability', 'Segment', 'SuggestedAction']]
print("Saving prediction report...")

# Save to CSV

output_df.to_csv('../output/model_predictions.csv', index=False)
print("✅ Prediction report saved to ../output/model_predictions.csv")