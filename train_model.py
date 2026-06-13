# IMPORT LIBRARIES

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score,confusion_matrix,ConfusionMatrixDisplay,classification_report,roc_auc_score,precision_score)

import matplotlib.pyplot as plt

# Load Dataset
df = pd.read_csv(r"LoanDefaultDataset (1).csv")


# DATA CLEANING
df.drop(["LoanID","Education","HasDependents","LoanPurpose","MaritalStatus"],axis=1,inplace=True)

# Split Features & Target
x = df.iloc[:,:-1]
y = df.iloc[:,-1]

# One Hot Encoding
x = pd.get_dummies(x, drop_first=True)

print("Original Class Distribution:\n", y.value_counts())

# SPLIT FIRST
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

# MODEL BUILDING
rf = RandomForestClassifier(n_estimators=200,max_depth=8,min_samples_leaf=2,class_weight='balanced_subsample',random_state=42,n_jobs=-1)

rf.fit(x_train,y_train)

# PREDICTION
y_predict = rf.predict(x_test)
y_prob = rf.predict_proba(x_test)[:,1]

# EVALUATION
print("Accuracy:", accuracy_score(y_test,y_predict))
print("ROC-AUC:", roc_auc_score(y_test,y_prob))
print("Precision:",precision_score(y_test,y_predict))
print("\nClassification Report:\n")
print(classification_report(y_test,y_predict))

# Confusion Matrix
cm = confusion_matrix(y_test,y_predict)
ConfusionMatrixDisplay(cm).plot()
plt.savefig("confusion_matrix.png")
plt.close()

# Feature Importance
importance = rf.feature_importances_
features = x.columns

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance:\n")
print(importance_df)

plt.figure()
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.gca().invert_yaxis()
plt.title("Feature Importance")
plt.savefig("feature_importance.png")
plt.close()

import joblib

joblib.dump(rf, "loan_model.pkl")
joblib.dump(x.columns, "model_columns.pkl")

print("Model saved successfully!")