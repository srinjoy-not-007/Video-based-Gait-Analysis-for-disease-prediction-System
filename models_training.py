import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from scipy.stats import uniform, randint


DATASET_CSV = "gait_features_with_labels.csv"  
MODEL_DIR = "models"
SHAP_DIR = "shap_plots"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SHAP_DIR, exist_ok=True)


df = pd.read_csv(DATASET_CSV)
print("Dataset loaded! Shape:", df.shape)

X = df.drop(columns=['label'])
y = df['label']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


print("\nTraining Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_cv_scores = cross_val_score(lr_model, X_train, y_train, cv=cv, scoring='accuracy')
print("Logistic Regression Cross-Validation Accuracy:", lr_cv_scores.mean())

lr_model.fit(X_train, y_train)
joblib.dump(lr_model, os.path.join(MODEL_DIR, "logistic_regression.pkl"))

lr_pred = lr_model.predict(X_test)
print("\nLogistic Regression Report:\n", classification_report(y_test, lr_pred))


print("\nTuning SVM with GridSearchCV...")
svm_params = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 0.01, 0.001],
    'kernel': ['rbf', 'linear']
}

svm_model = SVC(probability=True, random_state=42)
svm_grid = GridSearchCV(svm_model, svm_params, cv=cv, scoring='accuracy', n_jobs=-1)
svm_grid.fit(X_train, y_train)

best_svm = svm_grid.best_estimator_
joblib.dump(best_svm, os.path.join(MODEL_DIR, "svm_model.pkl"))

print("Best SVM Params:", svm_grid.best_params_)

svm_pred = best_svm.predict(X_test)
print("\nSVM Report:\n", classification_report(y_test, svm_pred))


print("\nTuning RandomForest with GridSearchCV...")
rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

rf_model = RandomForestClassifier(random_state=42)
rf_grid = GridSearchCV(rf_model, rf_params, cv=cv, scoring='accuracy', n_jobs=-1)
rf_grid.fit(X_train, y_train)

best_rf = rf_grid.best_estimator_
joblib.dump(best_rf, os.path.join(MODEL_DIR, "random_forest_model.pkl"))

print("Best RF Params:", rf_grid.best_params_)

rf_pred = best_rf.predict(X_test)
print("\nRandomForest Report:\n", classification_report(y_test, rf_pred))


print("\nTuning XGBoost with RandomizedSearchCV...")
xgb_params = {
    'n_estimators': randint(100, 300),
    'max_depth': randint(3, 10),
    'learning_rate': uniform(0.01, 0.3),
    'subsample': uniform(0.5, 0.5)
}

xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgb_random = RandomizedSearchCV(xgb_model, param_distributions=xgb_params, n_iter=20, cv=cv, scoring='accuracy', n_jobs=-1, random_state=42)
xgb_random.fit(X_train, y_train)

best_xgb = xgb_random.best_estimator_
joblib.dump(best_xgb, os.path.join(MODEL_DIR, "xgboost_model.pkl"))

print("Best XGBoost Params:", xgb_random.best_params_)

xgb_pred = best_xgb.predict(X_test)
print("\nXGBoost Report:\n", classification_report(y_test, xgb_pred))

# SHAP Analysis on Best XGBoost
print("\nRunning SHAP analysis on XGBoost...")

explainer = shap.TreeExplainer(best_xgb)
shap_values = explainer.shap_values(X_test)


plt.figure()
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False, feature_names=df.drop(columns=['label']).columns)
plt.title("SHAP Feature Importance - XGBoost")
plt.savefig(os.path.join(SHAP_DIR, "shap_summary_bar.png"))
plt.close()

plt.figure()
shap.summary_plot(shap_values, X_test, show=False, feature_names=df.drop(columns=['label']).columns)
plt.title("SHAP Summary Plot - XGBoost")
plt.savefig(os.path.join(SHAP_DIR, "shap_summary.png"))
plt.close()

print(f"\nSHAP plots saved in {SHAP_DIR}")


def plot_3d_joint_angles(df, joint_angle_cols, label_col='label'):
    fig = go.Figure()

    labels = df[label_col].unique()
    colors = ['red', 'green', 'blue', 'orange']

    for idx, label in enumerate(labels):
        subset = df[df[label_col] == label]
        fig.add_trace(go.Scatter3d(
            x=subset[joint_angle_cols[0]],
            y=subset[joint_angle_cols[1]],
            z=subset[joint_angle_cols[2]],
            mode='markers',
            marker=dict(size=5, color=colors[idx % len(colors)]),
            name=str(label)
        ))

    fig.update_layout(
        title='3D Joint Angle Visualization',
        scene=dict(
            xaxis_title=joint_angle_cols[0],
            yaxis_title=joint_angle_cols[1],
            zaxis_title=joint_angle_cols[2]
        )
    )
    fig.show()

print("\n=== All Done! ===")
