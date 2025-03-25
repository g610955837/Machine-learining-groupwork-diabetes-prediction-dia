import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import joblib

data = pd.read_csv("D:\python_work\diabetes_project\已处理-标准化.csv")

# 假设目标变量为 "Outcome"，其余为特征
X = data.drop(columns=["Outcome"])  
y = data["Outcome"]  

# k-fold 交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scoring_metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]


# 1.随机森林模型
rf = RandomForestClassifier(random_state=42, n_estimators=100)
MODEL_PATH = "D:\python_work\diabetes_project\Randomforest.pkl"
joblib.dump(rf, MODEL_PATH)
print(f"模型已保存到 {MODEL_PATH}")
# 使用 k-fold 交叉验证评估
rf_scores = {}
for metric in scoring_metrics:
    scores = cross_val_score(rf, X, y, cv=kf, scoring=metric)
    rf_scores[metric] = scores.mean()
print("\n随机森林模型 k-fold 评估结果：")
for metric, score in rf_scores.items():
    print(f"{metric.capitalize()}: {score:.4f}")


#  XGBoost 模型
xgb = XGBClassifier(eval_metric="logloss", random_state=42)
# 使用 k-fold 交叉验证评估
MODEL_PATH = "D:\python_work\diabetes_project\XGBoost.pkl"
joblib.dump(xgb, MODEL_PATH)
print(f"模型已保存到 {MODEL_PATH}")
xgb_scores = {}
for metric in scoring_metrics:
    scores = cross_val_score(xgb, X, y, cv=kf, scoring=metric)
    xgb_scores[metric] = scores.mean()
print("\nXGBoost 模型 k-fold 评估结果：")
for metric, score in xgb_scores.items():
    print(f"{metric.capitalize()}: {score:.4f}")


# 创建评估结果表
results = pd.DataFrame({
    "模型": ["随机森林", "XGBoost"],
    "准确率": [rf_scores["accuracy"], xgb_scores["accuracy"]],
    "精确率": [rf_scores["precision"], xgb_scores["precision"]],
    "召回率": [rf_scores["recall"], xgb_scores["recall"]],
    "F1分数": [rf_scores["f1"], xgb_scores["f1"]],
    "AUC": [rf_scores["roc_auc"], xgb_scores["roc_auc"]]
})

print("\n模型性能对比：")
print(results)

from sklearn.metrics import roc_curve, auc

def plot_roc_kfold(model, X, y, kf, model_name):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    plt.plot(mean_fpr, mean_tpr, label=f"{model_name} (AUC = {mean_auc:.4f})")

# 绘制所有模型的 ROC 曲线
plt.figure(figsize=(8, 6))
plot_roc_kfold(RandomForestClassifier(random_state=42), X, y, kf, "random forest")
plot_roc_kfold(XGBClassifier(eval_metric="logloss", random_state=42), X, y, kf, "XGBoost")

plt.plot([0, 1], [0, 1], "k--", label="Random Guessing")
plt.title("ROC Curve Comparison (k-fold Average")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.legend()
plt.show()