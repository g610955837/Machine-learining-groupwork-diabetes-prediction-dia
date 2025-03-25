import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

# 加载标准化后的数据
data = pd.read_csv("D:\python_work\diabetes_project\已处理-标准化.csv")
X = data.drop(columns=["Outcome"])  
y = data["Outcome"]  

# 定义基模型
rf = RandomForestClassifier(random_state=42, n_estimators=100)
xgb = XGBClassifier(objective="binary:logistic", eval_metric="logloss", random_state=42)
lr = LogisticRegression(random_state=42)

# 定义元模型（最终模型）
meta_model = LogisticRegression()

# 创建堆叠分类器
stacking_clf = StackingClassifier(
    estimators=[("RandomForest", rf), ("XGBoost", xgb), ("LogisticRegression", lr)],
    final_estimator=meta_model,  # 使用逻辑回归作为元模型
    cv=5  # 使用 5 折交叉验证生成基模型的预测
)
MODEL_PATH = "D:\python_work\diabetes_project\StackingClassifier.pkl"
joblib.dump(stacking_clf, MODEL_PATH)
print(f"模型已保存到 {MODEL_PATH}")
# k-fold 交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 定义评估指标
scoring_metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]

results = {}
for metric in scoring_metrics:
    scores = cross_val_score(stacking_clf, X, y, cv=kf, scoring=metric)
    results[metric] = scores.mean()

print("\nK-Fold 交叉验证评估结果：")
for metric, score in results.items():
    print(f"{metric.capitalize()}: {score:.4f}")