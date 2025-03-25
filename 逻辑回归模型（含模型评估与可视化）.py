import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


file_path = "D:\python_work\diabetes_project\已处理-标准化.csv"

# 加载数据
data = pd.read_csv(file_path)

# 查看数据的前几行以确保正确加载
print(data.head())

X = data.iloc[:, :-1]  # 特征
y = data.iloc[:, -1]   # 标签

# 分割数据为训练集和测试集（80%-20%）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 逻辑回归模型
model = LogisticRegression(C=1.0, max_iter=10000)

# 训练模型
model.fit(X_train, y_train)

# 进行预测
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # 获取预测为正类的概率

test_df = X_test.copy()  # 这里X_test是一个DataFrame或类似DataFrame的对象
test_df['predicted_label'] = y_pred  # 添加预测的类别标签作为新列
test_df['predicted_proba'] = y_prob  # 添加预测为正类的概率作为新列

# 打印结果查看
print(test_df)

# 评估模型性能
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC-ROC: {roc_auc:.4f}")

# 可视化目标变量（糖尿病标签）的分布情况
sns.countplot(x=y)
plt.title("Distribution of Target Variable")
plt.xlabel("Diabetes (0: No, 1: Yes)")
plt.ylabel("Count")
plt.show()

# 为每个特征绘制箱形图
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
for feature in features:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Outcome', y=feature, data=data)
    plt.title(f'{feature} Distribution by Outcome')
    plt.xlabel('Outcome')
    plt.ylabel(feature)
    plt.show()

# ROC曲线和AUC可视化
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label=f"AUC-ROC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

# 混淆矩阵可视化
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Diabetes", "Diabetes"], yticklabels=["No Diabetes", "Diabetes"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()