import tkinter as tk
from tkinter import messagebox
import numpy as np
import joblib  # 用于加载已保存的机器学习模型

# 加载训练好的机器学习模型
MODEL_PATH = "D:\python_work\diabetes_project\StackingClassifier.pkl"
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    def show_error():
        root = tk.Tk()
        root.withdraw()  # 隐藏主窗口
        messagebox.showerror("错误", f"无法找到模型文件：{MODEL_PATH}")
        exit()
    show_error()

# 定义界面功能
def predict_diabetes():
    """从用户输入中获取数据并预测是否有患糖尿病的风险"""
    try:
        pregnancies = float(entry_pregnancies.get())
        glucose = float(entry_glucose.get())
        blood_pressure = float(entry_blood_pressure.get())
        skin_thickness = float(entry_skin_thickness.get())
        insulin = float(entry_insulin.get())
        bmi = float(entry_bmi.get())
        dpf = float(entry_dpf.get())
        age = float(entry_age.get())

        data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
        prediction = model.predict(data)

        if prediction[0] == 1:
            messagebox.showinfo("预测结果", "模型预测结果：患者可能有患糖尿病的风险。")
        else:
            messagebox.showinfo("预测结果", "模型预测结果：患者没有患糖尿病的风险。")
    except ValueError:
        messagebox.showerror("输入错误", "请确保所有输入值均为有效的数字。")

# 创建主窗口
root = tk.Tk()
root.title("糖尿病早期检测 - 用户界面")
root.geometry("400x800")  # 将高度从600增加到700
root.resizable(False, False)

# 添加标题
tk.Label(root, text="糖尿病早期检测", font=("Arial", 16), pady=10).pack()

# 输入框布局
frame = tk.Frame(root)
frame.pack(pady=10, padx=20)

labels = [
    "怀孕次数 (Pregnancies):",
    "血糖值 (Glucose):",
    "血压值 (Blood Pressure):",
    "皮肤厚度 (Skin Thickness):",
    "胰岛素值 (Insulin):",
    "体重指数 (BMI):",
    "糖尿病家族史 (DPF):",
    "年龄 (Age):"
]
entries = []

for label_text in labels:
    tk.Label(frame, text=label_text, anchor="w").pack(fill="x", pady=5)
    entry = tk.Entry(frame)
    entry.pack(fill="x", pady=5)
    entries.append(entry)

entry_pregnancies = entries[0]
entry_glucose = entries[1]
entry_blood_pressure = entries[2]
entry_skin_thickness = entries[3]
entry_insulin = entries[4]
entry_bmi = entries[5]
entry_dpf = entries[6]
entry_age = entries[7]

# 按钮布局
button_frame = tk.Frame(root)
button_frame.pack(pady=20)

tk.Button(button_frame, text="预测", font=("Arial", 14), command=predict_diabetes).pack(pady=10, anchor="center")
tk.Button(button_frame, text="退出", font=("Arial", 12), command=root.destroy).pack(pady=10, anchor="center")

root.mainloop()
# 运行主循环
root.mainloop()