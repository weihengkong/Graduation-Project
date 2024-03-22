import pandas as pd

# 读取数据文件
with open("data.txt", "r") as file:
    data = file.readlines()

# 解析数据并存储到DataFrame中
rows = []
for line in data:
    if line.startswith("function"):
        values = line.strip().split()
        rows.append({"function1": float(values[1]), "function2": float(values[3]), "function3": float(values[5])})
df = pd.DataFrame(rows)

# 将数据输出到Excel文件
df.to_excel("outputGnome.xlsx", index=False)
