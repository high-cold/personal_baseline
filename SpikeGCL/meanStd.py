import statistics

# 输入三个数
num1 = float(input("请输入第一个数: "))
num2 = float(input("请输入第二个数: "))
num3 = float(input("请输入第三个数: "))

# 计算均值
mean = statistics.mean([num1, num2, num3])

# 计算方差
variance = statistics.variance([num1, num2, num3])

# 输出均值和方差，保留两位小数
print(f"均值: {mean:.2f} ± 方差: {variance:.2f}")
