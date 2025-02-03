import matplotlib.pyplot as plt
import numpy as np

# 주어진 수치
values = [0.9574, 0.5572, 0.6716, 0.5753, 0.5970, 0.6231, 0.7954, 0.7954]

# 두 개씩 그룹으로 묶기 위한 인덱스 설정
x = np.arange(len(values) // 2)  # 그룹 개수
width = 0.4  # 막대 너비

# 첫 번째, 두 번째 값 그룹화하여 그리기
plt.figure(figsize=(8, 5))
plt.bar(x - width / 2, values[0::2], width, color='red', label='Shuffle')
plt.bar(x + width / 2, values[1::2], width, color='blue', label='Not Shuffle')

# 축 설정
plt.ylabel('Value')
plt.title('Grouped Bar Chart')
# 사용자 지정 X축 라벨
labels = ['CondGNet', 'CondNet', 'Runtime\nActivation Magnitude', 'Runtime\nWeight Magnitude']
plt.xticks(x, labels=labels)  # 라벨 적용 및 기울이기
plt.ylim(0, 1)
plt.legend()

# 그래프 표시
plt.show()
