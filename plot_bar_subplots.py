import matplotlib.pyplot as plt
import numpy as np

# 주어진 수치
# values = [0.9574, 0.5572, 0.6716, 0.5753, 0.5970, 0.6231, 0.6894, 0.6894]
values = [0.7270, 0.1293, 0.1603, 0.1430, 0.4682, 0.4530, 0.5238, 0.5236]

# 두 개씩 그룹으로 묶기 위한 인덱스 설정
x = np.arange(len(values) // 2)  # 그룹 개수
width = 0.4  # 막대 너비

# 첫 번째, 두 번째 값 그룹화하여 그리기
# plt.figure(figsize=(2.5, 2))
fig, ax = plt.subplots(1,2, figsize=(5.3, 3), width_ratios=[1,1], constrained_layout=True)

# plt.bar(x - width / 2, values[0::2], width, color='red', label='Shuffle')
# plt.bar(x + width / 2, values[1::2], width, color='blue', label='Not Shuffle')
ax[0].bar(x - width / 2, values[0::2], width, color='black', edgecolor='black', label='Shuffle')
ax[0].bar(x + width / 2, values[1::2], width, color='white', edgecolor='black', label='Not Shuffle')

# 축 설정
ax[0].set_ylabel('Accuracy', fontsize=9)
# 사용자 지정 X축 라벨
labels = ['CondGNet\n(Ours)', 'CondNet', 'Runtime\nAM', 'Runtime\nWM']
# ax[0].set_xticks(x, labels=labels, fontsize=9)  # 라벨 적용 및 기울이기
# for i, label in enumerate(labels):
ax[0].set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
y_ticks = np.arange(0, 1.1, 0.2)
ax[0].set_yticks(y_ticks, labels=["0" if i == 0 else f"{i:.1f}" if i != 1 else "1" for i in y_ticks], fontsize=9)
ax[0].set_ylim(0, 1)
# plt.legend(fontsize=9)

ax[0].spines["top"].set_visible(False)
ax[0].spines["right"].set_visible(False)



# plt.bar(x - width / 2, values[0::2], width, color='red', label='Shuffle')
# plt.bar(x + width / 2, values[1::2], width, color='blue', label='Not Shuffle')
ax[1].bar(x - width / 2, values[0::2], width, color='black', edgecolor='black', label='Shuffle')
ax[1].bar(x + width / 2, values[1::2], width, color='white', edgecolor='black', label='Not Shuffle')

# 축 설정
# ax[0].set_ylabel('Accuracy', fontsize=9)
# 사용자 지정 X축 라벨
# labels = ['CGN (Ours)', 'CN', 'RAM', 'RWM']
ax[1].set_xticks(x, labels=labels, fontsize=9)  # 라벨 적용 및 기울이기
y_ticks = np.arange(0, 1.1, 0.2)
ax[1].set_yticks(y_ticks, labels=["0" if i == 0 else f"{i:.1f}" if i != 1 else "1" for i in y_ticks], fontsize=9)
ax[1].set_ylim(0, 1)
# plt.legend(fontsize=9)

ax[1].spines["top"].set_visible(False)
ax[1].spines["right"].set_visible(False)


# 그래프 표시
plt.savefig("cnn_bar.pdf", format="pdf", bbox_inches = 'tight')