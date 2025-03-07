import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# 컬러맵 및 정규화 설정 (gray() 사용)
cmap = cm.gray  # Grayscale 컬러맵
norm = mcolors.Normalize(vmin=0, vmax=29)

# 새로운 컬러바 전용 Figure 생성
fig, ax = plt.subplots(figsize=(0.7, 1.2))  # 세로 길이를 길게 조정
fontsize = 9
# 컬러바 추가
cbar = plt.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax)
cbar.ax.tick_params(labelsize=9)  # 컬러바 눈금 폰트 크기 설정
cbar.set_label("Pruning Iteration", fontsize=fontsize)
plt.tight_layout()
# 불필요한 축 제거
ax.remove()
fig.subplots_adjust(top=0.9, bottom=0.1)  # top과 bottom을 조정하여 y축 여백 추가
plt.savefig("fig1.pdf", format="pdf", bbox_inches='tight')


# 컬러바 출력
# plt.show()