# import matplotlib.pyplot as plt
#
# # 범례 라벨 및 색상 정의
# legend_labels = {
#     "Shuffle": ("black", "black"),  # 검은색 배경, 검은 테두리
#     "Not Shuffle": ("white", "black")  # 흰색 배경, 검은 테두리
# }
#
# # 빈 플롯 생성 (여백 최소화)
# plt.figure(figsize=(5.2, 0.2))
#
# # 범례 추가
# plt.legend(
#     [plt.Line2D([0], [0], color=border, marker="s", markersize=10, markerfacecolor=fill, markeredgecolor=border, linestyle="None")
#      for _, (fill, border) in legend_labels.items()],
#     list(legend_labels.keys()),
#     loc="center",
#     fontsize=9,
#     framealpha=0.8,
#     ncol=2  # 2열로 정렬
# )
#
# # 축 및 프레임 숨기기
# plt.axis("off")
#
# # 여백 최소화
# plt.tight_layout()
#
# # 그래프 출력
# plt.savefig("bar_legend.pdf", format="pdf", bbox_inches = 'tight')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 범례 라벨 및 색상 정의
legend_labels = {
    "Not Shuffle": ("black", "black"),  # 검은색 배경, 검은 테두리
    "Shuffle": ("white", "black")  # 흰색 배경, 검은 테두리
}

# 빈 플롯 생성 (여백 최소화)
plt.figure(figsize=(5.2, 0.3))

# 범례 아이템 생성
legend_patches = [
    mpatches.Rectangle((0, 0), 2, 0.5, facecolor=fill, edgecolor=border, linewidth=1.5)
    for fill, border in legend_labels.values()
]

# 범례 추가
plt.legend(
    legend_patches,
    list(legend_labels.keys()),
    loc="center",
    fontsize=9,
    framealpha=0.8,
    ncol=2  # 2열로 정렬
)

# 축 및 프레임 숨기기
plt.axis("off")

# 여백 최소화
plt.tight_layout()

# 그래프 출력
plt.savefig("bar_legend.pdf", format="pdf", bbox_inches = 'tight')
