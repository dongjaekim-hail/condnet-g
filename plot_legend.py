import matplotlib.pyplot as plt

# 범례 라벨 및 색상, 스타일 정의
legend_labels = {
    "CondGNet (Ours)": ("blue", "-"),
    "CondNet": ("green", "-"),
    "Runtime Activation Magnitude": ("red", "-"),
    "Runtime Weight Magnitude": ("orange", "-"),
    "Unstructured LTH": ("black", "-"),
    "Structured LTH": ("black", ":")  # 점선 스타일로 변경
}

# 빈 플롯 생성 (여백 최소화)
plt.figure(figsize=(5.2, 0.9))

# 범례 추가 (2열)
plt.legend(
    [plt.Line2D([0], [0], color=color, linestyle=linestyle) for _, (color, linestyle) in legend_labels.items()],
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
plt.savefig("legend.pdf", format="pdf", bbox_inches = 'tight')