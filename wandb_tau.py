import wandb
from wandb import Api

# 1. WandB 로그인
wandb.login()

# 2. W&B API 객체 생성
api = Api()

# 3. 원하는 프로젝트와 엔터티의 실행 목록 가져오기
runs = api.runs("hails/condg_mlp")

# 4. 특정 run 찾기
target_run = None
for run in runs:
    if run.name == "cond_mlp_schedule_s=7_v=0.2_tau=0.4":
        target_run = run
        break

# 5. 해당 run이 없는 경우 에러 처리
if target_run is None:
    raise ValueError("해당 이름의 run을 찾을 수 없습니다.")

# 6. 로그 데이터 추출
history = target_run.history(keys=["test/epoch_tau"])

# 7. 'test/epoch_tau' 데이터와 스텝 정보 추출
epoch_tau_values = history["test/epoch_tau"].dropna().tolist()
steps = history["_step"].dropna().tolist()

# 8. 데이터 출력
print("Step-wise 'test/epoch_tau' values:")
for step, value in zip(steps, epoch_tau_values):
    print(f"Step: {step}, test/epoch_tau: {value}")

# 9. 결과를 CSV 파일로 저장 (선택 사항)
import pandas as pd

df = pd.DataFrame({
    "Step": steps,
    "test/epoch_tau": epoch_tau_values
})

# 파일 저장
df.to_csv("test_epoch_tau_values.csv", index=False)
print("Data saved to test_epoch_tau_values.csv")
