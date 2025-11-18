import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
import numpy as np

# --- (★설정★) ---
# 2단계에서 만든 라벨링된 데이터 파일
INPUT_FILE_NAME = "mobile_legends_labeled_for_koelectra.csv"

# 사용할 KoELECTRA 모델
MODEL_NAME = "monologg/koelectra-base-v3-discriminator"

# 학습 결과물을 저장할 폴더 이름
OUTPUT_DIR = "./koelectra_results"


# --------------------

def load_data(file_name):
    """CSV 파일을 불러와 DataFrame으로 반환"""
    print(f"데이터 파일을 불러옵니다: {file_name}")
    try:
        df = pd.read_csv(file_name)
        df = df.dropna()  # 혹시 모를 결측치 제거
        return df
    except FileNotFoundError:
        print(f"[오류] '{file_name}' 파일을 찾을 수 없습니다.")
        return None


# PyTorch Dataset 클래스 정의
class GameReviewDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(p):
    """학습 중 평가를 위한 정확도 계산 함수"""
    preds = np.argmax(p.predictions, axis=1)
    return {'accuracy': accuracy_score(p.label_ids, preds)}


def main():
    # 1. 데이터 로드
    df = load_data(INPUT_FILE_NAME)
    if df is None:
        return

    # (★중요★)
    # 2만 건은 학습에 너무 오래 걸리므로, 샘플 5000건만 사용합니다.
    # (원본 데이터의 10% 내외 - 2천 건 이상' 요구사항 충족)
    # (컴퓨터 성능이 좋으면 이 숫자를 늘리세요)
    if len(df) > 5000:
        print(f"원본 데이터가 너무 많습니다. {len(df)}건 -> 5000건으로 샘플링합니다.")
        df = df.sample(n=5000, random_state=42)

    # 2. 학습/검증 데이터 분리 (8:2)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['document'].tolist(),
        df['label'].tolist(),
        test_size=0.2,
        random_state=42
    )

    # 3. KoELECTRA 토크나이저 로드
    print(f"토크나이저 로드 중: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 4. 텍스트 토크나이징
    print("텍스트 토크나이징 중...")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

    # 5. PyTorch Dataset 생성
    train_dataset = GameReviewDataset(train_encodings, train_labels)
    val_dataset = GameReviewDataset(val_encodings, val_labels)

    # 6. KoELECTRA 모델 로드 (분류용)
    print(f"모델 로드 중: {MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    # 7. 학습 설정 (TrainingArguments)
    # (GPU가 없으면 'no_cuda=True'를 추가해야 합니다)

    # --- (★수정된 부분★) ---
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,  # 결과물 저장 경로
        num_train_epochs=3,  # 총 학습 Epoch (3~5 추천)
        per_device_train_batch_size=16,  # 학습 Batch 크기
        per_device_eval_batch_size=64,  # 평가 Batch 크기
        warmup_steps=500,  #
        weight_decay=0.01,  #
        logging_dir='./logs',  # 로그 저장 경로
        logging_steps=10,
        eval_strategy="epoch",  # (이전 수정사항)
        save_strategy="epoch",  # (이전 수정사항)
        load_best_model_at_end=True,  # 학습 후 최고 성능 모델 로드
        report_to="none",  # (★wandb 오류 해결★) 시각화 도구 사용 안 함
        # no_cuda=True                   # (★주의★) GPU가 없으면 이 주석을 해제하세요!
    )
    # --- (★수정 완료★) ---

    # 8. Trainer 초기화
    trainer = Trainer(
        model=model,  # 학습시킬 모델
        args=training_args,  # 학습 설정
        train_dataset=train_dataset,  # 학습 데이터
        eval_dataset=val_dataset,  # 평가 데이터
        compute_metrics=compute_metrics,  # 평가 함수
    )

    # 9. (★핵심★) 모델 학습 시작
    print("\n--- KoELECTRA 모델 학습을 시작합니다 ---")
    trainer.train()

    # 10. 학습 완료 및 평가
    print("\n--- 학습 완료. 최종 모델 평가 ---")
    eval_results = trainer.evaluate()
    print(f"최종 평가 정확도: {eval_results['eval_accuracy']:.4f}")

    # 11. 학습된 모델 저장
    trainer.save_model(f"{OUTPUT_DIR}/best_model")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/best_model")D

# --- 메인 코드 실행 ---
if __name__ == "__main__":
    main()