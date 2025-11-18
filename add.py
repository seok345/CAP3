import pandas as pd
import re
import numpy as np

# (★필수★) 1단계에서 생성된 실제 파일명으로 수정하세요.
# 예: "mobile_legends_reviews_20000.csv"
INPUT_FILE_NAME = "mobile_legends_reviews_20000.csv"

# KoELECTRA 학습용으로 저장할 파일명 (이 파일이 생성됩니다)
OUTPUT_FILE_NAME = "mobile_legends_labeled_for_koelectra.csv"


def preprocess_text(text):
    """
    KoELECTRA 학습을 위한 간단한 텍스트 전처리 함수.
    (이모티콘, 특수문자, 불필요한 공백 제거)
    """
    if not isinstance(text, str):
        return ""

    text = str(text)
    # 이모티콘 및 특수 문자 제거 (한글, 영어, 숫자, 기본 구두점만 남김)
    text = re.sub(r'[^가-힣A-Za-z0-9 .,!?]', ' ', text)
    # 과도한 ㅋ, ㅎ 자음 반복 제거 (예: ㅋㅋㅋㅋㅋ -> ㅋㅋ)
    text = re.sub(r'([ㄱ-ㅎㅏ-ㅣ])\1{2,}', r'\1\1', text)
    # 불필요한 공백을 하나의 공백으로
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_and_label_data(file_name):
    print(f"파일을 불러옵니다: {file_name}")
    try:
        df = pd.read_csv(file_name)
    except FileNotFoundError:
        print(f"[오류] '{file_name}' 파일을 찾을 수 없습니다.")
        print("1단계에서 생성된 파일명을 INPUT_FILE_NAME 변수에 정확히 입력해주세요.")
        return None
    except pd.errors.EmptyDataError:
        print(f"[오류] '{file_name}' 파일이 비어있습니다. 1단계 크롤링이 정상적으로 완료되었는지 확인하세요.")
        return None

    print(f"원본 데이터 수: {len(df)}건")

    # 1. 중복 리뷰 및 결측치 제거
    df = df.drop_duplicates(subset=['review_text'])
    df = df.dropna(subset=['review_text', 'rating'])
    print(f"중복/결측치 제거 후 데이터 수: {len(df)}건")

    # 2. 텍스트 전처리
    print("텍스트 전처리를 시작합니다... (시간이 조금 걸릴 수 있습니다)")
    df['processed_text'] = df['review_text'].apply(preprocess_text)

    # 전처리 후 비어있게 된 리뷰 다시 제거
    df = df[df['processed_text'].str.len() > 0]

    # 3. (★핵심★) 라벨링 작업
    # 별점 3점은 중립이라 제외
    df_labeled = df[df['rating'] != 3].copy()

    # 4, 5점은 1 (긍정), 1, 2점은 0 (부정)으로 라벨링
    df_labeled['label'] = np.where(df_labeled['rating'] >= 4, 1, 0)

    print(f"라벨링 완료. (3점 제외) 총 데이터: {len(df_labeled)}건")
    print(f" - 긍정 (1): {len(df_labeled[df_labeled['label'] == 1])}건")
    print(f" - 부정 (0): {len(df_labeled[df_labeled['label'] == 0])}건")

    # 4. KoELECTRA 학습에 필요한 컬럼만 선택
    final_df = df_labeled[['processed_text', 'label']]
    final_df.columns = ['document', 'label']  # (관례) NSMC와 동일한 컬럼명 사용

    return final_df


# --- 실행 ---
labeled_data = load_and_label_data(INPUT_FILE_NAME)

if labeled_data is not None:
    # 3. 최종 결과 저장
    labeled_data.to_csv(OUTPUT_FILE_NAME, index=False, encoding='utf-8-sig')

    print(f"\n[성공] KoELECTRA 학습용 파일 저장 완료: {OUTPUT_FILE_NAME}")
    print("\n--- 최종 데이터 샘플 ---")
    print(labeled_data.head())