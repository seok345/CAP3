import pandas as pd
import re
import numpy as np
from google_play_scraper import reviews, Sort
import os

# --- (★설정 1★) ---
# 1단계에서 수집한 "모바일 레전드" 원본 파일명 (4만 건짜리)
MLBB_RAW_FILE = "mobile_legends_reviews_total_40000.csv"
# --------------------

# --- (★설정 2★) ---
# 새로 수집할 "와일드 리프트" 설정
WR_APP_ID = 'com.riotgames.league.wildrift'
WR_APP_NAME = 'wild_rift'
TARGET_COUNT = 20000  # 수집할 리뷰 수
LANG = 'ko'
COUNTRY = 'kr'

# 와일드 리프트 저장용 파일명
WR_RAW_FILE = f"{WR_APP_NAME}_reviews_{TARGET_COUNT}.csv"
# --------------------

# --- (★설정 3★) ---
# 최종 결과물 파일명
OUTPUT_MODELING_FILE = "combined_labeled_for_koelectra.csv"
OUTPUT_VERIFICATION_FILE = "combined_VERIFICATION.csv"


# --------------------

def preprocess_text(text):
    if not isinstance(text, str): return ""
    text = str(text)
    text = re.sub(r'[^가-힣A-Za-z0-9 .,!?]', ' ', text)
    text = re.sub(r'([ㄱ-ㅎㅏ-ㅣ])\1{2,}', r'\1\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def apply_label(rating):
    if rating >= 4: return 1  # 긍정
    if rating <= 2: return 0  # 부정
    return -1  # 중립 (3점)


def scrape_wild_rift():
    if os.path.exists(WR_RAW_FILE):
        print(f"[{WR_RAW_FILE}] 파일이 이미 존재하므로 크롤링을 건너뜁니다.")
        return True

    print(f"[{WR_APP_NAME}] 리뷰 수집 시작... (목표: {TARGET_COUNT}건)")
    try:
        result_reviews, _ = reviews(
            WR_APP_ID,
            lang=LANG,
            country=COUNTRY,
            count=TARGET_COUNT,
            sort=Sort.NEWEST,
            filter_score_with=None
        )

        if not result_reviews:
            print("데이터 수집 실패.")
            return False

        df = pd.DataFrame(result_reviews)
        df_final = df[['content', 'score', 'at']].copy()
        df_final.columns = ['review_text', 'rating', 'date']

        df_final.to_csv(WR_RAW_FILE, index=False, encoding='utf-8-sig')
        print(f"[성공] 와일드 리프트 {len(df_final)}건 수집 완료.")
        return True
    except Exception as e:
        print(f"[오류] 수집 중 에러 발생: {e}")
        return False


def main():
    # 1. 와일드 리프트 데이터 준비
    if not scrape_wild_rift():
        return

    print("\n--- 데이터 불러오기 및 개수 맞추기 (Balancing) ---")

    try:
        # 파일 불러오기
        df_ml = pd.read_csv(MLBB_RAW_FILE)
        df_ml['app_name'] = 'mobile_legends'

        df_wr = pd.read_csv(WR_RAW_FILE)
        df_wr['app_name'] = 'wild_rift'

        print(f"1. 원본 데이터 수")
        print(f"   - 모바일 레전드: {len(df_ml)}건")
        print(f"   - 와일드 리프트: {len(df_wr)}건")

        # (★핵심★) 두 데이터 중 더 적은 쪽의 개수에 맞춤
        min_count = min(len(df_ml), len(df_wr))
        print(f"\n2. 1:1 비율을 위해 각각 {min_count}건으로 맞춥니다.")

        # 랜덤 샘플링으로 개수 맞추기 (random_state=42로 고정하여 결과 일정하게)
        if len(df_ml) > min_count:
            df_ml = df_ml.sample(n=min_count, random_state=42)

        if len(df_wr) > min_count:
            df_wr = df_wr.sample(n=min_count, random_state=42)

        print(f"   - [조정 완료] 모바일 레전드: {len(df_ml)}건")
        print(f"   - [조정 완료] 와일드 리프트: {len(df_wr)}건")

    except Exception as e:
        print(f"[오류] 파일 읽기 실패: {e}")
        return

    # 3. 데이터 합치기
    df_combined = pd.concat([df_ml, df_wr], ignore_index=True)
    print(f"\n데이터 병합 완료: 총 {len(df_combined)}건")

    # 4. 전처리 및 라벨링
    print("전처리 중... (중복 제거 안 함)")

    # 결측치만 제거
    df_combined = df_combined.dropna(subset=['review_text', 'rating'])

    # 텍스트 청소 -> 'document' 컬럼에 저장
    df_combined['document'] = df_combined['review_text'].apply(preprocess_text)
    df_combined = df_combined[df_combined['document'].str.len() > 0]

    # 라벨링
    print("라벨링 중...")
    df_combined['label'] = df_combined['rating'].apply(apply_label)

    # 5. 최종 저장 (3점 제외)
    df_final = df_combined[df_combined['label'] != -1].copy()

    # 결과물 1: 모델링용
    modeling_df = df_final[['document', 'label', 'app_name']]
    modeling_df.to_csv(OUTPUT_MODELING_FILE, index=False, encoding='utf-8-sig')

    # 결과물 2: 검증용
    verification_df = df_final[['app_name', 'review_text', 'rating', 'document', 'label']]
    verification_df.columns = ['app_name', 'review_text', 'rating', 'processed_text', 'label']
    verification_df.to_csv(OUTPUT_VERIFICATION_FILE, index=False, encoding='utf-8-sig')

    print(f"\n[완료] '{OUTPUT_MODELING_FILE}' 저장되었습니다.")
    print(f"최종 데이터 수: {len(modeling_df)}건 (두 게임 비슷하게 맞춰짐)")
    print("\n이제 visualize.py를 다시 실행해보세요! 리뷰 수가 비슷하게 나올 겁니다.")


if __name__ == "__main__":
    main()