import pandas as pd
from google_play_scraper import reviews, Sort
import time

# --- (★설정★) ---
# 1. 수집할 앱 ID (모바일 레전드)
app_id = 'com.mobile.legends'
app_name = 'mobile_legends'

# 2. 앱당 수집할 리뷰 수 (교수님 요구사항)
target_count = 20000

# 3. 수집할 언어/국가
lang = 'ko'
country = 'kr'
# --------------------

print(f"[{app_name} ({app_id})] 리뷰 수집 시작... (목표: {target_count}건)")
print("이 작업은 수 분에서 수십 분이 소요될 수 있습니다...")

try:
    # (★핵심★)
    # google_play_scraper의 reviews 함수를 사용합니다.
    # sort=Sort.NEWEST: 최신순으로 정렬 (가장 안정적)
    result_reviews, continuation_token = reviews(
        app_id,
        lang=lang,
        country=country,
        count=target_count,
        sort=Sort.NEWEST,  # 최신순으로 수집
        filter_score_with=None  # 모든 별점 가져오기
    )

    if not result_reviews:
        print(f"[{app_name}] 데이터를 수집하지 못했습니다. 앱 ID를 확인해주세요.")

    else:
        current_app_count = len(result_reviews)
        print(f"\n[성공] {app_name}: 총 {current_app_count}건의 리뷰를 수집했습니다.")

        # Pandas DataFrame으로 변환
        df = pd.DataFrame(result_reviews)

        # 프로젝트에 필요한 컬럼만 선택
        df_final = df[['content', 'score', 'at']].copy()
        df_final.columns = ['review_text', 'rating', 'date']

        # 2단계: 개별 CSV 파일로 저장
        file_name = f"{app_name}_reviews_{current_app_count}.csv"
        df_final.to_csv(file_name, index=False, encoding='utf-8-sig')

        print(f"\n--- {file_name} 저장 완료 ---")
        print(df_final.head())

except Exception as e:
    print(f"\n[오류 발생] {app_name} 리뷰 수집 중 오류가 발생했습니다.")
    print(f"오류 유형: {type(e)}")
    print(f"오류 메시지: {repr(e)}")