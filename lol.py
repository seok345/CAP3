import pandas as pd
from google_play_scraper import reviews

# --- (★설정★) ---
# 1. 수집할 앱 ID (와일드 리프트)
app_id = 'com.riotgames.league.wildrift'

# 2. 수집할 리뷰 수 (교수님 요구사항)
target_count = 20000

# 3. 수집할 언어/국가
lang = 'ko'
country = 'kr'
# --------------------

print(f"[{app_id}] 리뷰 수집 시작... (목표: {target_count}건)")
print("이 작업은 수 분에서 수십 분이 소요될 수 있습니다...")

try:
    # (★핵심★)
    # google_play_scraper의 reviews 함수를 사용합니다.
    # count: 수집할 리뷰 개수
    # lang/country: 한국어/한국
    result_reviews, continuation_token = reviews(
        app_id,
        lang=lang,
        country=country,
        count=target_count,
        filter_score_with=None  # (중요) 별점 필터 없이 모든 리뷰 가져오기
    )

    if not result_reviews:
        print("데이터를 수집하지 못했습니다. 앱 ID를 확인해주세요.")

    else:
        print(f"\n[성공] 총 {len(result_reviews)}건의 리뷰를 수집했습니다.")

        # 3. Pandas DataFrame으로 변환
        df = pd.DataFrame(result_reviews)

        # (★프로젝트용★)
        # KoELECTRA와 토픽 모델링에 필요한 컬럼만 선택합니다.
        # 'content' -> 리뷰 텍스트
        # 'score' -> 별점 (1~5점)
        df_final = df[['content', 'score', 'at']]

        # 교수님 요구사항에 맞게 컬럼명 변경
        df_final.columns = ['review_text', 'rating', 'date']

        # 4. CSV 파일로 저장
        file_name = f"wild_rift_reviews_{len(df_final)}.csv"
        df_final.to_csv(file_name, index=False, encoding='utf-8-sig')

        print(f"\n--- {file_name} 저장 완료 ---")
        print(df_final.head())

except Exception as e:
    print(f"\n[오류 발생] 리뷰 수집 중 오류가 발생했습니다.")
    print(f"오류 유형: {type(e)}")
    print(f"오류 메시지: {repr(e)}")