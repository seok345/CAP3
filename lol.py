import pandas as pd
from google_play_scraper import reviews, Sort
import time
import random

# --- (★설정★) ---
app_id = 'com.riotgames.league.wildrift'  # 와일드 리프트 앱 ID
app_name = 'wild_rift'
target_count = 40000  # 목표: 5만 건
batch_size = 2000  # 한 번에 수집할 양 (2000건 권장)
lang = 'ko'
country = 'kr'
# --------------------

print(f"[{app_name}] 대용량 리뷰 수집 시작... (목표: {target_count}건)")
print("안전한 수집을 위해 조금씩 나누어 수집합니다. 시간이 오래 걸릴 수 있습니다.")

all_reviews = []
continuation_token = None
batch_num = 1

try:
    while len(all_reviews) < target_count:
        # 남은 수량 계산
        remaining = target_count - len(all_reviews)
        current_count = min(batch_size, remaining)

        print(f"\n[{batch_num}회차] {current_count}건 요청 중... (현재 수집됨: {len(all_reviews)}건)")

        # 리뷰 수집 (이어하기 토큰 사용)
        result, continuation_token = reviews(
            app_id,
            lang=lang,
            country=country,
            count=current_count,
            sort=Sort.NEWEST,  # 최신순 정렬
            filter_score_with=None,  # 모든 별점 포함
            continuation_token=continuation_token
        )

        if not result:
            print("더 이상 수집할 리뷰가 없거나, 접근이 제한되었습니다.")
            break

        all_reviews.extend(result)
        print(f"  -> 성공! {len(result)}건 추가됨.")

        # 중간 저장 (10,000건마다)
        if len(all_reviews) % 10000 < batch_size and len(all_reviews) > 0:
            temp_df = pd.DataFrame(all_reviews)
            temp_df.to_csv(f"{app_name}_temp_{len(all_reviews)}.csv", index=False, encoding='utf-8-sig')
            print(f"  -> [중간 저장] {len(all_reviews)}건 저장 완료")

        if continuation_token is None:
            print("모든 리뷰를 다 가져왔습니다 (마지막 페이지 도달).")
            break

        # 봇 차단 방지: 랜덤하게 2~5초 대기
        sleep_time = random.uniform(2, 5)
        print(f"  -> 차단 방지를 위해 {sleep_time:.1f}초 대기...")
        time.sleep(sleep_time)

        batch_num += 1

except Exception as e:
    print(f"\n[오류 발생] 수집 중단. 현재까지 수집된 데이터를 저장합니다.")
    print(f"오류 메시지: {e}")

# 최종 결과 저장
if all_reviews:
    print(f"\n[최종 완료] 총 {len(all_reviews)}건 수집 성공!")

    df = pd.DataFrame(all_reviews)

    # 필요한 컬럼만 선택 및 이름 변경
    df_final = df[['content', 'score', 'at']].copy()
    df_final.columns = ['review_text', 'rating', 'date']

    # 최종 파일 저장
    file_name = f"{app_name}_reviews_total_{len(df_final)}.csv"
    df_final.to_csv(file_name, index=False, encoding='utf-8-sig')
    print(f"--- 최종 파일 저장 완료: {file_name} ---")
    print(df_final.head())
else:
    print("\n수집된 데이터가 없습니다.")