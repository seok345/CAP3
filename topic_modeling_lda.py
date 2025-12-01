import pandas as pd
from konlpy.tag import Okt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# --- (★설정★) ---
# 방금 만든 통합/균형맞춘 데이터 파일명
INPUT_FILE_NAME = "combined_labeled_for_koelectra.csv"

# 토픽 모델링 설정
NUM_TOPICS = 5  # 추출할 토픽(주제) 개수
NUM_TOP_WORDS = 10  # 토픽별로 보여줄 상위 단어 개수
MAX_DOCS_PER_GAME = 10000  # 게임당 분석할 최대 리뷰 수 (너무 많으면 느려짐)


# --------------------

class OktTokenizer:
    """KoNLPy Okt 명사 추출 토크나이저 (불용어 처리 포함)"""

    def __init__(self):
        self.okt = Okt()
        # (★중요★) 분석에 불필요한 단어 제거 (불용어 리스트)
        self.stopwords = {
            '게임', '모바일', '레전드', '진짜', '정말', '너무', '이거', '그냥',
            '오늘', '유저', '플레이', '사람', '생각', '정도', '다시', '계속',
            '지금', '보고', '와일드', '리프트', '롤', '오브', '리그', '모레',
            '모바일레전드', '와일드리프트', '업데이트', '패치'  # 게임 이름 자체도 제외하면 더 구체적인 토픽이 나옴
        }

    def __call__(self, text):
        # 1. 명사 추출
        try:
            nouns = self.okt.nouns(str(text))
        except:
            return []

        # 2. 2글자 이상이고, 불용어가 아닌 단어만 선택
        words = [word for word in nouns if len(word) > 1 and word not in self.stopwords]
        return words


def run_topic_modeling(df, game_name):
    """특정 게임 데이터에 대해 LDA 토픽 모델링 수행"""
    print(f"\n======== [{game_name}] 토픽 모델링 시작 ========")

    # 데이터 샘플링 (너무 많으면 오래 걸림)
    if len(df) > MAX_DOCS_PER_GAME:
        print(f"  데이터가 많아 {MAX_DOCS_PER_GAME}건으로 샘플링합니다.")
        df = df.sample(n=MAX_DOCS_PER_GAME, random_state=42)

    documents = df['document'].dropna().tolist()
    print(f"  분석 대상 리뷰 수: {len(documents)}건")

    # 1. 토크나이저 준비
    print("  1. 형태소 분석(명사 추출) 중... (시간이 걸릴 수 있습니다)")
    okt_tokenizer = OktTokenizer()

    # 2. 벡터화 (CountVectorizer)
    vectorizer = CountVectorizer(
        tokenizer=okt_tokenizer,
        max_df=0.85,  # 85% 이상 문서에 등장하는 흔한 단어 제외
        min_df=5  # 5번 미만 등장하는 희귀 단어 제외
    )
    dtm = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()

    # 3. LDA 모델링
    print(f"  2. LDA 모델 학습 중... (토픽 {NUM_TOPICS}개)")
    lda_model = LatentDirichletAllocation(
        n_components=NUM_TOPICS,
        random_state=42,
        n_jobs=-1  # 모든 CPU 코어 사용
    )
    lda_model.fit(dtm)

    # 4. 결과 출력
    print(f"\n  --- [{game_name}] 주요 토픽 결과 ---")
    for topic_idx, topic in enumerate(lda_model.components_):
        top_words_idx = topic.argsort()[::-1][:NUM_TOP_WORDS]
        top_words = [feature_names[i] for i in top_words_idx]
        print(f"  [토픽 {topic_idx + 1}] {' | '.join(top_words)}")


def main():
    # 1. 데이터 로드
    print(f"데이터 파일을 불러옵니다: {INPUT_FILE_NAME}")
    try:
        df = pd.read_csv(INPUT_FILE_NAME)
    except FileNotFoundError:
        print(f"[오류] '{INPUT_FILE_NAME}' 파일을 찾을 수 없습니다.")
        return

    # 'app_name' 컬럼 확인
    if 'app_name' not in df.columns:
        print("[오류] 'app_name' 컬럼이 없습니다. 이전 통합 코드를 다시 실행해주세요.")
        return

    # 2. 게임별로 데이터 나누기
    df_wr = df[df['app_name'] == 'wild_rift']
    df_ml = df[df['app_name'] == 'mobile_legends']

    # 3. 각각 토픽 모델링 수행
    if not df_wr.empty:
        run_topic_modeling(df_wr, "리그 오브 레전드: 와일드 리프트")
    else:
        print("\n[경고] 와일드 리프트 데이터가 없습니다.")

    if not df_ml.empty:
        run_topic_modeling(df_ml, "모바일 레전드: Bang Bang")
    else:
        print("\n[경고] 모바일 레전드 데이터가 없습니다.")


if __name__ == "__main__":
    main()