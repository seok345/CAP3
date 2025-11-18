import pandas as pd
from konlpy.tag import Okt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# --- (★설정★) ---
# 2단계에서 만든 라벨링된 데이터 파일
INPUT_FILE_NAME = "mobile_legends_labeled_for_koelectra.csv"

# 1. 토픽 모델링에 사용할 최대 리뷰 수 (너무 많으면 오래 걸림)
MAX_DOCUMENTS = 10000

# 2. 추출할 토픽(주제)의 개수
NUM_TOPICS = 5

# 3. 토픽별로 보여줄 상위 키워드 개수
NUM_TOP_WORDS = 10


# --------------------

class OktTokenizer:
    """
    LDA 토픽 모델링을 위한 KoNLPy Okt 명사 추출 토크나이저
    (불용어 처리 포함)
    """

    def __init__(self):
        self.okt = Okt()
        # (★중요★) 토픽 모델링에 불필요한 단어 (불용어)
        # (게임 리뷰에 맞춰 '게임', '모바일', '레전드' 등 추가)
        self.stopwords = {'게임', '모바일', '레전드', '진짜', '정말', '너무',
                          '이거', '그냥', '오늘', '유저', '플레이'}

    def __call__(self, text):
        # 명사 추출
        nouns = self.okt.nouns(text)

        # 2글자 이상, 불용어 제외
        words = [word for word in nouns if len(word) > 1 and word not in self.stopwords]
        return words


def load_data(file_name, max_docs):
    """CSV 파일을 불러와 document 리스트로 반환"""
    print(f"데이터 파일을 불러옵니다: {file_name}")
    try:
        df = pd.read_csv(file_name)
        df = df.dropna(subset=['document'])

        # 샘플링
        if len(df) > max_docs:
            print(f"원본 데이터가 너무 많습니다. {len(df)}건 -> {max_docs}건으로 샘플링합니다.")
            df = df.sample(n=max_docs, random_state=42)

        return df['document'].tolist()

    except FileNotFoundError:
        print(f"[오류] '{file_name}' 파일을 찾을 수 없습니다.")
        return None


def display_topics(model, feature_names, num_top_words):
    """LDA 모델의 토픽별 상위 키워드를 출력하는 함수"""
    print("\n--- 토픽 모델링 결과 (주제별 상위 키워드) ---")
    for topic_idx, topic in enumerate(model.components_):
        # 토픽별 상위 단어 추출
        top_words_idx = topic.argsort()[::-1][:num_top_words]
        top_words = [feature_names[i] for i in top_words_idx]

        print(f"[토픽 #{topic_idx + 1}] {' | '.join(top_words)}")


def main():
    # 1. 데이터 로드 (리뷰 텍스트 리스트)
    documents = load_data(INPUT_FILE_NAME, MAX_DOCUMENTS)
    if documents is None:
        return

    # 2. 한국어 명사 추출기 (OktTokenizer) 준비
    print("\nKoNLPy Okt 토크나이저(명사 추출기)를 준비합니다...")
    okt_tokenizer = OktTokenizer()

    # 3. 문서-단어 행렬 (DTM) 생성 (CountVectorizer)
    # (min_df=5: 최소 5개 문서, max_df=0.85: 최대 85% 문서에 등장한 단어만)
    print("문서-단어 행렬(DTM)을 생성합니다... (시간이 매우 오래 걸릴 수 있습니다)")
    vectorizer = CountVectorizer(
        tokenizer=okt_tokenizer,
        max_df=0.85,
        min_df=5
    )
    dtm = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()

    # 4. (★핵심★) LDA 토픽 모델링 실행
    print(f"DTM 생성 완료. LDA 토픽 모델링을 시작합니다... (토픽 {NUM_TOPICS}개)")
    lda_model = LatentDirichletAllocation(
        n_components=NUM_TOPICS,
        random_state=42,
        n_jobs=-1  # 모든 CPU 코어 사용
    )
    lda_model.fit(dtm)

    # 5. 결과 출력
    display_topics(lda_model, feature_names, NUM_TOP_WORDS)


# --- 메인 코드 실행 ---
if __name__ == "__main__":
    main()