import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import platform
import sys

# --- (★설정★) ---
# 방금 만든 통합 데이터 파일명
INPUT_FILE_NAME = "combined_labeled_for_koelectra.csv"


# --------------------

def set_korean_font():
    """운영체제에 따른 한글 폰트 설정 (그래프 깨짐 방지)"""
    system_name = platform.system()
    if system_name == 'Windows':
        # 윈도우는 '맑은 고딕'
        rc('font', family='Malgun Gothic')
    elif system_name == 'Darwin':
        # 맥은 'AppleGothic'
        rc('font', family='AppleGothic')
    else:
        # 리눅스 등 (나눔고딕이 설치되어 있다고 가정)
        rc('font', family='NanumGothic')

    # 마이너스 기호 깨짐 방지
    plt.rcParams['axes.unicode_minus'] = False


def draw_pie_charts(file_name):
    print(f"데이터를 불러옵니다: {file_name}")
    try:
        df = pd.read_csv(file_name)
    except FileNotFoundError:
        print(f"[오류] '{file_name}' 파일을 찾을 수 없습니다.")
        print("이전 단계의 '통합 코드'를 먼저 실행해서 파일을 만들어주세요.")
        return

    # 'app_name' 컬럼 확인
    if 'app_name' not in df.columns:
        print("[오류] 데이터에 'app_name' 컬럼이 없습니다.")
        print("통합 코드를 다시 실행해서 파일을 갱신해주세요.")
        return

    # 한글 폰트 설정
    set_korean_font()

    # 데이터 분리
    df_wr = df[df['app_name'] == 'wild_rift']
    df_ml = df[df['app_name'] == 'mobile_legends']

    print(f"\n--- 데이터 집계 ---")
    print(f"와일드 리프트 데이터: {len(df_wr)}건")
    print(f"모바일 레전드 데이터: {len(df_ml)}건")

    if len(df_wr) == 0 and len(df_ml) == 0:
        print("[오류] 데이터가 비어있습니다.")
        return

    # 그래프 그리기 (1행 2열)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle('게임별 리뷰 긍정/부정 비율 비교', fontsize=20)

    labels = ['부정 (Negative)', '긍정 (Positive)']
    colors = ['#ff9999', '#66b3ff']  # 빨강(부정), 파랑(긍정)
    explode = (0.05, 0.05)  # 조각 떼어내기 효과

    # 1. 와일드 리프트
    if not df_wr.empty:
        counts = df_wr['label'].value_counts().sort_index()
        # 0(부정), 1(긍정) 순서가 보장되도록 처리
        values = [counts.get(0, 0), counts.get(1, 0)]

        axes[0].pie(values, labels=labels, autopct='%1.1f%%', startangle=90,
                    colors=colors, explode=explode, shadow=True, textprops={'fontsize': 12})
        axes[0].set_title(f'리그 오브 레전드: 와일드 리프트\n(총 {len(df_wr)}건)', fontsize=14)
    else:
        axes[0].text(0.5, 0.5, '데이터 없음', ha='center')

    # 2. 모바일 레전드
    if not df_ml.empty:
        counts = df_ml['label'].value_counts().sort_index()
        values = [counts.get(0, 0), counts.get(1, 0)]

        axes[1].pie(values, labels=labels, autopct='%1.1f%%', startangle=90,
                    colors=colors, explode=explode, shadow=True, textprops={'fontsize': 12})
        axes[1].set_title(f'모바일 레전드: Bang Bang\n(총 {len(df_ml)}건)', fontsize=14)
    else:
        axes[1].text(0.5, 0.5, '데이터 없음', ha='center')

    print("\n그래프를 출력합니다...")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    draw_pie_charts(INPUT_FILE_NAME)