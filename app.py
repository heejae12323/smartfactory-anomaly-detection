from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt  # (not used directly; kept for compatibility)
import seaborn as sns  # (optional, not used but kept from original)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
# 아래 LangChain 관련 import는 선택 기능(LLM 답변)에 사용
from langchain_openai import ChatOpenAI  # (optional)
from langchain.memory import ConversationBufferMemory  # (kept)
from langchain.chains import ConversationChain  # (kept)
from langchain.prompts import PromptTemplate  # (kept)
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import glob
from datetime import datetime
import json
import re
from typing import Dict, List, Any, Optional, Tuple

# =============================
# Page & Global Styles
# =============================
st.set_page_config(
    page_title="🏭 Smart Factory AI Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Global CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .status-success {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 0.8rem;
        border-radius: 8px;
        border-left: 4px solid #00c851;
        margin: 0.5rem 0;
    }
    .status-warning {
        background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
        padding: 0.8rem;
        border-radius: 8px;
        border-left: 4px solid #ff6b6b;
        margin: 0.5rem 0;
    }
    .chat-container { background-color: #f8f9fa; border-radius: 10px; padding: 20px; margin: 10px 0; }
    .user-message { background-color: #007bff; color: white; padding: 10px 15px; border-radius: 18px; margin: 5px 0; max-width: 80%; float: right; clear: both; }
    .ai-message { background-color: #e9ecef; color: #333; padding: 10px 15px; border-radius: 18px; margin: 5px 0; max-width: 80%; float: left; clear: both; }
    .insight-card { background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin: 10px 0; border-radius: 5px; }
    .question-card { background-color: #d4edda; border-left: 4px solid #28a745; padding: 15px; margin: 10px 0; border-radius: 5px; cursor: pointer; transition: background-color 0.3s; }
    .question-card:hover { background-color: #c3e6cb; }
    .analysis-summary { background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 20px; margin: 20px 0; }
    .metric-card h3 { margin: 0; font-size: 1rem; opacity: 0.9; }
    .metric-card h1 { margin: 0.5rem 0; font-size: 2.2rem; font-weight: bold; }
    .metric-card p { margin: 0; font-size: 0.9rem; opacity: 0.85; }
</style>
""",
    unsafe_allow_html=True,
)

# =============================
# Helper: SmartFactoryLLMAnalyzer
# =============================
class SmartFactoryLLMAnalyzer:
    def __init__(self):
        self.analysis_context: Dict[str, Any] = {}
        self.conversation_history: List[Dict[str, Any]] = []

    def update_context(
        self,
        df: pd.DataFrame,
        unit_status: Optional[pd.DataFrame] = None,
        anomaly_data: Optional[Dict] = None
    ):
        sensor_cols = [c for c in df.columns if 'sensor' in c]
        self.analysis_context = {
            'total_units': int(df['unit'].nunique()) if 'unit' in df.columns else 0,
            'total_records': int(len(df)),
            'avg_rul': float(df['RUL'].mean()) if 'RUL' in df.columns else 0.0,
            'min_rul': float(df['RUL'].min()) if 'RUL' in df.columns else 0.0,
            'max_rul': float(df['RUL'].max()) if 'RUL' in df.columns else 0.0,
            'critical_units': int(df[df['RUL'] < 30]['unit'].nunique()) if 'RUL' in df.columns else 0,
            'sensor_columns': sensor_cols,
            'anomaly_rate': float((df['anomaly'] == -1).mean() * 100) if 'anomaly' in df.columns else 0.0,
            'unit_status': unit_status.to_dict('records') if unit_status is not None else [],
            'data_summary': df.describe().to_dict() if not df.empty else {},
        }

    def generate_insights(self) -> List[str]:
        insights: List[str] = []
        ctx = self.analysis_context
        if ctx.get('avg_rul', 0) < 50:
            insights.append(f"⚠️ **주의**: 평균 RUL이 {ctx['avg_rul']:.1f}로 낮아 전반적인 장비 상태가 우려됩니다.")
        if ctx.get('anomaly_rate', 0) > 10:
            insights.append(f"🚨 **경고**: 이상 징후율이 {ctx['anomaly_rate']:.1f}%로 높습니다. 즉시 점검이 필요합니다.")
        elif ctx.get('anomaly_rate', 0) > 5:
            insights.append(f"⚡ **주의**: 이상 징후율이 {ctx['anomaly_rate']:.1f}%입니다. 모니터링을 강화하세요.")
        total_units = max(ctx.get('total_units', 1), 1)
        critical_ratio = (ctx.get('critical_units', 0) / total_units) * 100
        if critical_ratio > 20:
            insights.append(f"🔥 **긴급**: 전체 장비의 {critical_ratio:.1f}%({ctx['critical_units']}대)가 위험 상태입니다.")
        if ctx.get('total_records', 0) < 1000:
            insights.append("📊 **정보**: 데이터 수가 적어 분석 정확도가 제한될 수 있습니다.")
        return insights

    def generate_questions(self) -> List[str]:
        questions = [
            "가장 위험한 장비는 어떤 것들인가요?",
            "이상 징후가 가장 많이 발생하는 센서는 무엇인가요?",
            "예방정비를 우선적으로 해야 할 장비는?",
            "RUL이 가장 짧은 장비들의 공통점은?",
            "센서별 이상 패턴 분석 결과는?",
            "장비별 고장 예측 시기는 언제인가요?",
            "비용 효율적인 정비 계획을 제안해주세요",
            "이상 징후와 RUL 간의 상관관계는?",
            "센서 데이터 트렌드 분석 결과는?",
            "장비 성능 개선을 위한 권고사항은?",
        ]
        ctx = self.analysis_context
        if ctx.get('critical_units', 0) > 0:
            questions.insert(0, f"위험 상태인 {ctx['critical_units']}대 장비의 상세 분석은?")
        if ctx.get('anomaly_rate', 0) > 5:
            questions.insert(1, f"이상 징후율 {ctx['anomaly_rate']:.1f}%의 주요 원인은?")
        return questions[:8]

    def analyze_question(self, question: str) -> str:
        ctx = self.analysis_context
        q = question.lower()
        if any(w in q for w in ['위험', '위험한', 'critical', '긴급']):
            units = ctx.get('unit_status', [])
            critical_units = [u for u in units if float(u.get('RUL', 100)) < 30]
            if critical_units:
                unit_list = ', '.join([f"장비 {int(u['unit'])}" for u in critical_units[:5] if 'unit' in u])
                return (
                    f"🚨 **위험 장비 분석**\n\n가장 위험한 장비들: {unit_list}\n\n이들 장비는 RUL이 30 미만으로 즉시 점검이 필요합니다. "
                    "우선순위에 따라 예방정비를 실시하세요."
                )
            return "현재 위험 상태인 장비가 식별되지 않았습니다."
        elif any(w in q for w in ['이상', 'anomaly', '징후', '패턴']):
            anomaly_rate = ctx.get('anomaly_rate', 0)
            return (
                f"📊 **이상 징후 분석**\n\n전체 이상 징후율: {anomaly_rate:.2f}%\n\n"
                "이상 징후는 Isolation Forest 알고리즘으로 탐지되었으며, 정상 범위를 벗어난 센서 값들을 식별합니다. "
                "높은 이상율을 보이는 장비는 우선 점검 대상입니다."
            )
        elif any(w in q for w in ['rul', '수명', '잔여', '예측']):
            avg_rul = ctx.get('avg_rul', 0)
            min_rul = ctx.get('min_rul', 0)
            return (
                f"⏰ **RUL 분석**\n\n평균 RUL: {avg_rul:.1f} 사이클\n최소 RUL: {min_rul:.1f} 사이클\n\n"
                "잔여 유용 수명(RUL)이 30 미만인 장비는 즉시 정비가 필요하며, 50 미만인 장비는 예방정비 대상으로 분류됩니다."
            )
        elif any(w in q for w in ['센서', 'sensor', '측정값']):
            sensor_count = len(ctx.get('sensor_columns', []))
            return (
                f"🔧 **센서 분석**\n\n총 센서 수: {sensor_count}개\n\n"
                "각 센서는 온도, 압력, 진동 등 다양한 물리량을 측정합니다. 센서별 변동성과 이상 패턴을 분석하여 고장 징후를 조기에 감지할 수 있습니다."
            )
        elif any(w in q for w in ['정비', 'maintenance', '계획', '우선순위']):
            critical_units = ctx.get('critical_units', 0)
            return (
                f"🛠️ **정비 계획 권고**\n\n1. 즉시 정비 필요: {critical_units}대 (RUL < 30)\n2. 예방정비 대상: RUL 30-50 구간 장비\n3. 정상 운영: RUL > 50 장비\n\n"
                "위험도에 따라 우선순위를 정하여 체계적인 정비를 실시하세요."
            )
        elif any(w in q for w in ['비용', '효율', 'cost', 'roi']):
            return (
                "💰 **비용 효율성 분석**\n\n예방정비를 통해 예상되는 효과:\n- 비계획정지 감소: 70-80%\n- 정비비용 절감: 30-40%\n- 장비 수명 연장: 20-30%\n\n위험 장비 우선 정비로 ROI를 극대화할 수 있습니다."
            )
        else:
            return (
                f"📈 **종합 분석 결과**\n\n- 전체 장비: {ctx.get('total_units', 0)}대\n- 총 데이터: {ctx.get('total_records', 0):,}건\n- 평균 RUL: {ctx.get('avg_rul', 0):.1f}\n- 이상 징후율: {ctx.get('anomaly_rate', 0):.2f}%\n\n"
                "상세한 분석이 필요한 특정 영역이 있으시면 구체적으로 질문해주세요."
            )

# =============================
# Environment & Header
# =============================
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

st.markdown('<h1 class="main-header">🏭 Smart Factory AI Dashboard</h1>', unsafe_allow_html=True)
st.markdown("---")

# =============================
# Session State init
# =============================
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = SmartFactoryLLMAnalyzer()
if 'chat_history' not in st.session_state:
    st.session_state.chat_history: List[Dict[str, Any]] = []
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

# =============================
# Sidebar: Controls & Data Source
# =============================
with st.sidebar:
    st.markdown("### 🎛️ 제어판")
    if api_key:
        st.markdown('<div class="status-success">✅ AI 시스템 연결됨</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-warning">⚠️ API Key 설정 필요</div>', unsafe_allow_html=True)

    # OpenAI LLM toggle
    if 'use_llm' not in st.session_state:
        st.session_state.use_llm = bool(api_key)
    st.session_state.use_llm = st.checkbox(
        "🤖 OpenAI 응답 사용",
        value=st.session_state.use_llm,
        disabled=not bool(api_key),
        help="켜면 채팅/예상질문 응답을 OpenAI LLM이 생성합니다."
    )

    # ---- 현재 시점/윈도우 설정 ----
    st.markdown("### ⏱️ 분석 기준 시점")
    snapshot_pct = st.slider(
        "현재 시점 (수명 대비 %)", min_value=10, max_value=95, value=60, step=5,
        help="훈련 데이터는 고장까지 기록되므로, 실제 운영처럼 각 장비 수명의 몇 % 지점을 '현재'라고 가정합니다."
    )
    anom_window = st.number_input(
        "이상율 계산 윈도우(사이클)", min_value=5, max_value=300, value=30, step=5,
        help="현재 시점 직전 N사이클만 모아서 이상율을 계산합니다."
    )
    critical_rul_thresh = st.number_input(
        "위험 장비 RUL 임계치", min_value=1, max_value=200, value=30, step=1
    )

    st.markdown("### 📁 데이터 선택")

    data_source = st.radio(
        "데이터 소스를 선택하세요:",
        ["📂 데이터 폴더에서 선택", "📤 파일 업로드"],
        help="기존 데이터 폴더의 파일을 선택하거나 새로운 파일을 업로드할 수 있습니다",
    )

    uploaded_file = None

    if data_source == "📂 데이터 폴더에서 선택":
        train_files = glob.glob("data/train_*.txt")
        train_files = [os.path.basename(f) for f in train_files]

        if train_files:
            st.markdown("#### 📊 사용 가능한 데이터 파일")
            for file_name in train_files:
                file_path = os.path.join("data", file_name)
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    if file_size > 5 * 1024 * 1024:
                        size_color = "🔴"
                    elif file_size > 2 * 1024 * 1024:
                        size_color = "🟡"
                    else:
                        size_color = "🟢"
                    st.markdown(f"{size_color} **{file_name}** ({file_size/1024:.1f}KB)")
                else:
                    st.markdown(f"❌ **{file_name}** (파일 없음)")

            st.markdown("---")
            selected_file = st.selectbox(
                "🎯 분석할 Train 파일 선택:", train_files, help="분석할 train 데이터 파일을 선택하세요"
            )

            if selected_file:
                file_path = os.path.join("data", selected_file)
                file_size = os.path.getsize(file_path)
                # Dummy object to carry name/size (no read/seek)
                uploaded_file = type("UploadedFile", (), {"name": selected_file, "size": file_size})()
        else:
            st.warning("📁 data 폴더에 train_*.txt 파일을 찾을 수 없습니다.")

        st.markdown("---")
        st.markdown("#### 📁 전체 데이터 폴더 정보")
        all_files = glob.glob("data/*.txt")
        if all_files:
            file_categories = {
                "🚂 Train 데이터": [f for f in all_files if "train" in str(f).lower()],
                "🧪 Test 데이터": [f for f in all_files if "test" in str(f).lower()],
                "⏰ RUL 데이터": [f for f in all_files if "rul" in str(f).lower()],
                "📄 기타": [
                    f for f in all_files if not any(x in str(f).lower() for x in ["train", "test", "rul"])
                ],
            }
            for category, files in file_categories.items():
                if files:
                    st.markdown(f"**{category}**")
                    for file_path in files:
                        file_name = os.path.basename(file_path)
                        if os.path.exists(file_path):
                            file_size = os.path.getsize(file_path)
                            st.markdown(f"  • {file_name} ({file_size/1024:.1f}KB)")
                        else:
                            st.markdown(f"  • {file_name} (파일 없음)")
        else:
            st.info("📁 data 폴더가 비어있습니다.")
    else:
        uploaded_file = st.file_uploader(
            "센서 데이터 파일을 선택하세요", type=["txt", "csv"], help="CSV 또는 TXT 형식의 센서 데이터를 업로드하세요"
        )
        if uploaded_file:
            st.success(f"📄 {uploaded_file.name}")
            st.info(f"크기: {uploaded_file.size/1024:.1f}KB")

# =============================
# LLM Answer Helper (safe; no unterminated f-strings)
# =============================
def llm_answer(question: str, ctx: dict, api_key: str) -> str:
    """OpenAI(LangChain)로 답변 생성. 실패 시 규칙기반 폴백."""
    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=api_key)

        summary_lines = [
            f"- 총 장비: {ctx.get('total_units', 0)}대",
            f"- 평균 RUL: {ctx.get('avg_rul', 0):.1f}",
            f"- 위험 장비: {ctx.get('critical_units', 0)}대",
            f"- 이상 징후율: {ctx.get('anomaly_rate', 0):.2f}%",
        ]
        summary = "\n".join(summary_lines)

        prompt_lines = [
            "당신은 스마트팩토리 설비 상태 분석 전문가입니다.",
            "다음 데이터 요약을 참고해 한국어로 간결하고 실무형 조언을 제시하세요.",
            "",
            "[데이터 요약]",
            summary,
            "",
            "[질문]",
            question,
            "",
            "[요구]",
            "- 핵심 수치 1~2개 인용",
            "- 실행 가능한 권고 3개 이하",
            "- 마크다운으로 5~8줄",
        ]
        prompt = "\n".join(prompt_lines)

        resp = llm.invoke(prompt)
        content = getattr(resp, "content", None)
        return content if content is not None else str(resp)

    except Exception as e:
        # LLM 실패 시 규칙기반 답변으로 폴백
        fallback = (
            st.session_state.analyzer.analyze_question(question)
            if "analyzer" in st.session_state else ""
        )
        return f"(LLM 호출 실패: {e})\n\n{fallback}"

# =============================
# Helper: delimiter detection & snapshot
# =============================
def _detect_separator_from_text(first_line: str) -> str:
    """첫 줄을 보고 구분자 추정."""
    if "\t" in first_line:
        return "\t"
    if "," in first_line:
        return ","
    if " " in first_line and len(first_line.split()) > 1:
        return r"\s+"
    return ","  # fallback

def make_snapshot(df_all: pd.DataFrame, pct: int = 60, window: int = 30) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    각 unit별 수명 대비 pct% 지점을 '현재'로 보고 그 시점의 단일 행(snap_df)을 뽑고,
    그 시점 직전 window사이클 구간만 모은 win_df를 반환.
    """
    if df_all.empty:
        return df_all.iloc[0:0].copy(), df_all.iloc[0:0].copy()

    life = df_all.groupby("unit")["time"].max().rename("max_time")
    tmp = df_all.merge(life, on="unit", how="left").copy()
    tmp["time_snap"] = (tmp["max_time"] * (pct / 100.0)).astype(int).clip(lower=1)

    snap_df = (
        tmp[tmp["time"] <= tmp["time_snap"]]
        .sort_values(["unit", "time"])
        .groupby("unit")
        .tail(1)
        .copy()
    )

    win_df = tmp.merge(
        snap_df[["unit", "time", "time_snap"]].rename(columns={"time": "snap_time"}),
        on="unit",
        how="left",
        suffixes=("", "_snap"),
    )
    win_df = win_df[
        (win_df["time"] >= (win_df["time_snap"] - window + 1)) & (win_df["time"] <= win_df["time_snap"])
    ].copy()

    return snap_df, win_df

# =============================
# Main: Data Load & Analysis
# =============================
df: Optional[pd.DataFrame] = None
unit_status: Optional[pd.DataFrame] = None

if uploaded_file is not None:
    try:
        with st.spinner("🔄 데이터를 분석하고 있습니다..."):
            # Case A: st.file_uploader (has read/seek)
            if hasattr(uploaded_file, "read") and callable(getattr(uploaded_file, "read", None)) and hasattr(uploaded_file, "seek"):
                uploaded_file.seek(0)
                try:
                    # pandas python engine can infer sep when sep=None
                    df = pd.read_csv(uploaded_file, sep=None, engine="python", header=None, encoding="utf-8")
                except Exception:
                    # Fallback: manual first-line detection
                    uploaded_file.seek(0)
                    content = uploaded_file.read().decode("utf-8", errors="ignore")
                    lines = content.splitlines()
                    first_line = lines[0] if lines else ''
                    sep = _detect_separator_from_text(first_line)
                    from io import StringIO
                    df = pd.read_csv(StringIO(content), sep=sep, header=None, engine="python", encoding="utf-8")
            # Case B: Local file chosen from folder (no read/seek)
            else:
                file_path = os.path.join("data", uploaded_file.name)
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    first_line = f.readline().strip()
                sep = _detect_separator_from_text(first_line)
                df = pd.read_csv(file_path, sep=sep, header=None, engine="python", encoding="utf-8")

        # ===== Preprocess =====
        if df is not None and not df.empty:
            try:
                expected_cols = 26
                if df.shape[1] == expected_cols:
                    df.columns = ["unit", "time", "os1", "os2", "os3"] + [f"sensor_{i}" for i in range(1, 22)]
                else:
                    df.columns = [f"col_{i}" for i in range(df.shape[1])]
                    if df.shape[1] >= 2:
                        df.rename(columns={df.columns[0]: 'unit', df.columns[1]: 'time'}, inplace=True)

                # numeric conversion
                for col in df.columns:
                    if col in ['unit', 'time'] or 'sensor' in col or 'os' in col:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                # essential keys
                df = df.dropna(subset=['unit', 'time'])
                if df.empty:
                    st.error("데이터 처리 후 유효한 행이 없습니다.")
                    st.stop()

                # === 여기부터 추가: unit/time 타입 강제 & 정렬 ===
                df["unit"] = pd.to_numeric(df["unit"], errors="coerce").astype("Int64")
                df["time"] = pd.to_numeric(df["time"], errors="coerce").astype("Int64")
                # 타입 강제 후 혹시 생긴 NaN 방어
                df = df.dropna(subset=["unit", "time"])
                if df.empty:
                    st.error("unit/time 정제 후 유효한 행이 없습니다.")
                    st.stop()
                # 진짜 정수로 확정 + 정렬
                df["unit"] = df["unit"].astype(int)
                df["time"] = df["time"].astype(int)
                df = df.sort_values(["unit", "time"]).reset_index(drop=True)
                # === 추가 끝 ===

            except Exception as e:
                st.error(f"데이터 전처리 중 오류: {str(e)}")
                st.stop()

            # ===== RUL & Anomaly Detection =====
            if 'unit' in df.columns and 'time' in df.columns:
                try:
                    if not pd.api.types.is_numeric_dtype(df['unit']) or not pd.api.types.is_numeric_dtype(df['time']):
                        st.error("unit과 time 컬럼이 숫자 형식이어야 합니다.")
                        st.stop()

                    max_cycle = df.groupby('unit')["time"].max().reset_index()
                    max_cycle.columns = ["unit", "max_time"]
                    df = df.merge(max_cycle, on='unit', how='left')
                    df['RUL'] = df['max_time'] - df['time']
                    df.drop(columns=['max_time'], inplace=True)

                    sensor_cols = [c for c in df.columns if ('sensor' in c) or c.startswith('col_')]
                    sensor_cols = [c for c in sensor_cols if c not in ['unit', 'time', 'RUL']]

                    selected_sensors: List[str] = []
                    for c in sensor_cols:
                        if pd.api.types.is_numeric_dtype(df[c]):
                            std_val = float(df[c].std(skipna=True))
                            if not pd.isna(std_val) and std_val > 0.01:
                                selected_sensors.append(c)

                    if len(selected_sensors) > 0:
                        try:
                            sensor_data = df[selected_sensors].copy()
                            sensor_data = sensor_data.fillna(sensor_data.mean())
                            scaler = StandardScaler()
                            df_scaled = scaler.fit_transform(sensor_data)
                            iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
                            df['anomaly'] = iso.fit_predict(df_scaled)

                            # ===== Dashboard Metrics (스냅샷 기준) =====
                            st.markdown("## 📊 실시간 모니터링 대시보드")

                            # 현재시점 스냅샷/윈도우 데이터
                            snap_df, win_df = make_snapshot(df, pct=int(snapshot_pct), window=int(anom_window))

                            # 스냅샷이 비는 경우 방어 로직 (time_snap이 너무 작거나 타입 문제일 때)
                            if len(snap_df) == 0:
                                # 각 unit의 최솟값 1행이라도 현재시점으로 간주
                                snap_df = df.sort_values(["unit", "time"]).groupby("unit").head(1).copy()
                                # 윈도우도 최소로 재구성
                                win_df = df.merge(
                                    snap_df[["unit", "time"]].rename(columns={"time": "snap_time"}),
                                    on="unit", how="left"
                                )
                                win_df = win_df[(win_df["time"] <= win_df["snap_time"]) & (win_df["time"] >= win_df["snap_time"] - int(anom_window) + 1)].copy()

                            # 총 장비(스냅샷에 실제로 존재하는 유닛 기준)
                            total_units = int(snap_df["unit"].nunique())

                            # 위험 장비: 현재시점 RUL < 임계치
                            critical_units_now = int((snap_df["RUL"] < int(critical_rul_thresh)).sum())

                            # 이상 카운트/율: 현재시점 직전 window 구간에서만 계산
                            if "anomaly" in win_df.columns and len(win_df) > 0:
                                anomaly_count = int((win_df["anomaly"] == -1).sum())
                                total_count = int(len(win_df))
                                anomaly_rate = (anomaly_count / total_count * 100) if total_count > 0 else 0.0
                            else:
                                anomaly_count, total_count, anomaly_rate = 0, 0, 0.0

                            # 평균 RUL: 현재시점 기준
                            avg_rul_now = float(snap_df["RUL"].mean()) if len(snap_df) > 0 else 0.0

                            # 카드 4개
                            c1, c2, c3, c4 = st.columns(4)
                            with c1:
                                st.markdown(
                                    f"""
                                    <div class="metric-card">
                                        <h3>🏭 총 장비</h3>
                                        <h1>{total_units}</h1>
                                        <p>대</p>
                                    </div>
                                    """, unsafe_allow_html=True
                                )
                            with c2:
                                st.markdown(
                                    f"""
                                    <div class="metric-card">
                                        <h3>⚠️ 이상 징후</h3>
                                        <h1>{anomaly_count}</h1>
                                        <p>건 ({anomaly_rate:.1f}%)</p>
                                    </div>
                                    """, unsafe_allow_html=True
                                )
                            with c3:
                                st.markdown(
                                    f"""
                                    <div class="metric-card">
                                        <h3>⏰ 평균 RUL</h3>
                                        <h1>{avg_rul_now:.0f}</h1>
                                        <p>사이클</p>
                                    </div>
                                    """, unsafe_allow_html=True
                                )
                            with c4:
                                st.markdown(
                                    f"""
                                    <div class="metric-card">
                                        <h3>🚨 위험 장비</h3>
                                        <h1>{critical_units_now}</h1>
                                        <p>대</p>
                                    </div>
                                    """, unsafe_allow_html=True
                                )

                            st.markdown("---")
                            st.success("✅ 데이터가 성공적으로 분석되었습니다!")

                            # === Hook up AI Analyzer Context ===
                            unit_status_now = (
                                win_df.assign(is_ano=(win_df.get("anomaly", 1) == -1).astype(int))
                                .groupby("unit", as_index=False)
                                .agg(RUL=("RUL", "mean"), time=("time", "count"), anomaly_count=("is_ano", "sum"))
                            )
                            if len(unit_status_now) > 0:
                                unit_status_now["anomaly_rate"] = np.where(
                                    unit_status_now["time"] > 0,
                                    unit_status_now["anomaly_count"] / unit_status_now["time"] * 100, 0
                                )
                            else:
                                unit_status_now = pd.DataFrame(columns=["unit", "RUL", "time", "anomaly_count", "anomaly_rate"])

                            # 컨텍스트 갱신
                            st.session_state.analyzer.update_context(
                                df.assign(_is_window=df["time"].isin(win_df["time"]) & df["unit"].isin(win_df["unit"])),
                                unit_status=unit_status_now
                            )
                            st.session_state.analysis_complete = True

                            # 디버그용(잠깐 확인해보고 필요없으면 지워도 됨)
                            st.caption(
                                f"snapshot%={snapshot_pct}, window={anom_window}, 임계치={critical_rul_thresh} | "
                                f"df_units={df['unit'].nunique()}, snap_units={snap_df['unit'].nunique()}, win_rows={len(win_df)}"
                            )

                            # === Hook up AI Analyzer Context ===
                            st.session_state.analyzer.update_context(
                                df.assign(_is_window=df["time"].isin(win_df["time"]) & df["unit"].isin(win_df["unit"])),
                                unit_status=unit_status_now
                            )
                            st.session_state.analysis_complete = True

                            # Basic data info (snapshot/window)
                            st.markdown("### 📋 데이터 정보")
                            i1, i2, i3 = st.columns(3)
                            with i1:
                                st.metric("총 레코드 수(윈도우)", len(win_df))
                            with i2:
                                st.metric("장비 수", total_units)
                            with i3:
                                st.metric("센서 수", len(selected_sensors))

                        except Exception as e:
                            st.error(f"이상 탐지 중 오류 발생: {str(e)}")
                            st.info("기본 통계 분석으로 진행합니다.")

                            st.markdown("## 📊 기본 데이터 분석")
                            c1, c2, c3, c4 = st.columns(4)
                            with c1:
                                st.metric("총 장비", f"{int(df['unit'].nunique())}대")
                            with c2:
                                st.metric("총 데이터", f"{len(df)}건")
                            with c3:
                                st.metric("평균 RUL", f"{float(df['RUL'].mean()):.1f}")
                            with c4:
                                st.metric("위험 장비", f"{int(df[df['RUL'] < 30]['unit'].nunique())}대")

                            st.session_state.analyzer.update_context(df, None)
                            st.session_state.analysis_complete = True
                    else:
                        st.error("❌ 분석 가능한 센서 데이터를 찾을 수 없습니다!")
                        st.info("데이터에 변동성이 있는 숫자형 센서 컬럼이 필요합니다.")
                        st.markdown("### 📋 데이터 구조")
                        st.write(f"데이터 크기: {df.shape}")
                        st.write(f"컬럼: {list(df.columns)}")
                        st.markdown("### 👀 샘플 데이터")
                        st.dataframe(df.head())
                except Exception as e:
                    st.error(f"RUL 계산 중 오류 발생: {str(e)}")
                    st.info("기본 데이터 표시로 진행합니다.")
                    st.markdown("### 📋 로드된 데이터")
                    st.write(f"데이터 크기: {df.shape}")
                    st.dataframe(df.head(10))
            else:
                st.error("❌ 'unit'과 'time' 컬럼이 필요합니다!")
                st.info("데이터의 첫 번째 컬럼은 'unit', 두 번째 컬럼은 'time'이어야 합니다.")
                st.markdown("### 📋 현재 데이터 구조")
                st.write(f"컬럼: {list(df.columns)}")
                st.dataframe(df.head())
        else:
            st.error("❌ 데이터를 로드할 수 없습니다.")
    except Exception as e:
        st.error(f"❌ 데이터 처리 중 오류 발생: {str(e)}")
        st.error("파일 형식과 내용을 확인해주세요.")
        with st.expander("🔧 디버깅 정보"):
            st.write(f"오류 유형: {type(e).__name__}")
            st.write(f"오류 메시지: {str(e)}")
            if hasattr(uploaded_file, 'name'):
                st.write(f"파일명: {uploaded_file.name}")
            if hasattr(uploaded_file, 'size'):
                st.write(f"파일 크기: {uploaded_file.size} bytes")
else:
    # Landing Section
    st.markdown(
        """
        <div style="text-align: center; padding: 3rem 0;">
            <h2>🚀 스마트팩토리 AI 분석 시스템</h2>
            <p style="font-size: 1.2rem; color: #666;">센서 데이터를 업로드하여 지능형 이상 탐지 및 AI 분석을 시작하세요</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            """
            ### ✨ 주요 기능
            
            🔍 **실시간 이상 탐지**  
            - Isolation Forest 알고리즘으로 이상 패턴 자동 감지
            - 장비별 실시간 상태 모니터링
            
            📊 **인터랙티브 시각화**  
            - Plotly 기반 동적 차트
            - 장비 상태 히트맵 및 트렌드 분석
            
            🤖 **AI 데이터 분석가**  
            - 자연어로 데이터 질의응답
            - 자동 인사이트 생성 및 권고사항 제시
            
            💡 **비즈니스 인사이트**  
            - 예방정비 우선순위 제시  
            - ROI 기반 의사결정 지원
            """
        )
        st.markdown("### 📋 지원 데이터 형식")
        st.code(
            """
            unit  time  os1  os2  os3  sensor_1  sensor_2  ...  sensor_21
            1     1     -0.1  0.2  0.5   518.67   641.82          2388.02
            1     2     -0.2  0.1  0.4   518.67   642.15          2388.07
            ...
            """
        )
        st.info("👈 왼쪽 사이드바에서 데이터 파일을 선택하거나 업로드하여 시작하세요!")
        st.markdown("---")
        st.markdown("### 💡 데이터 분석 팁")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                """
                **🚂 Train 데이터 특징:**
                - FD001: 100개 장비, 21개 센서
                - FD002: 260개 장비, 21개 센서  
                - FD003: 100개 장비, 21개 센서
                - FD004: 249개 장비, 21개 센서
                
                **📊 데이터 구조:**
                - unit: 장비 ID
                - time: 사이클 시간
                - os1~3: 운영 설정
                - sensor_1~21: 센서 값
                """
            )
        with c2:
            st.markdown(
                """
                **🔍 분석 순서:**
                1. 데이터 폴더에서 train 파일 선택
                2. 자동 컬럼 인식 및 RUL 계산
                3. 이상 탐지 알고리즘 실행
                4. 시각화 및 AI 분석
                
                **⚠️ 주의사항:**
                - 대용량 파일은 로딩 시간이 오래 걸릴 수 있음
                - 메모리 부족 시 파일 크기 확인 필요
                """
            )

# =============================
# AI Assistant Tabs
# =============================
st.markdown("---")
main_tabs = st.tabs(["💬 AI 대화", "📈 인사이트", "❓ 예상 질문"])

with main_tabs[0]:
    st.markdown("## 💬 AI 분석가와 대화하기")

    # Sample data loader for quick testing
    cent1, cent2, cent3 = st.columns([1, 2, 1])
    with cent2:
        if st.button("📁 샘플 데이터로 테스트", use_container_width=True):
            np.random.seed(42)
            units = list(range(1, 11))
            rows: List[Dict[str, Any]] = []
            for unit in units:
                cycles = np.random.randint(50, 200)
                for cycle in range(1, cycles + 1):
                    rul = cycles - cycle
                    sensors = {}
                    for i in range(1, 22):
                        base_value = np.random.normal(500, 50)
                        noise = np.random.normal(0, 10)
                        degradation = cycle * 0.1
                        sensors[f'sensor_{i}'] = base_value + noise + degradation
                    rows.append({'unit': unit, 'time': cycle, 'RUL': rul, **sensors})
            sdf = pd.DataFrame(rows)
            sdf['anomaly'] = np.random.choice([-1, 1], len(sdf), p=[0.1, 0.9])
            ust = sdf.groupby('unit').agg({'RUL': 'mean', 'time': 'count'}).reset_index()
            anomaly_counts = sdf[sdf['anomaly'] == -1].groupby('unit').size().reset_index(name='anomaly_count')
            ust = ust.merge(anomaly_counts, on='unit', how='left')
            ust['anomaly_count'] = ust['anomaly_count'].fillna(0)
            ust['anomaly_rate'] = (ust['anomaly_count'] / ust['time'] * 100)
            st.session_state.analyzer.update_context(sdf, ust)
            st.session_state.analysis_complete = True
            st.success("✅ 샘플 데이터가 로드되었습니다!")
            st.rerun()

    if st.session_state.analysis_complete:
        st.markdown("### 질문을 입력하세요:")
        chat_container = st.container()
        with chat_container:
            for chat in st.session_state.chat_history:
                if chat['type'] == 'user':
                    st.markdown(f"""
                    <div class="user-message">{chat['content']}</div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="ai-message">{chat['content']}</div>
                    """, unsafe_allow_html=True)

        user_question = st.text_input(
            "질문을 입력하세요:", placeholder="예: 가장 위험한 장비는 무엇인가요?", key="user_input"
        )
        col_a, col_b = st.columns([1, 4])
        with col_a:
            if st.button("💬 질문하기", use_container_width=True):
                if user_question:
                    st.session_state.chat_history.append({
                        'type': 'user', 'content': user_question, 'timestamp': datetime.now()
                    })
                    answer = st.session_state.analyzer.analyze_question(user_question)
                    if st.session_state.use_llm and api_key:
                        answer = llm_answer(user_question, st.session_state.analyzer.analysis_context, api_key)
                    st.session_state.chat_history.append({
                        'type': 'ai', 'content': answer, 'timestamp': datetime.now()
                    })
                    st.rerun()
        with col_b:
            if st.button("🗑️ 대화 기록 삭제", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
    else:
        st.info("📁 먼저 센서 데이터를 분석하세요. 위의 '샘플 데이터로 테스트' 버튼을 사용할 수도 있습니다.")

with main_tabs[1]:
    st.markdown("## 📈 AI 생성 인사이트")
    if st.session_state.analysis_complete:
        insights = st.session_state.analyzer.generate_insights()
        if insights:
            st.markdown("### 🔍 주요 발견사항")
            for i, insight in enumerate(insights):
                st.markdown(
                    f"""
                    <div class="insight-card">
                        <strong>인사이트 {i+1}</strong><br>{insight}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.success("🎉 현재 시스템 상태가 양호합니다!")

        st.markdown("### 📊 상세 분석 결과")
        ctx = st.session_state.analyzer.analysis_context
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                """
                <div class="analysis-summary">
                    <h4>🏭 장비 현황</h4>
                    <ul>
                        <li>총 장비 수: {total_units}대</li>
                        <li>전체 데이터: {total_records:,}건</li>
                        <li>센서 수: {sensor_count}개</li>
                    </ul>
                </div>
                """.format(
                    total_units=ctx.get('total_units', 0),
                    total_records=ctx.get('total_records', 0),
                    sensor_count=len(ctx.get('sensor_columns', [])),
                ),
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                """
                <div class="analysis-summary">
                    <h4>⚠️ 위험도 분석</h4>
                    <ul>
                        <li>평균 RUL: {avg_rul:.1f} 사이클</li>
                        <li>위험 장비: {critical_units}대</li>
                        <li>이상 징후율: {anomaly_rate:.2f}%</li>
                    </ul>
                </div>
                """.format(
                    avg_rul=ctx.get('avg_rul', 0),
                    critical_units=ctx.get('critical_units', 0),
                    anomaly_rate=ctx.get('anomaly_rate', 0),
                ),
                unsafe_allow_html=True,
            )
    else:
        st.info("📊 분석 데이터가 없습니다. 먼저 데이터를 로드하거나 샘플 데이터를 사용하세요.")

with main_tabs[2]:
    st.markdown("## ❓ 예상 질문 및 빠른 분석")
    if st.session_state.analysis_complete:
        questions = st.session_state.analyzer.generate_questions()
        st.markdown("### 🤔 자주 묻는 질문들")
        st.markdown("아래 질문들을 클릭하면 즉시 답변을 받을 수 있습니다:")
        for i, q in enumerate(questions):
            if st.button(f"❓ {q}", key=f"q_{i}", use_container_width=True):
                st.session_state.chat_history.append({'type': 'user', 'content': q, 'timestamp': datetime.now()})
                ans = st.session_state.analyzer.analyze_question(q)
                if st.session_state.use_llm and api_key:
                    ans = llm_answer(q, st.session_state.analyzer.analysis_context, api_key)
                st.session_state.chat_history.append({'type': 'ai', 'content': ans, 'timestamp': datetime.now()})
                st.rerun()
        st.markdown("---")
        st.markdown("### 💡 분석 팁")
        st.info(
            """
            **효과적인 질문 예시:**
            - "RUL이 10 미만인 장비들의 특징은?"
            - "센서 5번의 이상 패턴을 분석해주세요"  
            - "예방정비 비용 대비 효과는?"
            - "장비별 고장 위험도 순위는?"
            """
        )
    else:
        st.info("❓ 질문 목록을 생성하려면 먼저 데이터를 분석해주세요. 또는 샘플 데이터를 사용하세요.")
