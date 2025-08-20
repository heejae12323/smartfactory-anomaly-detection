from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import glob

# 페이지 설정
st.set_page_config(
    page_title="🏭 Smart Factory AI Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 사이드바 스타일링
st.markdown("""
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
    
    .chat-container {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
    }
    
    .chat-message {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .insight-card {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.2rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# 환경 변수 로드
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# 메인 헤더
st.markdown('<h1 class="main-header">🏭 Smart Factory AI Dashboard</h1>', unsafe_allow_html=True)
st.markdown("---")

# 사이드바 설정
with st.sidebar:
    st.markdown("### 🎛️ 제어판")
    
    # API 상태 표시
    if api_key:
        st.markdown(f'<div class="status-success">✅ AI 시스템 연결됨</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="status-warning">⚠️ API Key 설정 필요</div>', unsafe_allow_html=True)
    
    st.markdown("### 📁 데이터 선택")
    
    # 데이터 소스 선택
    data_source = st.radio(
        "데이터 소스를 선택하세요:",
        ["📂 데이터 폴더에서 선택", "📤 파일 업로드"],
        help="기존 데이터 폴더의 파일을 선택하거나 새로운 파일을 업로드할 수 있습니다"
    )
    
    uploaded_file = None
    
    # 사이드바 – 데이터 소스 선택에서 폴더 안의 파일 선택 처리
    if data_source == "📂 데이터 폴더에서 선택":
        import glob
        import os

        train_files = glob.glob("data/train_*.txt")
        train_files = [os.path.basename(f) for f in train_files]
        
        if train_files:
            # 파일 정보 표시
            st.markdown("#### 📊 사용 가능한 데이터 파일")
            for file_name in train_files:
                file_path = os.path.join("data", file_name)
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    
                    # 파일 크기에 따른 색상 구분
                    if file_size > 5 * 1024 * 1024:  # 5MB 이상
                        size_color = "🔴"
                    elif file_size > 2 * 1024 * 1024:  # 2MB 이상
                        size_color = "🟡"
                    else:
                        size_color = "🟢"
                    
                    st.markdown(f"{size_color} **{file_name}** ({file_size/1024:.1f}KB)")
                else:
                    st.markdown(f"❌ **{file_name}** (파일 없음)")
            
            st.markdown("---")
            
            selected_file = st.selectbox(
                "🎯 분석할 Train 파일 선택:",
                train_files,
                help="분석할 train 데이터 파일을 선택하세요"
            )
            
            if selected_file:
                file_path = os.path.join("data", selected_file)
                file_size = os.path.getsize(file_path)

                # uploaded_file 대신 로컬 경로를 나타내는 객체로 처리 (read/seek 메서드를 정의하지 않음)
                uploaded_file = type(
                    "UploadedFile",
                    (),
                    {"name": selected_file, "size": file_size}
                )()   

        else:
            st.warning("📁 data 폴더에 train_*.txt 파일을 찾을 수 없습니다.")
        
        # 전체 데이터 폴더 정보 표시
        st.markdown("---")
        st.markdown("#### 📁 전체 데이터 폴더 정보")
        
        all_files = glob.glob("data/*.txt")
        if all_files:
            file_categories = {
                "🚂 Train 데이터": [f for f in all_files if "train" in str(f).lower()],
                "🧪 Test 데이터": [f for f in all_files if "test" in str(f).lower()],
                "⏰ RUL 데이터": [f for f in all_files if "rul" in str(f).lower()],
                "📄 기타": [f for f in all_files if not any(x in str(f).lower() for x in ["train", "test", "rul"])]
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
    
    else:  # 파일 업로드
        uploaded_file = st.file_uploader(
            "센서 데이터 파일을 선택하세요",
            type=["txt", "csv"],
            help="CSV 또는 TXT 형식의 센서 데이터를 업로드하세요"
        )
        
        if uploaded_file:
            st.success(f"📄 {uploaded_file.name}")
            st.info(f"크기: {uploaded_file.size/1024:.1f}KB")

# 메인 컨텐츠
if uploaded_file is not None:
    try:
        # CSS 스타일 추가
        st.markdown("""
        <style>
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin: 0.5rem 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .metric-card h3 {
            margin: 0;
            font-size: 1rem;
            opacity: 0.9;
        }
        .metric-card h1 {
            margin: 0.5rem 0;
            font-size: 2.5rem;
            font-weight: bold;
        }
        .metric-card p {
            margin: 0;
            font-size: 0.9rem;
            opacity: 0.8;
        }
        </style>
        """, unsafe_allow_html=True)
        
        if uploaded_file is not None:
            try:
                with st.spinner("🔄 데이터를 분석하고 있습니다..."):
                    # 분기: 업로드된 파일(st.file_uploader) vs 폴더에서 선택한 파일
                    if (
                        hasattr(uploaded_file, "read")
                        and callable(getattr(uploaded_file, "read", None))
                        and hasattr(uploaded_file, "seek")
                    ):
                        # st.file_uploader로 업로드된 파일 처리
                        uploaded_file.seek(0)
                        # 첫 줄 파악 및 구분자 감지 (생략)
                        # …
                        # 데이터프레임 읽기
                        df = pd.read_csv(
                            uploaded_file,
                            sep=separator,
                            header=None,
                            engine="python",
                            encoding="utf-8",
                        )
                    else:
                        # 폴더에서 선택한 로컬 파일 처리
                        file_path = os.path.join("data", uploaded_file.name)
                        # 첫 줄 파악 및 구분자 감지
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                            first_line = f.readline().strip()
                        if "\t" in first_line:
                            separator = "\t"
                        elif "," in first_line:
                            separator = ","
                        elif " " in first_line and len(first_line.split()) > 1:
                            separator = r"\s+"
                        else:
                            separator = ","
                        # 데이터프레임 읽기
                        df = pd.read_csv(
                            file_path,
                            sep=separator,
                            header=None,
                            engine="python",
                            encoding="utf-8",
                        )

            except Exception as e:
                st.error(f"❌ 데이터 로드 중 오류: {str(e)}")
                st.stop()
        
        # 데이터가 성공적으로 로드된 경우에만 계속 진행
        if df is not None and not df.empty:
            # 데이터 타입 변환 및 정리
            try:
                # 컬럼명 설정
                expected_cols = 26
                if df.shape[1] == expected_cols:
                    df.columns = ["unit", "time", "os1", "os2", "os3"] + [f"sensor_{i}" for i in range(1, 22)]
                else:
                    df.columns = [f"col_{i}" for i in range(df.shape[1])]
                    if df.shape[1] >= 3:
                        df.columns[0] = "unit"
                        df.columns[1] = "time"
                
                # 숫자형 변환 시도
                for col in df.columns:
                    if col in ['unit', 'time'] or 'sensor' in col or 'os' in col:
                        try:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        except:
                            pass
                
                # NaN 값 처리
                df = df.dropna(subset=['unit', 'time'])
                
                # 데이터 타입 확인
                if df.empty:
                    st.error("데이터 처리 후 유효한 행이 없습니다.")
                    st.stop()
                
            except Exception as e:
                st.error(f"데이터 전처리 중 오류: {str(e)}")
                st.stop()
            
            # RUL 계산
            if "unit" in df.columns and "time" in df.columns:
                try:
                    # unit과 time이 숫자형인지 확인
                    if not pd.api.types.is_numeric_dtype(df['unit']) or not pd.api.types.is_numeric_dtype(df['time']):
                        st.error("unit과 time 컬럼이 숫자 형식이어야 합니다.")
                        st.stop()
                    
                    max_cycle = df.groupby("unit")["time"].max().reset_index()
                    max_cycle.columns = ["unit", "max_time"]
                    df = df.merge(max_cycle, on="unit", how="left")
                    df["RUL"] = df["max_time"] - df["time"]
                    df.drop(columns=["max_time"], inplace=True)
                    
                    # 센서 컬럼 찾기
                    sensor_cols = [col for col in df.columns if "sensor" in col or col.startswith("col_")]
                    sensor_cols = [col for col in sensor_cols if col not in ['unit', 'time', 'RUL']]
                    
                    # 변동성 있는 센서 선택
                    selected_sensors = []
                    for col in sensor_cols:
                        try:
                            if pd.api.types.is_numeric_dtype(df[col]):
                                std_val = float(df[col].std())
                                if not pd.isna(std_val) and std_val > 0.01:
                                    selected_sensors.append(col)
                        except:
                            continue
                    
                    if len(selected_sensors) > 0:
                        # 이상 탐지
                        try:
                            from sklearn.preprocessing import StandardScaler
                            from sklearn.ensemble import IsolationForest
                            import numpy as np
                            
                            # 결측값 처리
                            sensor_data = df[selected_sensors].fillna(df[selected_sensors].mean())
                            
                            scaler = StandardScaler()
                            df_scaled = scaler.fit_transform(sensor_data)
                            iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
                            df["anomaly"] = iso.fit_predict(df_scaled)
                            
                            # unit_status를 먼저 계산하여 전역적으로 사용 가능하게 함
                            unit_status = df.groupby('unit').agg({
                                'RUL': 'mean',
                                'time': 'count'
                            }).reset_index()
                            
                            # 이상치 수를 별도로 계산
                            anomaly_counts = df[df['anomaly'] == -1].groupby('unit').size().reset_index(name='anomaly_count')
                            unit_status = unit_status.merge(anomaly_counts, on='unit', how='left')
                            unit_status['anomaly_count'] = unit_status['anomaly_count'].fillna(0)
                            
                            # 안전한 나눗셈
                            unit_status['anomaly_rate'] = np.where(
                                unit_status['time'] > 0,
                                unit_status['anomaly_count'] / unit_status['time'] * 100,
                                0
                            )
                            unit_status = unit_status.sort_values('anomaly_rate', ascending=False)
                            
                            # 📊 대시보드 메트릭스
                            st.markdown("## 📊 실시간 모니터링 대시보드")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                total_units = int(df['unit'].nunique())
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h3>🏭 총 장비</h3>
                                    <h1>{total_units}</h1>
                                    <p>대</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col2:
                                anomaly_count = int((df["anomaly"] == -1).sum())
                                total_count = len(df)
                                anomaly_rate = (anomaly_count / total_count * 100) if total_count > 0 else 0
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h3>⚠️ 이상 징후</h3>
                                    <h1>{anomaly_count}</h1>
                                    <p>건 ({anomaly_rate:.1f}%)</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col3:
                                avg_rul = float(df['RUL'].mean())
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h3>⏰ 평균 RUL</h3>
                                    <h1>{avg_rul:.0f}</h1>
                                    <p>사이클</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col4:
                                critical_units = int(df[df['RUL'] < 30]['unit'].nunique())
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h3>🚨 위험 장비</h3>
                                    <h1>{critical_units}</h1>
                                    <p>대</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            st.markdown("---")
                            
                            # 성공 메시지
                            st.success("✅ 데이터가 성공적으로 분석되었습니다!")
                            
                            # 기본 데이터 정보 표시
                            st.markdown("### 📋 데이터 정보")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("총 레코드 수", len(df))
                            with col2:
                                st.metric("장비 수", df['unit'].nunique())
                            with col3:
                                st.metric("센서 수", len(selected_sensors))
                        
                        except Exception as e:
                            st.error(f"이상 탐지 중 오류 발생: {str(e)}")
                            st.info("기본 통계 분석으로 진행합니다.")
                            
                            # 기본 분석 (이상 탐지 없이)
                            st.markdown("## 📊 기본 데이터 분석")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                total_units = int(df['unit'].nunique())
                                st.metric("총 장비", f"{total_units}대")
                            
                            with col2:
                                total_records = len(df)
                                st.metric("총 데이터", f"{total_records}건")
                            
                            with col3:
                                avg_rul = float(df['RUL'].mean())
                                st.metric("평균 RUL", f"{avg_rul:.1f}")
                            
                            with col4:
                                critical_units = int(df[df['RUL'] < 30]['unit'].nunique())
                                st.metric("위험 장비", f"{critical_units}대")
                    
                    else:
                        st.error("❌ 분석 가능한 센서 데이터를 찾을 수 없습니다!")
                        st.info("데이터에 변동성이 있는 숫자형 센서 컬럼이 필요합니다.")
                        
                        # 데이터 구조 표시
                        st.markdown("### 📋 데이터 구조")
                        st.write(f"데이터 크기: {df.shape}")
                        st.write(f"컬럼: {list(df.columns)}")
                        
                        # 샘플 데이터 표시
                        st.markdown("### 👀 샘플 데이터")
                        st.dataframe(df.head())
                
                except Exception as e:
                    st.error(f"RUL 계산 중 오류 발생: {str(e)}")
                    st.info("기본 데이터 표시로 진행합니다.")
                    
                    # 기본 데이터 표시
                    st.markdown("### 📋 로드된 데이터")
                    st.write(f"데이터 크기: {df.shape}")
                    st.dataframe(df.head(10))
            
            else:
                st.error("❌ 'unit'과 'time' 컬럼이 필요합니다!")
                st.info("데이터의 첫 번째 컬럼은 'unit', 두 번째 컬럼은 'time'이어야 합니다.")
                
                # 현재 컬럼 구조 표시
                st.markdown("### 📋 현재 데이터 구조")
                st.write(f"컬럼: {list(df.columns)}")
                st.dataframe(df.head())
        
        else:
            st.error("❌ 데이터를 로드할 수 없습니다.")
            
    except Exception as e:
        st.error(f"❌ 데이터 처리 중 오류 발생: {str(e)}")
        st.error("파일 형식과 내용을 확인해주세요.")
        
        # 디버깅 정보 표시
        with st.expander("🔧 디버깅 정보"):
            st.write(f"오류 유형: {type(e).__name__}")
            st.write(f"오류 메시지: {str(e)}")
            if hasattr(uploaded_file, 'name'):
                st.write(f"파일명: {uploaded_file.name}")
            if hasattr(uploaded_file, 'size'):
                st.write(f"파일 크기: {uploaded_file.size} bytes")

else:
    # 랜딩 페이지
    st.markdown("""
    <div style="text-align: center; padding: 3rem 0;">
        <h2>🚀 스마트팩토리 AI 분석 시스템</h2>
        <p style="font-size: 1.2rem; color: #666;">센서 데이터를 업로드하여 지능형 이상 탐지 및 AI 분석을 시작하세요</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
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
        """)
        
        st.markdown("### 📋 지원 데이터 형식")
        st.code("""
        unit  time  os1  os2  os3  sensor_1  sensor_2  ...  sensor_21
        1     1     -0.1  0.2  0.5   518.67   641.82          2388.02
        1     2     -0.2  0.1  0.4   518.67   642.15          2388.07
        ...
        """)
        
        st.info("👈 왼쪽 사이드바에서 데이터 파일을 선택하거나 업로드하여 시작하세요!")
        
        # 데이터 분석 팁 추가
        st.markdown("---")
        st.markdown("### 💡 데이터 분석 팁")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
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
            """)
        
        with col2:
            st.markdown("""
            **🔍 분석 순서:**
            1. 데이터 폴더에서 train 파일 선택
            2. 자동 컬럼 인식 및 RUL 계산
            3. 이상 탐지 알고리즘 실행
            4. 시각화 및 AI 분석
            
            **⚠️ 주의사항:**
            - 대용량 파일은 로딩 시간이 오래 걸릴 수 있음
            - 메모리 부족 시 파일 크기 확인 필요
            """)