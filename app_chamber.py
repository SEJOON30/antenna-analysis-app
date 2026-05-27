import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from fpdf import FPDF
import math

# ==========================================
# 1. 시스템 환경 및 공통 모듈
# ==========================================
st.set_page_config(layout="wide", page_title="Antenna Analysis Pro", page_icon="📡")

def apply_ppt_style(df):
    """표를 PPT 스타일로 깔끔하게 렌더링"""
    return df.style.set_table_styles([
        {'selector': 'th', 'props': [('background-color', '#2c3e50'), ('color', 'white'), ('font-weight', 'bold'), ('text-align', 'center')]},
        {'selector': 'td', 'props': [('text-align', 'center'), ('border', '1px solid #ddd')]}
    ])

# ==========================================
# 2. 데이터 처리 엔진 (Data Loader & Parser)
# ==========================================
def parse_s1p(file):
    """S1P 파일을 읽어 주파수, S11, VSWR, Real, Imag (스미스차트용) 반환"""
    try:
        content = file.read().decode("utf-8", errors='ignore')
        lines = [l for l in content.splitlines() if not l.strip().startswith('!') and not l.strip().startswith('#')]
        data = [l.split() for l in lines if len(l.split()) >= 3]
        
        df = pd.DataFrame(data).iloc[:, 0:3].astype(float)
        df.columns = ['freq', 'val1', 'val2']
        
        # S1P 포맷이 Mag/Phase 인지 Real/Imag 인지 추정하여 통일된 Gamma(반사계수) 도출
        if df['val1'].mean() < 0: # dB / Phase 포맷으로 간주
            df['s11_db'] = df['val1']
            mag = 10 ** (df['s11_db'] / 20)
            phase_rad = np.radians(df['val2'])
            df['real'] = mag * np.cos(phase_rad)
            df['imag'] = mag * np.sin(phase_rad)
        else: # Real / Imag 포맷
            df['real'] = df['val1']
            df['imag'] = df['val2']
            mag = np.sqrt(df['real']**2 + df['imag']**2)
            df['s11_db'] = 20 * np.log10(mag + 1e-10)
            
        gamma = 10 ** (df['s11_db'] / 20)
        df['vswr'] = (1 + gamma) / (1 - gamma + 1e-10)
        return df
    except Exception as e:
        return None

def find_resonance(df):
    """가장 매칭이 잘 된(S11이 가장 낮은) 공진 주파수 탐색"""
    if df is None or df.empty: return None, None
    min_idx = df['s11_db'].idxmin()
    return df.loc[min_idx, 'freq'], df.loc[min_idx, 's11_db']

# ==========================================
# 3. 리포트 생성 엔진 (출판사)
# ==========================================
def create_pdf(report_data, diagnosis_text):
    """분석 결과를 영문 기반 PDF로 출력 (한글 폰트 에러 방지)"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 18)
    pdf.cell(0, 15, txt="Antenna Performance Expert Report", ln=True, align='C')
    pdf.line(10, 25, 200, 25)
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt="1. Project Spec & Results", ln=True)
    pdf.set_font("Arial", size=11)
    for k, v in report_data.items():
        pdf.cell(0, 8, txt=f" - {k}: {v}", ln=True)
    
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt="2. Expert System Diagnosis", ln=True)
    pdf.set_font("Arial", size=11)
    for line in diagnosis_text:
        pdf.multi_cell(0, 8, txt=line)
    
    return pdf.output(dest='S').encode('latin-1', errors='ignore')

# ==========================================
# 4. 메인 어플리케이션 및 관제탑
# ==========================================
def main():
    # ------------------------------------------------
    # 사이드바: 🎯 Target 설정 및 📂 데이터 업로드
    # ------------------------------------------------
    st.sidebar.title("📡 Settings & Upload")
    
    st.sidebar.markdown("### 🎯 Target Spec")
    t_freq = st.sidebar.number_input("Target Freq (MHz)", value=2400.0, step=10.0)
    t_s11 = st.sidebar.number_input("Target S11 (dB)", value=-10.0, step=0.5)
    t_vswr = st.sidebar.number_input("Target VSWR", value=2.0, step=0.1)
    t_eff = st.sidebar.number_input("Target Efficiency (%)", value=40.0, step=1.0)
    t_gain = st.sidebar.number_input("Target Gain (dBi)", value=2.0, step=0.5)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📂 Current Data (V1.0)")
    s1p_curr = st.sidebar.file_uploader("S1P File (Network Analyzer)", type=["s1p"], key="c_s1p")
    cham_summ_curr = st.sidebar.file_uploader("Chamber Summary (Excel)", type=["xlsx", "csv"], key="c_sum")
    
    st.sidebar.markdown("---")
    with st.sidebar.expander("📂 Previous Data (비교 분석용)"):
        st.info("이전 데이터를 업로드하면 시스템이 자동으로 [비교 분석 모드]로 전환됩니다.")
        s1p_prev = st.file_uploader("Previous S1P File", type=["s1p"], key="p_s1p")
        cham_summ_prev = st.file_uploader("Previous Chamber Summary", type=["xlsx", "csv"], key="p_sum")

    menu = st.sidebar.radio("엔지니어링 Workflow", ["1. Project Home", "2. Data Viewer", "3. Deep Debugging", "4. Master Report"])

    # ------------------------------------------------
    # 메뉴 1: Project Home
    # ------------------------------------------------
    if menu == "1. Project Home":
        st.title("🏠 Project Home & Tuning History")
        st.markdown("현재 설정된 Target Spec이 전체 분석의 **PASS/FAIL**을 결정하는 기준선이 됩니다.")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Target Freq", f"{t_freq} MHz")
        c2.metric("Target S11 / VSWR", f"{t_s11} dB / {t_vswr}")
        c3.metric("Target Efficiency", f"{t_eff} %")
        c4.metric("Target Gain", f"{t_gain} dBi")
        
        st.markdown("### 📝 Tuning History (버전 관리)")
        st.text_area("현재 버전(V1.0) 튜닝 시 변경된 물리적 치수나 매칭 소자값을 기록하세요.", 
                     value="[예시]\n- 매칭 회로: 병렬 Inductor 2.2nH -> 1.8nH 교체\n- 기구물: 하우징 Clearance 0.5mm 추가 확보", height=150)
        
        st.info("💡 **가이드:** 왼쪽 사이드바에서 현재 데이터를 업로드하고 [2. Data Viewer]에서 스미스 차트와 계측 무결성을 확인하세요. 이후 [3. Deep Debugging]에서 인과관계를 심층 분석합니다.")

    # ------------------------------------------------
    # 메뉴 2: Data Viewer (현재 데이터 무결성 확인)
    # ------------------------------------------------
    elif menu == "2. Data Viewer":
        st.title("📊 계측 데이터 정밀 뷰어 (Current Data)")
        
        tab1, tab2 = st.tabs(["⚙️ Network Analyzer (S-Parameter)", "📡 Chamber (Radiation)"])
        
        with tab1:
            if s1p_curr:
                df_c = parse_s1p(s1p_curr)
                st.success(f"파일 로드 완료: {len(df_c)} points ({df_c['freq'].min()} ~ {df_c['freq'].max()} MHz)")
                
                # 정밀 스케일 제어
                c1, c2 = st.columns(2)
                s11_min = c1.number_input("S11 Min (dB)", value=-40.0)
                vswr_max = c2.number_input("VSWR Max", value=10.0)
                
                col1, col2, col3 = st.columns([1.5, 1.5, 1])
                # S11 그래프
                with col1:
                    fig1 = go.Figure()
                    fig1.add_trace(go.Scatter(x=df_c['freq'], y=df_c['s11_db'], name='S11', line=dict(color='#2980b9')))
                    fig1.add_hline(y=t_s11, line_dash="dash", line_color="red", annotation_text="Target S11")
                    fig1.update_layout(title="S11 Return Loss", yaxis_range=[s11_min, 5])
                    st.plotly_chart(fig1, use_container_width=True)
                
                # VSWR 그래프
                with col2:
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(x=df_c['freq'], y=df_c['vswr'], name='VSWR', line=dict(color='#d35400')))
                    fig2.add_hline(y=t_vswr, line_dash="dash", line_color="red", annotation_text="Target VSWR")
                    fig2.update_layout(title="VSWR", yaxis_range=[1.0, vswr_max])
                    st.plotly_chart(fig2, use_container_width=True)
                    
                # Smith Chart (스미스 차트 구현!)
                with col3:
                    fig3 = go.Figure()
                    fig3.add_trace(go.Scattersmith(imag=df_c['imag'], real=df_c['real'], name="Impedance", marker_color='#27ae60'))
                    fig3.update_layout(title="Smith Chart (Impedance)", smith=dict(realaxis_gridcolor='lightgray', imagaxis_gridcolor='lightgray'))
                    st.plotly_chart(fig3, use_container_width=True)
            else:
                st.warning("👈 S1P 데이터를 업로드해주세요.")
                
        with tab2:
            if cham_summ_curr:
                st.success("챔버 요약 데이터 로드 대기 중 (실제 엑셀 파싱 연동 시 활성화)")
                # 가상 챔버 데이터
                df_cham = pd.DataFrame({"Freq (MHz)": [2400, 2440, 2480], "Efficiency (%)": [42.1, 45.3, 41.8], "Peak Gain (dBi)": [1.2, 1.5, 1.1]})
                st.dataframe(apply_ppt_style(df_cham), use_container_width=True)
            else:
                st.warning("👈 챔버 데이터를 업로드해주세요.")

    # ------------------------------------------------
    # 메뉴 3: Deep Debugging (심층 진단 - 가장 상세한 핵심)
    # ------------------------------------------------
    elif menu == "3. Deep Debugging":
        st.title("🔍 전문가 심층 진단 (Expert Analysis)")
        
        # 스마트 모드 감지기
        is_comparison = (s1p_curr is not None) and (s1p_prev is not None)
        mode_text = "🔵 [비교 분석 모드] 이전 V0.9 데이터와의 Delta를 추적합니다." if is_comparison else "🟢 [단일 분석 모드] 현재 V1.0 데이터의 Target 달성 여부를 정밀 분석합니다."
        st.info(mode_text)
        
        if s1p_curr:
            df_c = parse_s1p(s1p_curr)
            df_p = parse_s1p(s1p_prev) if is_comparison else None
            
            c_res_freq, c_res_s11 = find_resonance(df_c)
            p_res_freq, p_res_s11 = find_resonance(df_p) if is_comparison else (None, None)
            
            st.markdown("---")
            d_tab1, d_tab2, d_tab3, d_tab4 = st.tabs(["1. 매칭 및 주파수 분석 (Impedance)", "2. 효율 및 이득 진단 (Radiation)", "3. 공간 방사 패턴", "4. 💡 종합 진단 (Expert Diagnosis)"])
            
            # [탭 1: 매칭 및 임피던스 분석]
            with d_tab1:
                st.subheader("공진 주파수(Resonance) 추적 및 매칭 평가")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Delta 지표 표시
                    if is_comparison:
                        shift = c_res_freq - p_res_freq
                        st.metric("현재 공진 주파수", f"{c_res_freq} MHz", f"{shift} MHz (Shift)", delta_color="inverse" if abs(c_res_freq - t_freq) > abs(p_res_freq - t_freq) else "normal")
                    else:
                        st.metric("현재 공진 주파수", f"{c_res_freq} MHz", f"Target({t_freq}MHz) 대비 {c_res_freq - t_freq} MHz 오차", delta_color="inverse")
                        
                    # 매칭 손실 (Mismatch Loss 계산)
                    gamma = 10**(c_res_s11/20)
                    mismatch_loss = (1 - gamma**2) * 100
                    st.metric("최소 매칭 손실 (Est.)", f"{mismatch_loss:.2f} %", "반사에 의해 손실되는 전력", delta_color="off")
                
                with col2:
                    # 중첩 그래프 (Overlay)
                    fig = go.Figure()
                    if is_comparison:
                        fig.add_trace(go.Scatter(x=df_p['freq'], y=df_p['s11_db'], name='Previous (V0.9)', line=dict(color='gray', dash='dot')))
                    fig.add_trace(go.Scatter(x=df_c['freq'], y=df_c['s11_db'], name='Current (V1.0)', line=dict(color='#2980b9', width=3)))
                    fig.add_hline(y=t_s11, line_dash="dash", line_color="red")
                    fig.update_layout(title="S11 Overlay Analysis", yaxis_range=[-40, 5])
                    st.plotly_chart(fig, use_container_width=True)

            # [탭 2, 3: 방사 품질 및 커버리지 (구조만 잡고 넘어감)]
            with d_tab2:
                st.subheader("방사 효율 및 Peak Gain 분석")
                st.info("챔버 데이터가 업로드되면 효율(Efficiency)과 이득(Gain)의 변화량을 중첩 그래프로 표시합니다.")
            with d_tab3:
                st.subheader("방사 패턴 왜곡 및 커버리지 진단")
                st.info("Null Density(사각지대) 및 SLL(부엽) 수치를 계산하여 기구 간섭 여부를 진단하는 영역입니다.")

            # [탭 4: 💡 인과관계 추적 (Expert Diagnosis)]
            with d_tab4:
                st.subheader("💡 Expert System 자동 진단 결과")
                
                # 복합 If-Then 인과관계 로직
                if is_comparison:
                    st.write("**[ 비교 추적 진단 (V0.9 -> V1.0) ]**")
                    if abs(shift) < 5:
                        st.success("✅ 공진 주파수 변화가 거의 없습니다. 매칭 회로 변경은 안정적으로 적용되었습니다.")
                    elif c_res_freq < p_res_freq:
                        st.warning(f"⚠️ 공진 주파수가 {abs(shift)}MHz 하향(Down) 이동했습니다. 안테나 물리적 길이가 길어졌거나 직렬 인덕턴스가 증가한 영향입니다.")
                    else:
                        st.warning(f"⚠️ 공진 주파수가 {abs(shift)}MHz 상향(Up) 이동했습니다. 방사체가 짧아졌거나 기구물 압착으로 갭(Gap)이 좁아졌을 수 있습니다.")
                else:
                    st.write("**[ 단일 품질 진단 (V1.0) ]**")
                    if c_res_s11 <= t_s11:
                        st.success(f"✅ 최고 공진점에서의 S11({c_res_s11:.1f}dB)이 Target({t_s11}dB)을 만족합니다. 매칭 상태 양호.")
                    else:
                        st.error(f"🚨 S11 매칭 불량({c_res_s11:.1f}dB). Target을 불만족합니다. L, C 소자값을 재조정하여 매칭을 깊게 파주세요.")
                        
                st.write("**[ 커버리지 및 패턴 추정 코멘트 ]**")
                st.info("S11 매칭이 우수함에도 불구하고 방사 효율이 낮다면, 매칭단 문제가 아닌 1) 하우징 간섭, 2) 그라운드(GND) 면적 부족, 3) 배터리/FPCB 등 주변 메탈 성분에 의한 패턴 찌그러짐을 의심해야 합니다.")

        else:
            st.warning("데이터를 분석하려면 사이드바에서 현재 데이터(S1P)를 업로드하세요.")

    # ------------------------------------------------
    # 메뉴 4: Master Report
    # ------------------------------------------------
    elif menu == "4. Master Report":
        st.title("📄 Master Report Export")
        st.write("분석된 최종 결과를 리포트로 출력하여 보고서에 바로 활용하세요.")
        
        report_data = {
            "Target Frequency": f"{t_freq} MHz",
            "Target S11": f"{t_s11} dB",
            "Target Efficiency": f"{t_eff} %"
        }
        diagnosis_text = [
            "Based on the S-parameter analysis:",
            "- Impedance matching condition has been evaluated.",
            "- Check the Deep Debugging tab for detailed frequency shifts.",
            "- Mechanical clearances should be verified if efficiency drops."
        ]
        
        st.markdown("### 📊 최종 진단 요약")
        st.table(pd.DataFrame(list(report_data.items()), columns=["Item", "Value"]))
        
        # PDF 다운로드
        pdf_bytes = create_pdf(report_data, diagnosis_text)
        st.download_button(
            label="📥 전문가 진단 리포트 (PDF) 다운로드",
            data=pdf_bytes,
            file_name="Antenna_Master_Report.pdf",
            mime="application/pdf",
            use_container_width=True
        )

if __name__ == "__main__":
    main()
