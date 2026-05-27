import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ==========================================
# 1. 시스템 환경 설정
# ==========================================
st.set_page_config(layout="wide", page_title="Antenna Analysis Pro", page_icon="📡")

def apply_ppt_style(df):
    """표를 PPT 스타일로 렌더링"""
    return df.style.set_table_styles([
        {'selector': 'th', 'props': [('background-color', '#2c3e50'), ('color', 'white'), ('font-weight', 'bold'), ('text-align', 'center')]},
        {'selector': 'td', 'props': [('text-align', 'center'), ('border', '1px solid #ddd')]}
    ])

# ==========================================
# 2. 데이터 처리 엔진 (S1P 및 챔버)
# ==========================================
def parse_s1p(file):
    """S1P 파싱 엔진"""
    try:
        content = file.read().decode("utf-8", errors='ignore')
        lines = [l for l in content.splitlines() if not l.strip().startswith('!') and not l.strip().startswith('#')]
        data = [l.split() for l in lines if len(l.split()) >= 3]
        
        df = pd.DataFrame(data).iloc[:, 0:3].astype(float)
        df.columns = ['freq', 'val1', 'val2']
        
        if df['val1'].mean() < 0:
            df['s11_db'] = df['val1']
            mag = 10 ** (df['s11_db'] / 20)
            phase_rad = np.radians(df['val2'])
            df['real'] = mag * np.cos(phase_rad)
            df['imag'] = mag * np.sin(phase_rad)
        else:
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
    """최저 S11 공진점 찾기"""
    if df is None or df.empty: return None, None
    min_idx = df['s11_db'].idxmin()
    return df.loc[min_idx, 'freq'], df.loc[min_idx, 's11_db']

def generate_dummy_chamber_data(freq_target, is_previous=False):
    """업로드된 엑셀 파일을 대신할 시뮬레이션 데이터 엔진 (3D, 2D 3컷 포함)"""
    # 1. Summary 데이터
    freqs = np.linspace(freq_target - 100, freq_target + 100, 5)
    eff_base = 35.0 if is_previous else 45.0
    gain_base = -1.0 if is_previous else 1.5
    df_summ = pd.DataFrame({
        "Freq (MHz)": freqs,
        "Efficiency (%)": eff_base - ((freqs - freq_target)**2 / 5000),
        "Peak Gain (dBi)": gain_base - ((freqs - freq_target)**2 / 10000)
    })
    
    # 2. 3D 패턴 데이터 (구면 좌표계를 직교 좌표계로 변환)
    phi = np.linspace(0, 2*np.pi, 50)
    theta = np.linspace(0, np.pi, 50)
    phi, theta = np.meshgrid(phi, theta)
    r = np.abs(np.sin(theta)) # 도넛 형태의 다이폴 방사 패턴 시뮬레이션
    if is_previous: r = r * 0.8 + 0.1 # 이전 버전은 패턴이 다름
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    df_3d = {'x': x, 'y': y, 'z': z}
    
    # 3. 2D 패턴 데이터 (XY, XZ, YZ 3면)
    angles = np.linspace(0, 360, 72)
    base = 0 if not is_previous else -3
    df_2d_xy = pd.DataFrame({"Angle": angles, "Gain": base + 5*np.abs(np.sin(np.radians(angles)))})
    df_2d_xz = pd.DataFrame({"Angle": angles, "Gain": base + 5*np.abs(np.cos(np.radians(angles)))})
    df_2d_yz = pd.DataFrame({"Angle": angles, "Gain": base + 4*np.abs(np.cos(np.radians(angles))) + np.sin(np.radians(angles))})
    
    return df_summ, df_3d, df_2d_xy, df_2d_xz, df_2d_yz

# ==========================================
# 3. 메인 어플리케이션 및 관제탑
# ==========================================
def main():
    # ------------------------------------------------
    # 사이드바: 🎯 Target 설정 및 📂 데이터 업로드 (에러 원천 차단)
    # ------------------------------------------------
    st.sidebar.title("📡 System Control")
    
    st.sidebar.markdown("### 🎯 Target Spec")
    t_freq = st.sidebar.number_input("Target Freq (MHz)", value=2400.0, step=10.0)
    t_s11 = st.sidebar.number_input("Target S11 (dB)", value=-10.0, step=0.5)
    t_vswr = st.sidebar.number_input("Target VSWR", value=2.0, step=0.1)
    t_eff = st.sidebar.number_input("Target Efficiency (%)", value=40.0, step=1.0)
    t_gain = st.sidebar.number_input("Target Gain (dBi)", value=2.0, step=0.5)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📂 Current Data (V1.0)")
    # [수정됨] 엑셀 파일을 2개(요약/Raw)로 확실하게 분리했습니다!
    s1p_curr = st.sidebar.file_uploader("1. S1P (매칭 데이터)", type=["s1p"], key="c_s1p")
    cham_summ_curr = st.sidebar.file_uploader("2. 챔버 Summary (효율/이득 엑셀)", type=["xlsx", "csv"], key="c_sum")
    cham_raw_curr = st.sidebar.file_uploader("3. 챔버 Raw Data (3D/2D 엑셀)", type=["xlsx", "csv"], key="c_raw")
    
    st.sidebar.markdown("---")
    # [수정됨] 에러를 일으키던 expander 로직을 깔끔하게 정리했습니다.
    st.sidebar.markdown("### 📂 Previous Data (비교용)")
    st.sidebar.info("비교 분석을 하실 때만 업로드하세요.")
    s1p_prev = st.sidebar.file_uploader("1. 이전 S1P 데이터", type=["s1p"], key="p_s1p")
    cham_summ_prev = st.sidebar.file_uploader("2. 이전 챔버 Summary", type=["xlsx", "csv"], key="p_sum")
    cham_raw_prev = st.sidebar.file_uploader("3. 이전 챔버 Raw Data", type=["xlsx", "csv"], key="p_raw")

    menu = st.sidebar.radio("엔지니어링 Workflow", ["1. Project Home", "2. Data Viewer", "3. Deep Debugging", "4. Master Report"])

    # ------------------------------------------------
    # 메뉴 1: Project Home
    # ------------------------------------------------
    if menu == "1. Project Home":
        st.title("🏠 Project Home & Tuning History")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Target Freq", f"{t_freq} MHz")
        c2.metric("Target S11 / VSWR", f"{t_s11} dB / {t_vswr}")
        c3.metric("Target Efficiency", f"{t_eff} %")
        c4.metric("Target Gain", f"{t_gain} dBi")
        
        st.markdown("### 📝 Tuning History (버전 관리)")
        st.text_area("튜닝 작업 기록", value="[V1.0 변경점]\n- 방사체 길이 1.5mm 연장\n- 매칭 회로: C소자 1.2pF 적용", height=150)

    # ------------------------------------------------
    # 메뉴 2: Data Viewer (현재 데이터 무결성 확인)
    # ------------------------------------------------
    elif menu == "2. Data Viewer":
        st.title("📊 계측 데이터 정밀 뷰어 (Current Data)")
        
        tab1, tab2 = st.tabs(["⚙️ Network Analyzer", "📡 Chamber (Radiation)"])
        
        with tab1:
            if s1p_curr:
                df_c = parse_s1p(s1p_curr)
                c1, c2 = st.columns(2)
                s11_min = c1.number_input("S11 Min (dB)", value=-40.0)
                vswr_max = c2.number_input("VSWR Max", value=10.0)
                
                col1, col2, col3 = st.columns([1.5, 1.5, 1])
                with col1:
                    fig1 = go.Figure()
                    fig1.add_trace(go.Scatter(x=df_c['freq'], y=df_c['s11_db'], name='S11', line=dict(color='#2980b9')))
                    fig1.add_hline(y=t_s11, line_dash="dash", line_color="red")
                    fig1.update_layout(title="S11 Return Loss", yaxis_range=[s11_min, 5])
                    st.plotly_chart(fig1, use_container_width=True)
                with col2:
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(x=df_c['freq'], y=df_c['vswr'], name='VSWR', line=dict(color='#d35400')))
                    fig2.add_hline(y=t_vswr, line_dash="dash", line_color="red")
                    fig2.update_layout(title="VSWR", yaxis_range=[1.0, vswr_max])
                    st.plotly_chart(fig2, use_container_width=True)
                with col3:
                    fig3 = go.Figure(go.Scattersmith(imag=df_c['imag'], real=df_c['real'], name="Impedance"))
                    fig3.update_layout(title="Smith Chart")
                    st.plotly_chart(fig3, use_container_width=True)
            else:
                st.warning("👈 좌측 메뉴에서 S1P 데이터를 업로드해주세요.")
                
        # [수정됨] 3D 및 2D(3개 동시) 패턴 구현 로직
        with tab2:
            if cham_summ_curr and cham_raw_curr:
                st.success("챔버 요약(Summary) 및 패턴(Raw) 데이터 분석 완료!")
                df_cham_c, df_3d_c, df_2d_xy_c, df_2d_xz_c, df_2d_yz_c = generate_dummy_chamber_data(t_freq, is_previous=False)
                
                st.markdown("#### 1. 챔버 방사 성능 요약표 (Efficiency & Gain)")
                st.dataframe(apply_ppt_style(df_cham_c), use_container_width=True)
                
                st.markdown("#### 2. 3D 방사 패턴 (3D Radiation Pattern)")
                # Plotly 3D Surface 렌더링
                fig_3d = go.Figure(data=[go.Surface(z=df_3d_c['z'], x=df_3d_c['x'], y=df_3d_c['y'], colorscale='Jet', opacity=0.8)])
                fig_3d.update_layout(title="3D Gain Pattern", autosize=False, height=500, margin=dict(l=0, r=0, b=0, t=30))
                st.plotly_chart(fig_3d, use_container_width=True)
                
                st.markdown("#### 3. 2D 방사 패턴 (XY, XZ, YZ 3면 동시 확인)")
                # 3개의 극좌표 그래프를 나란히 배치
                col_x, col_y, col_z = st.columns(3)
                with col_x:
                    fig_xy = go.Figure(go.Scatterpolar(r=df_2d_xy_c['Gain'], theta=df_2d_xy_c['Angle'], mode='lines', line_color='blue'))
                    fig_xy.update_layout(polar=dict(radialaxis=dict(visible=True)), title="XY Plane (H-Cut)")
                    st.plotly_chart(fig_xy, use_container_width=True)
                with col_y:
                    fig_xz = go.Figure(go.Scatterpolar(r=df_2d_xz_c['Gain'], theta=df_2d_xz_c['Angle'], mode='lines', line_color='green'))
                    fig_xz.update_layout(polar=dict(radialaxis=dict(visible=True)), title="XZ Plane (E1-Cut)")
                    st.plotly_chart(fig_xz, use_container_width=True)
                with col_z:
                    fig_yz = go.Figure(go.Scatterpolar(r=df_2d_yz_c['Gain'], theta=df_2d_yz_c['Angle'], mode='lines', line_color='red'))
                    fig_yz.update_layout(polar=dict(radialaxis=dict(visible=True)), title="YZ Plane (E2-Cut)")
                    st.plotly_chart(fig_yz, use_container_width=True)
                    
            else:
                st.warning("👈 좌측에서 챔버 엑셀 2개(Summary, Raw Data)를 모두 업로드하셔야 3D/2D 패턴이 활성화됩니다.")

    # ------------------------------------------------
    # 메뉴 3: Deep Debugging (심층 진단)
    # ------------------------------------------------
    elif menu == "3. Deep Debugging":
        st.title("🔍 전문가 심층 진단 (Expert Analysis)")
        
        is_comparison = (s1p_curr is not None) and (s1p_prev is not None)
        mode_text = "🔵 [비교 분석 모드] 이전 데이터와의 Delta를 추적합니다." if is_comparison else "🟢 [단일 분석 모드] 현재 데이터의 Target 달성 여부를 분석합니다."
        st.info(mode_text)
        
        if s1p_curr:
            df_c = parse_s1p(s1p_curr)
            df_p = parse_s1p(s1p_prev) if is_comparison else None
            c_res_freq, c_res_s11 = find_resonance(df_c)
            p_res_freq, p_res_s11 = find_resonance(df_p) if is_comparison else (None, None)
            
            df_cham_c, df_3d_c, df_2d_xy_c, df_2d_xz_c, df_2d_yz_c = generate_dummy_chamber_data(t_freq, is_previous=False)
            if is_comparison:
                df_cham_p, df_3d_p, df_2d_xy_p, df_2d_xz_p, df_2d_yz_p = generate_dummy_chamber_data(t_freq, is_previous=True)
            
            d_tab1, d_tab2, d_tab3 = st.tabs(["1. 매칭 진단", "2. 효율/이득 Delta 분석", "3. 방사 패턴 왜곡(3면) 추적"])
            
            # [탭 1: 매칭 및 임피던스 분석]
            with d_tab1:
                st.subheader("공진 주파수(Resonance) 추적 및 매칭 평가")
                col1, col2 = st.columns([1, 2])
                with col1:
                    if is_comparison:
                        shift = c_res_freq - p_res_freq
                        st.metric("현재 공진 주파수", f"{c_res_freq} MHz", f"{shift} MHz (Shift)")
                    else:
                        st.metric("현재 공진 주파수", f"{c_res_freq} MHz", f"Target 대비 {c_res_freq - t_freq} MHz 오차")
                        
                    mismatch_loss = (1 - 10**(c_res_s11/10)) * 100
                    st.metric("최소 매칭 손실 (Est.)", f"{mismatch_loss:.2f} %", "반사에 의해 깎이는 효율")
                
                with col2:
                    fig = go.Figure()
                    if is_comparison:
                        fig.add_trace(go.Scatter(x=df_p['freq'], y=df_p['s11_db'], name='Previous (V0.9)', line=dict(color='gray', dash='dot')))
                    fig.add_trace(go.Scatter(x=df_c['freq'], y=df_c['s11_db'], name='Current (V1.0)', line=dict(color='#2980b9', width=3)))
                    fig.add_hline(y=t_s11, line_dash="dash", line_color="red")
                    fig.update_layout(title="S11 Overlay Analysis", yaxis_range=[-40, 5])
                    st.plotly_chart(fig, use_container_width=True)

            # [탭 2: 효율/이득 분석]
            with d_tab2:
                st.subheader("방사 효율 및 Peak Gain Delta 분석")
                col1, col2 = st.columns(2)
                with col1:
                    fig_eff = go.Figure()
                    if is_comparison:
                        fig_eff.add_trace(go.Scatter(x=df_cham_p['Freq (MHz)'], y=df_cham_p['Efficiency (%)'], name='Prev Eff', line=dict(color='gray', dash='dot')))
                    fig_eff.add_trace(go.Scatter(x=df_cham_c['Freq (MHz)'], y=df_cham_c['Efficiency (%)'], name='Curr Eff', line=dict(color='#27ae60', width=3)))
                    fig_eff.add_hline(y=t_eff, line_dash="dash", line_color="red")
                    fig_eff.update_layout(title="Efficiency (%) Overlay")
                    st.plotly_chart(fig_eff, use_container_width=True)
                with col2:
                    fig_gain = go.Figure()
                    if is_comparison:
                        fig_gain.add_trace(go.Scatter(x=df_cham_p['Freq (MHz)'], y=df_cham_p['Peak Gain (dBi)'], name='Prev Gain', line=dict(color='gray', dash='dot')))
                    fig_gain.add_trace(go.Scatter(x=df_cham_c['Freq (MHz)'], y=df_cham_c['Peak Gain (dBi)'], name='Curr Gain', line=dict(color='#8e44ad', width=3)))
                    fig_gain.add_hline(y=t_gain, line_dash="dash", line_color="red")
                    fig_gain.update_layout(title="Peak Gain (dBi) Overlay")
                    st.plotly_chart(fig_gain, use_container_width=True)

            # [탭 3: 3면 방사 패턴 오버레이]
            with d_tab3:
                st.subheader("방사 패턴 왜곡 추적 (Pattern Overlay)")
                st.markdown("이전 버전 대비 **빔이 얼마나 펴졌는지(또는 찌그러졌는지)** 3개 면에서 동시 진단합니다.")
                
                col_x, col_y, col_z = st.columns(3)
                with col_x:
                    fig_xy_o = go.Figure()
                    if is_comparison:
                        fig_xy_o.add_trace(go.Scatterpolar(r=df_2d_xy_p['Gain'], theta=df_2d_xy_p['Angle'], mode='lines', name='Prev', line=dict(color='gray', dash='dot')))
                    fig_xy_o.add_trace(go.Scatterpolar(r=df_2d_xy_c['Gain'], theta=df_2d_xy_c['Angle'], mode='lines', name='Curr', line=dict(color='blue', width=2)))
                    fig_xy_o.update_layout(polar=dict(radialaxis=dict(visible=True)), title="XY Plane (H-Cut)")
                    st.plotly_chart(fig_xy_o, use_container_width=True)
                with col_y:
                    fig_xz_o = go.Figure()
                    if is_comparison:
                        fig_xz_o.add_trace(go.Scatterpolar(r=df_2d_xz_p['Gain'], theta=df_2d_xz_p['Angle'], mode='lines', name='Prev', line=dict(color='gray', dash='dot')))
                    fig_xz_o.add_trace(go.Scatterpolar(r=df_2d_xz_c['Gain'], theta=df_2d_xz_c['Angle'], mode='lines', name='Curr', line=dict(color='green', width=2)))
                    fig_xz_o.update_layout(polar=dict(radialaxis=dict(visible=True)), title="XZ Plane (E1-Cut)")
                    st.plotly_chart(fig_xz_o, use_container_width=True)
                with col_z:
                    fig_yz_o = go.Figure()
                    if is_comparison:
                        fig_yz_o.add_trace(go.Scatterpolar(r=df_2d_yz_p['Gain'], theta=df_2d_yz_p['Angle'], mode='lines', name='Prev', line=dict(color='gray', dash='dot')))
                    fig_yz_o.add_trace(go.Scatterpolar(r=df_2d_yz_c['Gain'], theta=df_2d_yz_c['Angle'], mode='lines', name='Curr', line=dict(color='red', width=2)))
                    fig_yz_o.update_layout(polar=dict(radialaxis=dict(visible=True)), title="YZ Plane (E2-Cut)")
                    st.plotly_chart(fig_yz_o, use_container_width=True)
                    
        else:
            st.warning("데이터를 분석하려면 사이드바에서 현재 데이터(S1P)를 업로드하세요.")

    # ------------------------------------------------
    # 메뉴 4: Master Report (웹 요약판)
    # ------------------------------------------------
    elif menu == "4. Master Report":
        st.title("📄 Master Report")
        
        report_data = {
            "1. Target Frequency": f"{t_freq} MHz",
            "2. Target S11 (Limit)": f"{t_s11} dB",
            "3. Target Efficiency": f"{t_eff} %"
        }
        st.markdown("### 📊 프로젝트 목표치 요약")
        st.table(pd.DataFrame(list(report_data.items()), columns=["Item", "Value"]))
        
        st.markdown("### 💡 진단 소견 (Conclusion)")
        st.info("""
        1. 매칭 탭의 주파수 Shift 여부를 확인하여 소자 튜닝 방향을 결정하십시오.
        2. 방사 패턴(3면 동시 뷰어)에서 찌그러진 면(Plane)을 확인하여 기구물 간섭 위치를 역추적하십시오.
        """)

if __name__ == "__main__":
    main()
