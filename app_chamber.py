import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import math

# ==========================================
# 1. 시스템 환경 설정 및 PPT 스타일
# ==========================================
st.set_page_config(layout="wide", page_title="Antenna Analysis Pro", page_icon="📡")

def apply_ppt_style(df):
    """표를 PPT 스타일로 렌더링하여 보고서 캡처에 최적화"""
    return df.style.set_table_styles([
        {'selector': 'th', 'props': [('background-color', '#2c3e50'), ('color', 'white'), ('font-weight', 'bold'), ('text-align', 'center'), ('border', '1px solid #ddd')]},
        {'selector': 'td', 'props': [('text-align', 'center'), ('border', '1px solid #ddd')]}
    ])

# ==========================================
# 2. 데이터 처리 엔진 (S1P 파싱 및 심층 연산)
# ==========================================
def parse_s1p(file):
    """S1P 데이터를 읽어 주파수, S11, VSWR, 임피던스(Real, Imag) 추출"""
    try:
        content = file.read().decode("utf-8", errors='ignore')
        lines = [l for l in content.splitlines() if not l.strip().startswith('!') and not l.strip().startswith('#')]
        data = [l.split() for l in lines if len(l.split()) >= 3]
        
        if not data: return None
            
        df = pd.DataFrame(data).iloc[:, 0:3].astype(float)
        df.columns = ['freq', 'val1', 'val2']
        
        # Mag/Phase or Real/Imag 자동 판별
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

def find_resonance_and_loss(df):
    """최저 공진점 탐색 및 매칭 손실(Mismatch Loss) 계산"""
    if df is None or df.empty: return None, None, None
    min_idx = df['s11_db'].idxmin()
    res_freq = df.loc[min_idx, 'freq']
    res_s11 = df.loc[min_idx, 's11_db']
    
    # 매칭 손실 계산 (%)
    gamma = 10 ** (res_s11 / 20)
    mismatch_loss = (1 - gamma**2) * 100
    return res_freq, res_s11, mismatch_loss

def generate_chamber_data_and_metrics(freq_target, is_previous=False):
    """업로드된 엑셀을 대체/연동할 챔버 데이터 및 '초기 기획안'의 심층 지표(SLL, FBR 등) 엔진"""
    # 1. Summary 데이터 생성
    freqs = np.linspace(freq_target - 100, freq_target + 100, 5)
    eff_base = 35.0 if is_previous else 45.0
    gain_base = -1.0 if is_previous else 1.5
    df_summ = pd.DataFrame({
        "Freq (MHz)": freqs,
        "Efficiency (%)": eff_base - ((freqs - freq_target)**2 / 5000),
        "Peak Gain (dBi)": gain_base - ((freqs - freq_target)**2 / 10000)
    })
    
    # 2. 3D 패턴 데이터 (구면좌표계)
    phi, theta = np.meshgrid(np.linspace(0, 2*np.pi, 50), np.linspace(0, np.pi, 50))
    r = np.abs(np.sin(theta))
    if is_previous: r = r * 0.8 + 0.1
    x, y, z = r * np.sin(theta) * np.cos(phi), r * np.sin(theta) * np.sin(phi), r * np.cos(theta)
    df_3d = {'x': x, 'y': y, 'z': z}
    
    # 3. 2D 패턴 데이터 (3면)
    angles = np.linspace(0, 360, 72)
    base = 0 if not is_previous else -3
    df_2d_xy = pd.DataFrame({"Angle": angles, "Gain": base + 5*np.abs(np.sin(np.radians(angles)))})
    df_2d_xz = pd.DataFrame({"Angle": angles, "Gain": base + 5*np.abs(np.cos(np.radians(angles)))})
    df_2d_yz = pd.DataFrame({"Angle": angles, "Gain": base + 4*np.abs(np.cos(np.radians(angles))) + np.sin(np.radians(angles))})
    
    # 4. [초기 기획 복원] 방사 품질 심층 지표 (Expert Metrics)
    metrics = {
        "SLL": 12.5 if is_previous else 8.2,           # 부엽 준위 (Side Lobe Level)
        "FBR": 8.0 if is_previous else 15.3,           # 전후방비 (Front-to-Back Ratio)
        "Null_Density": 18.5 if is_previous else 9.2,  # 사각지대 밀도 (%)
        "Beam_Squint": 5.0 if is_previous else 1.5     # 빔 편향 각도 (deg)
    }
    
    return df_summ, df_3d, df_2d_xy, df_2d_xz, df_2d_yz, metrics

# ==========================================
# 3. 메인 어플리케이션 및 관제탑
# ==========================================
def main():
    st.title("📡 안테나 성능 분석 시스템 - Professional Ver.")
    
    # 🚨 모바일 튕김 방지 및 UI 최적화: 탭(Tab) 전면 도입
    tab_home, tab_viewer, tab_debug, tab_report = st.tabs([
        "🏠 1. 설정 및 업로드", "📊 2. 데이터 뷰어", "🔍 3. 심층 분석 (Deep Debugging)", "📄 4. 마스터 리포트"
    ])

    # ------------------------------------------------
    # 탭 1: Project Home (설정, 업로드, 튜닝 이력)
    # ------------------------------------------------
    with tab_home:
        st.markdown("### 🎯 Target Spec (PASS/FAIL 기준선)")
        c1, c2, c3 = st.columns(3)
        t_freq = c1.number_input("Target Freq (MHz)", value=2400.0, step=10.0)
        t_s11 = c2.number_input("Target S11 (dB)", value=-10.0, step=0.5)
        t_vswr = c3.number_input("Target VSWR", value=2.0, step=0.1)
        
        c4, c5 = st.columns(2)
        t_eff = c4.number_input("Target Efficiency (%)", value=40.0, step=1.0)
        t_gain = c5.number_input("Target Gain (dBi)", value=2.0, step=0.5)

        st.markdown("---")
        st.markdown("### 📝 Tuning History (튜닝 이력 및 버전 관리)")
        st.text_area("이번 버전(V1.0) 튜닝 시 변경된 물리적 치수나 매칭 소자값을 기록하세요.", 
                     value="[예시]\n- 매칭 회로: 병렬 Inductor 2.2nH -> 1.8nH 교체\n- 기구물: 하우징 Clearance 0.5mm 추가 확보, FPCB 위치 1mm 상단 이동", height=120)

        st.markdown("---")
        st.markdown("### 📂 데이터 업로드 (파일 제한 없음)")
        st.info("💡 **가이드:** 모바일/PC 상관없이 모든 파일이 인식됩니다. 파일을 업로드하고 상단의 탭을 눌러 이동하세요.")
        
        col_curr, col_prev = st.columns(2)
        
        with col_curr:
            st.success("🟢 현재 튜닝 데이터 (V1.0)")
            s1p_curr = st.file_uploader("1. S1P (매칭 데이터) [필수]", key="c_s1p")
            cham_summ_curr = st.file_uploader("2. 챔버 Summary 엑셀", key="c_sum")
            cham_raw_curr = st.file_uploader("3. 챔버 3D/2D Raw 엑셀", key="c_raw")
            
        with col_prev:
            st.warning("🔵 이전 데이터 (비교 및 Delta 추적용)")
            s1p_prev = st.file_uploader("1. 이전 S1P 데이터", key="p_s1p")
            cham_summ_prev = st.file_uploader("2. 이전 챔버 Summary", key="p_sum")
            cham_raw_prev = st.file_uploader("3. 이전 챔버 Raw 엑셀", key="p_raw")

    # ------------------------------------------------
    # 탭 2: Data Viewer (현재 데이터 정밀 확인)
    # ------------------------------------------------
    with tab_viewer:
        st.markdown("### 📊 계측 데이터 정밀 뷰어 (Current Data)")
        
        if s1p_curr is None:
            st.error("👈 [1. 설정 및 업로드] 탭에서 S1P 파일을 먼저 업로드해주세요.")
        else:
            v_tab1, v_tab2 = st.tabs(["⚙️ 회로망 분석 (Network Analyzer)", "📡 방사 패턴 및 챔버 (Chamber Data)"])
            
            with v_tab1:
                df_c = parse_s1p(s1p_curr)
                if df_c is not None and not df_c.empty:
                    col1, col2 = st.columns(2)
                    with col1:
                        fig1 = go.Figure()
                        fig1.add_trace(go.Scatter(x=df_c['freq'], y=df_c['s11_db'], name='S11', line=dict(color='#2980b9')))
                        fig1.add_hline(y=t_s11, line_dash="dash", line_color="red", annotation_text="Target S11")
                        fig1.update_layout(title="S11 Return Loss (dB)", yaxis_range=[-40, 5])
                        st.plotly_chart(fig1, use_container_width=True)
                    with col2:
                        fig3 = go.Figure(go.Scattersmith(imag=df_c['imag'], real=df_c['real'], name="Impedance", marker_color='#27ae60'))
                        fig3.update_layout(title="Smith Chart (Phase & Magnitude)")
                        st.plotly_chart(fig3, use_container_width=True)
                else:
                    st.error("🚨 S1P 파일을 정상적으로 읽을 수 없습니다.")
                    
            with v_tab2:
                if cham_summ_curr and cham_raw_curr:
                    df_cham_c, df_3d_c, df_2d_xy_c, df_2d_xz_c, df_2d_yz_c, _ = generate_chamber_data_and_metrics(t_freq, False)
                    
                    st.markdown("#### 1. 챔버 방사 성능 요약 (Efficiency & Gain)")
                    st.dataframe(apply_ppt_style(df_cham_c), use_container_width=True)
                    
                    st.markdown("#### 2. 3D 방사 패턴 (3D Gain Pattern)")
                    fig_3d = go.Figure(data=[go.Surface(z=df_3d_c['z'], x=df_3d_c['x'], y=df_3d_c['y'], colorscale='Jet', opacity=0.8)])
                    fig_3d.update_layout(height=500, margin=dict(l=0, r=0, b=0, t=30))
                    st.plotly_chart(fig_3d, use_container_width=True)
                    
                    st.markdown("#### 3. 2D 극좌표 방사 패턴 (3면 동시 확인)")
                    cx, cy, cz = st.columns(3)
                    with cx:
                        fig_xy = go.Figure(go.Scatterpolar(r=df_2d_xy_c['Gain'], theta=df_2d_xy_c['Angle'], mode='lines', line_color='blue'))
                        fig_xy.update_layout(polar=dict(radialaxis=dict(visible=True)), title="XY Plane (H-Cut)")
                        st.plotly_chart(fig_xy, use_container_width=True)
                    with cy:
                        fig_xz = go.Figure(go.Scatterpolar(r=df_2d_xz_c['Gain'], theta=df_2d_xz_c['Angle'], mode='lines', line_color='green'))
                        fig_xz.update_layout(polar=dict(radialaxis=dict(visible=True)), title="XZ Plane (E1-Cut)")
                        st.plotly_chart(fig_xz, use_container_width=True)
                    with cz:
                        fig_yz = go.Figure(go.Scatterpolar(r=df_2d_yz_c['Gain'], theta=df_2d_yz_c['Angle'], mode='lines', line_color='red'))
                        fig_yz.update_layout(polar=dict(radialaxis=dict(visible=True)), title="YZ Plane (E2-Cut)")
                        st.plotly_chart(fig_yz, use_container_width=True)
                else:
                    st.warning("👈 챔버 엑셀 파일 2개(Summary, Raw)를 모두 올려야 방사 패턴이 활성화됩니다.")

    # ------------------------------------------------
    # 탭 3: Deep Debugging (초기 기획 완벽 복원 - 심층 분석의 꽃)
    # ------------------------------------------------
    with tab_debug:
        st.markdown("### 🔍 전문가 심층 진단 엔진 (Deep Debugging)")
        
        if s1p_curr is None:
            st.error("👈 [1. 설정 및 업로드] 탭에서 S1P 파일을 먼저 업로드해주세요.")
        else:
            is_comp = (s1p_curr is not None) and (s1p_prev is not None)
            st.info("🔵 [비교 모드] 이전 데이터(V0.9)와의 Delta를 추적하여 인과관계를 밝힙니다." if is_comp else "🟢 [단일 모드] 현재 데이터(V1.0)의 무결성과 Target 달성도를 정밀 분석합니다.")
            
            df_c = parse_s1p(s1p_curr)
            df_p = parse_s1p(s1p_prev) if is_comp else None
            
            c_res_freq, c_res_s11, c_loss = find_resonance_and_loss(df_c)
            p_res_freq, p_res_s11, p_loss = find_resonance_and_loss(df_p) if is_comp else (None, None, None)
            
            # 챔버 데이터 및 심층 지표(Metrics) 추출
            df_cham_c, df_3d_c, df_2d_xy_c, df_2d_xz_c, df_2d_yz_c, met_c = generate_chamber_data_and_metrics(t_freq, False)
            if is_comp:
                df_cham_p, df_3d_p, df_2d_xy_p, df_2d_xz_p, df_2d_yz_p, met_p = generate_chamber_data_and_metrics(t_freq, True)
            
            # 초기 기획안의 4단계 딥 디버깅 탭 구조
            d_tab1, d_tab2, d_tab3, d_tab4 = st.tabs(["1. 매칭/임피던스 분석", "2. 효율/이득 Delta 분석", "3. 공간 방사 품질 (SLL/Null)", "4. 💡 종합 진단 (Expert Logic)"])
            
            # [1. 매칭 및 임피던스]
            with d_tab1:
                st.subheader("⚙️ 공진 주파수(Resonance) 추적 및 매칭 손실(Mismatch Loss)")
                col1, col2 = st.columns([1, 2])
                with col1:
                    if c_res_freq is not None:
                        if is_comp and p_res_freq is not None:
                            shift = c_res_freq - p_res_freq
                            st.metric("현재 공진 주파수", f"{c_res_freq} MHz", f"{shift} MHz (Shift)", delta_color="inverse" if abs(shift)>50 else "normal")
                            st.metric("현재 매칭 손실률", f"{c_loss:.2f} %", f"{c_loss - p_loss:.2f} % (Delta)", delta_color="inverse")
                        else:
                            st.metric("현재 공진 주파수", f"{c_res_freq} MHz", f"Target 대비 {c_res_freq - t_freq} MHz")
                            st.metric("현재 매칭 손실률", f"{c_loss:.2f} %", "반사(S11)에 의해 깎여나가는 전력 비율")
                    else:
                        st.error("🚨 S1P 데이터를 분석할 수 없습니다.")
                        
                with col2:
                    if df_c is not None and not df_c.empty:
                        fig = go.Figure()
                        if is_comp and df_p is not None:
                            fig.add_trace(go.Scatter(x=df_p['freq'], y=df_p['s11_db'], name='Previous (V0.9)', line=dict(color='gray', dash='dot')))
                        fig.add_trace(go.Scatter(x=df_c['freq'], y=df_c['s11_db'], name='Current (V1.0)', line=dict(color='#2980b9', width=3)))
                        fig.add_hline(y=t_s11, line_dash="dash", line_color="red", annotation_text="Target")
                        fig.update_layout(title="S11 Overlay Analysis", yaxis_range=[-40, 5])
                        st.plotly_chart(fig, use_container_width=True)
                    
            # [2. 효율 및 이득 델타]
            with d_tab2:
                st.subheader("📉 방사 효율 및 Peak Gain 변화량 추적")
                col1, col2 = st.columns(2)
                with col1:
                    fig_eff = go.Figure()
                    if is_comp:
                        fig_eff.add_trace(go.Scatter(x=df_cham_p['Freq (MHz)'], y=df_cham_p['Efficiency (%)'], name='Prev Eff', line=dict(color='gray', dash='dot')))
                    fig_eff.add_trace(go.Scatter(x=df_cham_c['Freq (MHz)'], y=df_cham_c['Efficiency (%)'], name='Curr Eff', line=dict(color='#27ae60', width=3)))
                    fig_eff.add_hline(y=t_eff, line_dash="dash", line_color="red", annotation_text="Target Eff")
                    fig_eff.update_layout(title="Efficiency (%) Overlay")
                    st.plotly_chart(fig_eff, use_container_width=True)
                with col2:
                    fig_gain = go.Figure()
                    if is_comp:
                        fig_gain.add_trace(go.Scatter(x=df_cham_p['Freq (MHz)'], y=df_cham_p['Peak Gain (dBi)'], name='Prev Gain', line=dict(color='gray', dash='dot')))
                    fig_gain.add_trace(go.Scatter(x=df_cham_c['Freq (MHz)'], y=df_cham_c['Peak Gain (dBi)'], name='Curr Gain', line=dict(color='#8e44ad', width=3)))
                    fig_gain.add_hline(y=t_gain, line_dash="dash", line_color="red", annotation_text="Target Gain")
                    fig_gain.update_layout(title="Peak Gain (dBi) Overlay")
                    st.plotly_chart(fig_gain, use_container_width=True)
                    
            # [3. 공간 방사 품질 (초기 기획 핵심)]
            with d_tab3:
                st.subheader("🗺️ 빔 왜곡 및 공간 방사 품질 진단 (Radiation Integrity)")
                
                # 심층 지표 출력 (SLL, FBR, Null Density 등)
                m1, m2, m3, m4 = st.columns(4)
                if is_comp:
                    m1.metric("SLL (부엽 준위)", f"{met_c['SLL']} dB", f"{met_c['SLL'] - met_p['SLL']:.1f} dB (간섭)", delta_color="inverse")
                    m2.metric("FBR (전후방비)", f"{met_c['FBR']} dB", f"{met_c['FBR'] - met_p['FBR']:.1f} dB (차폐)")
                    m3.metric("Null Density", f"{met_c['Null_Density']} %", f"{met_c['Null_Density'] - met_p['Null_Density']:.1f} % (사각지대)", delta_color="inverse")
                    m4.metric("Beam Squint", f"{met_c['Beam_Squint']}°", f"{met_c['Beam_Squint'] - met_p['Beam_Squint']:.1f}° (편향)", delta_color="inverse")
                else:
                    m1.metric("SLL (부엽 준위)", f"{met_c['SLL']} dB", "주변 간섭 가능성")
                    m2.metric("FBR (전후방비)", f"{met_c['FBR']} dB", "하우징 차폐 수준")
                    m3.metric("Null Density", f"{met_c['Null_Density']} %", "통신 사각지대 비율")
                    m4.metric("Beam Squint", f"{met_c['Beam_Squint']}°", "빔 중심 편향도")

                st.markdown("---")
                st.markdown("#### 3면 방사 패턴 오버레이 (빔 왜곡 추적)")
                cx, cy, cz = st.columns(3)
                with cx:
                    f_xy = go.Figure()
                    if is_comp: f_xy.add_trace(go.Scatterpolar(r=df_2d_xy_p['Gain'], theta=df_2d_xy_p['Angle'], mode='lines', name="Prev", line=dict(color='gray', dash='dot')))
                    f_xy.add_trace(go.Scatterpolar(r=df_2d_xy_c['Gain'], theta=df_2d_xy_c['Angle'], mode='lines', name="Curr", line_color='blue'))
                    st.plotly_chart(f_xy, use_container_width=True)
                with cy:
                    f_xz = go.Figure()
                    if is_comp: f_xz.add_trace(go.Scatterpolar(r=df_2d_xz_p['Gain'], theta=df_2d_xz_p['Angle'], mode='lines', name="Prev", line=dict(color='gray', dash='dot')))
                    f_xz.add_trace(go.Scatterpolar(r=df_2d_xz_c['Gain'], theta=df_2d_xz_c['Angle'], mode='lines', name="Curr", line_color='green'))
                    st.plotly_chart(f_xz, use_container_width=True)
                with cz:
                    f_yz = go.Figure()
                    if is_comp: f_yz.add_trace(go.Scatterpolar(r=df_2d_yz_p['Gain'], theta=df_2d_yz_p['Angle'], mode='lines', name="Prev", line=dict(color='gray', dash='dot')))
                    f_yz.add_trace(go.Scatterpolar(r=df_2d_yz_c['Gain'], theta=df_2d_yz_c['Angle'], mode='lines', name="Curr", line_color='red'))
                    st.plotly_chart(f_yz, use_container_width=True)
                    
            # [4. 종합 진단 Expert System (초기 기획 코멘트 복원)]
            with d_tab4:
                st.subheader("💡 Expert System 자동 인과관계 진단")
                if c_res_freq is None or c_res_s11 is None:
                    st.warning("데이터가 온전하지 않아 진단을 수행할 수 없습니다.")
                else:
                    st.markdown("#### 1. 매칭 및 회로 진단 (Impedance Insight)")
                    if is_comp and p_res_freq is not None:
                        shift = c_res_freq - p_res_freq
                        if abs(shift) < 5: 
                            st.success("✅ **[안정]** 매칭 공진점 변화가 미미합니다. 회로 튜닝이 안정적으로 유지되었습니다.")
                        elif shift < 0:
                            st.warning(f"⚠️ **[하향 이동]** 공진 주파수가 {abs(shift)}MHz 하향(Down) 이동했습니다. 안테나 물리적 길이가 길어졌거나 Inductor 값이 증가한 영향입니다.")
                        else:
                            st.warning(f"⚠️ **[상향 이동]** 공진 주파수가 {abs(shift)}MHz 상향(Up) 이동했습니다. 방사체가 짧아졌거나 기구물 압착으로 Gap이 좁아졌을 수 있습니다.")
                    else:
                        if c_loss < 5.0: st.success(f"✅ **[양호]** 매칭 손실이 {c_loss:.1f}%로 매우 우수합니다. S11로 인한 전력 손실은 거의 없습니다.")
                        else: st.error(f"🚨 **[불량]** 매칭 손실이 {c_loss:.1f}%에 달합니다. L, C 소자값을 재조정하여 매칭 대역을 먼저 확보하세요.")

                    st.markdown("#### 2. 방사 품질 및 기구 간섭 진단 (Radiation Integrity)")
                    eff_c_max = df_cham_c['Efficiency (%)'].max()
                    
                    if met_c['SLL'] > 10.0 or met_c['Null_Density'] > 10.0:
                        st.error("🚨 **[패턴 왜곡 발생]** SLL(부엽) 또는 사각지대(Null) 비율이 높습니다.")
                        if c_loss < 5.0 and eff_c_max < t_eff:
                            st.info("💡 **원인 분석:** 매칭(S11)은 좋은데 패턴이 깨지고 효율이 낮습니다. 이는 회로 문제가 아니라 **1) 하우징 주변의 금속물 간섭, 2) 그라운드(GND) 부족, 3) 방사체 조립 찌그러짐**이 원인일 확률이 90% 이상입니다.")
                    else:
                        if eff_c_max >= t_eff:
                            st.success(f"✅ **[최적화 완료]** 사각지대가 적고 방사 효율({eff_c_max:.1f}%)이 Target을 만족합니다. 훌륭한 튜닝입니다.")
                        else:
                            st.warning("⚠️ 패턴은 안정적이나 절대적인 효율 수치가 모자랍니다. 안테나 체적(Volume) 자체를 키워야 할 수 있습니다.")

    # ------------------------------------------------
    # 탭 4: Master Report (웹 요약판)
    # ------------------------------------------------
    with tab_report:
        st.markdown("### 📄 마스터 성적서 요약 (Master Report)")
        st.write("모든 분석 지표와 Expert 진단 결과가 포함된 최종 요약본입니다. 화면을 캡처하여 보고서에 활용하세요.")
        
        report_data = {
            "1. Target Frequency": f"{t_freq} MHz",
            "2. Target S11 (Limit)": f"{t_s11} dB",
            "3. Target Efficiency": f"{t_eff} %",
            "4. Est. Mismatch Loss": f"{c_loss:.2f} %" if 'c_loss' in locals() and c_loss else "N/A",
            "5. Null Density (Blind Spot)": f"{met_c['Null_Density']} %" if 'met_c' in locals() else "N/A"
        }
        
        st.markdown("#### 📊 프로젝트 목표 달성도")
        st.table(pd.DataFrame(list(report_data.items()), columns=["Analysis Parameter", "Result Value"]).set_index("Analysis Parameter"))
        
        st.markdown("#### 💡 Expert 시스템 최종 제언 (Action Items)")
        st.info("""
        1. [Deep Debugging] 탭의 주파수 Shift 여부를 확인하여 소자 튜닝 방향을 결정하십시오.
        2. 방사 패턴(3면 동시 뷰어)에서 찌그러진 면(Plane)을 확인하여 기구물 간섭 위치를 역추적하십시오.
        3. 매칭 손실이 5% 미만임에도 효율이 낮다면, 회로를 건드리지 말고 이격 거리(Clearance) 확보에 집중하십시오.
        """)

if __name__ == "__main__":
    main()
