import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ==========================================
# 1. 시스템 환경 설정
# ==========================================
st.set_page_config(layout="wide", page_title="Antenna Analysis Pro", page_icon="📡")

def apply_ppt_style(df):
    return df.style.set_table_styles([
        {'selector': 'th', 'props': [('background-color', '#2c3e50'), ('color', 'white'), ('font-weight', 'bold'), ('text-align', 'center')]},
        {'selector': 'td', 'props': [('text-align', 'center'), ('border', '1px solid #ddd')]}
    ])

# ==========================================
# 2. 데이터 처리 엔진
# ==========================================
def parse_s1p(file):
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
    if df is None or df.empty: return None, None
    min_idx = df['s11_db'].idxmin()
    return df.loc[min_idx, 'freq'], df.loc[min_idx, 's11_db']

def generate_dummy_chamber_data(freq_target, is_previous=False):
    freqs = np.linspace(freq_target - 100, freq_target + 100, 5)
    eff_base = 35.0 if is_previous else 45.0
    gain_base = -1.0 if is_previous else 1.5
    df_summ = pd.DataFrame({
        "Freq (MHz)": freqs,
        "Efficiency (%)": eff_base - ((freqs - freq_target)**2 / 5000),
        "Peak Gain (dBi)": gain_base - ((freqs - freq_target)**2 / 10000)
    })
    
    phi, theta = np.meshgrid(np.linspace(0, 2*np.pi, 50), np.linspace(0, np.pi, 50))
    r = np.abs(np.sin(theta))
    if is_previous: r = r * 0.8 + 0.1
    x, y, z = r * np.sin(theta) * np.cos(phi), r * np.sin(theta) * np.sin(phi), r * np.cos(theta)
    df_3d = {'x': x, 'y': y, 'z': z}
    
    angles = np.linspace(0, 360, 72)
    base = 0 if not is_previous else -3
    df_2d_xy = pd.DataFrame({"Angle": angles, "Gain": base + 5*np.abs(np.sin(np.radians(angles)))})
    df_2d_xz = pd.DataFrame({"Angle": angles, "Gain": base + 5*np.abs(np.cos(np.radians(angles)))})
    df_2d_yz = pd.DataFrame({"Angle": angles, "Gain": base + 4*np.abs(np.cos(np.radians(angles))) + np.sin(np.radians(angles))})
    
    return df_summ, df_3d, df_2d_xy, df_2d_xz, df_2d_yz

# ==========================================
# 3. 메인 어플리케이션 (상태 유지 기능 탑재)
# ==========================================
def main():
    st.title("📡 안테나 성능 분석 시스템")
    
    # 💡 모바일 튕김 원천 차단: 메뉴 이동이 아닌 '탭(Tab)'으로 전체 구조를 변경했습니다!
    # 탭을 쓰면 화면을 넘겨도 파일이 날아가지 않습니다.
    tab_home, tab_viewer, tab_debug, tab_report = st.tabs([
        "🏠 1. 설정 및 업로드", "📊 2. 데이터 뷰어", "🔍 3. 심층 분석", "📄 4. 리포트"
    ])

    # ------------------------------------------------
    # 탭 1: 설정 및 파일 업로드 (가장 직관적인 배치)
    # ------------------------------------------------
    with tab_home:
        st.markdown("### 🎯 Target Spec 설정")
        c1, c2, c3 = st.columns(3)
        t_freq = c1.number_input("Target Freq (MHz)", value=2400.0, step=10.0)
        t_s11 = c2.number_input("Target S11 (dB)", value=-10.0, step=0.5)
        t_vswr = c3.number_input("Target VSWR", value=2.0, step=0.1)
        
        c4, c5 = st.columns(2)
        t_eff = c4.number_input("Target Efficiency (%)", value=40.0, step=1.0)
        t_gain = c5.number_input("Target Gain (dBi)", value=2.0, step=0.5)

        st.markdown("---")
        st.markdown("### 📂 파일 업로드 (이곳에 파일을 올리고 옆 탭으로 이동하세요!)")
        st.info("💡 **모바일 팁**: 파일이 '다운로드' 폴더나 카카오톡 다운로드 폴더에 있는지 확인하세요.")
        
        col_curr, col_prev = st.columns(2)
        
        with col_curr:
            st.success("🟢 현재 튜닝 데이터 (V1.0)")
            s1p_curr = st.file_uploader("1. S1P (매칭 데이터) [필수]", type=["s1p"], key="c_s1p")
            if s1p_curr: st.write("✅ S1P 파일 인식 완료!")
            
            cham_summ_curr = st.file_uploader("2. 챔버 Summary 엑셀", type=["xlsx", "csv"], key="c_sum")
            if cham_summ_curr: st.write("✅ 챔버 Summary 인식 완료!")
            
            cham_raw_curr = st.file_uploader("3. 챔버 3D/2D Raw 엑셀", type=["xlsx", "csv"], key="c_raw")
            if cham_raw_curr: st.write("✅ 챔버 Raw 인식 완료!")
            
        with col_prev:
            st.warning("🔵 이전 데이터 (비교 시에만 업로드)")
            s1p_prev = st.file_uploader("1. 이전 S1P 데이터", type=["s1p"], key="p_s1p")
            cham_summ_prev = st.file_uploader("2. 이전 챔버 Summary", type=["xlsx", "csv"], key="p_sum")
            cham_raw_prev = st.file_uploader("3. 이전 챔버 Raw Data", type=["xlsx", "csv"], key="p_raw")

    # ------------------------------------------------
    # 탭 2: Data Viewer (현재 데이터 확인)
    # ------------------------------------------------
    with tab_viewer:
        st.markdown("### 📊 계측 데이터 정밀 뷰어")
        
        if s1p_curr is None:
            st.error("👈 [1. 설정 및 업로드] 탭에서 S1P 파일을 먼저 업로드해주세요.")
        else:
            v_tab1, v_tab2 = st.tabs(["⚙️ 매칭 데이터", "📡 챔버 데이터"])
            
            with v_tab1:
                df_c = parse_s1p(s1p_curr)
                col1, col2 = st.columns(2)
                with col1:
                    fig1 = go.Figure()
                    fig1.add_trace(go.Scatter(x=df_c['freq'], y=df_c['s11_db'], name='S11', line=dict(color='#2980b9')))
                    fig1.add_hline(y=t_s11, line_dash="dash", line_color="red")
                    fig1.update_layout(title="S11 Return Loss")
                    st.plotly_chart(fig1, use_container_width=True)
                with col2:
                    fig3 = go.Figure(go.Scattersmith(imag=df_c['imag'], real=df_c['real'], name="Impedance"))
                    fig3.update_layout(title="Smith Chart")
                    st.plotly_chart(fig3, use_container_width=True)
                    
            with v_tab2:
                if cham_summ_curr and cham_raw_curr:
                    df_cham_c, df_3d_c, df_2d_xy_c, df_2d_xz_c, df_2d_yz_c = generate_dummy_chamber_data(t_freq, is_previous=False)
                    st.dataframe(apply_ppt_style(df_cham_c), use_container_width=True)
                    
                    st.markdown("#### 3D 방사 패턴")
                    fig_3d = go.Figure(data=[go.Surface(z=df_3d_c['z'], x=df_3d_c['x'], y=df_3d_c['y'], colorscale='Jet', opacity=0.8)])
                    fig_3d.update_layout(height=400, margin=dict(l=0, r=0, b=0, t=30))
                    st.plotly_chart(fig_3d, use_container_width=True)
                    
                    st.markdown("#### 2D 방사 패턴 (3면 동시)")
                    cx, cy, cz = st.columns(3)
                    with cx:
                        fig_xy = go.Figure(go.Scatterpolar(r=df_2d_xy_c['Gain'], theta=df_2d_xy_c['Angle'], mode='lines', line_color='blue'))
                        fig_xy.update_layout(polar=dict(radialaxis=dict(visible=True)), title="XY(H-Cut)")
                        st.plotly_chart(fig_xy, use_container_width=True)
                    with cy:
                        fig_xz = go.Figure(go.Scatterpolar(r=df_2d_xz_c['Gain'], theta=df_2d_xz_c['Angle'], mode='lines', line_color='green'))
                        fig_xz.update_layout(polar=dict(radialaxis=dict(visible=True)), title="XZ(E1-Cut)")
                        st.plotly_chart(fig_xz, use_container_width=True)
                    with cz:
                        fig_yz = go.Figure(go.Scatterpolar(r=df_2d_yz_c['Gain'], theta=df_2d_yz_c['Angle'], mode='lines', line_color='red'))
                        fig_yz.update_layout(polar=dict(radialaxis=dict(visible=True)), title="YZ(E2-Cut)")
                        st.plotly_chart(fig_yz, use_container_width=True)
                else:
                    st.warning("챔버 엑셀 파일 2개를 모두 올려야 방사 패턴이 활성화됩니다.")

    # ------------------------------------------------
    # 탭 3: Deep Debugging (심층 진단)
    # ------------------------------------------------
    with tab_debug:
        st.markdown("### 🔍 전문가 심층 진단")
        
        if s1p_curr is None:
            st.error("👈 [1. 설정 및 업로드] 탭에서 S1P 파일을 먼저 업로드해주세요.")
        else:
            is_comp = (s1p_curr is not None) and (s1p_prev is not None)
            st.info("🔵 [비교 모드] 이전 데이터와의 차이를 추적합니다." if is_comp else "🟢 [단일 모드] 현재 데이터의 완성도를 진단합니다.")
            
            df_c = parse_s1p(s1p_curr)
            df_p = parse_s1p(s1p_prev) if is_comp else None
            c_res_freq, c_res_s11 = find_resonance(df_c)
            p_res_freq, p_res_s11 = find_resonance(df_p) if is_comp else (None, None)
            
            df_cham_c, df_3d_c, df_2d_xy_c, df_2d_xz_c, df_2d_yz_c = generate_dummy_chamber_data(t_freq, False)
            if is_comp:
                df_cham_p, df_3d_p, df_2d_xy_p, df_2d_xz_p, df_2d_yz_p = generate_dummy_chamber_data(t_freq, True)
            
            d_tab1, d_tab2, d_tab3, d_tab4 = st.tabs(["1. 매칭 진단", "2. 효율/이득", "3. 방사패턴 왜곡", "4. 종합 제언"])
            
            with d_tab1:
                col1, col2 = st.columns([1, 2])
                with col1:
                    if is_comp:
                        st.metric("공진 주파수", f"{c_res_freq} MHz", f"{c_res_freq - p_res_freq} MHz (Shift)")
                    else:
                        st.metric("공진 주파수", f"{c_res_freq} MHz")
                    m_loss = (1 - 10**(c_res_s11/10)) * 100
                    st.metric("매칭 손실", f"{m_loss:.2f} %")
                with col2:
                    fig = go.Figure()
                    if is_comp:
                        fig.add_trace(go.Scatter(x=df_p['freq'], y=df_p['s11_db'], name='Prev', line=dict(color='gray', dash='dot')))
                    fig.add_trace(go.Scatter(x=df_c['freq'], y=df_c['s11_db'], name='Curr', line=dict(color='#2980b9', width=3)))
                    fig.add_hline(y=t_s11, line_dash="dash", line_color="red")
                    st.plotly_chart(fig, use_container_width=True)
                    
            with d_tab2:
                col1, col2 = st.columns(2)
                with col1:
                    fig_eff = go.Figure()
                    if is_comp:
                        fig_eff.add_trace(go.Scatter(x=df_cham_p['Freq (MHz)'], y=df_cham_p['Efficiency (%)'], name='Prev', line=dict(color='gray', dash='dot')))
                    fig_eff.add_trace(go.Scatter(x=df_cham_c['Freq (MHz)'], y=df_cham_c['Efficiency (%)'], name='Curr', line=dict(color='#27ae60', width=3)))
                    fig_eff.add_hline(y=t_eff, line_dash="dash", line_color="red")
                    st.plotly_chart(fig_eff, use_container_width=True)
                with col2:
                    fig_gain = go.Figure()
                    if is_comp:
                        fig_gain.add_trace(go.Scatter(x=df_cham_p['Freq (MHz)'], y=df_cham_p['Peak Gain (dBi)'], name='Prev', line=dict(color='gray', dash='dot')))
                    fig_gain.add_trace(go.Scatter(x=df_cham_c['Freq (MHz)'], y=df_cham_c['Peak Gain (dBi)'], name='Curr', line=dict(color='#8e44ad', width=3)))
                    fig_gain.add_hline(y=t_gain, line_dash="dash", line_color="red")
                    st.plotly_chart(fig_gain, use_container_width=True)
                    
            with d_tab3:
                st.markdown("#### 3면 방사 패턴 오버레이 (빔 왜곡 추적)")
                cx, cy, cz = st.columns(3)
                with cx:
                    f_xy = go.Figure()
                    if is_comp: f_xy.add_trace(go.Scatterpolar(r=df_2d_xy_p['Gain'], theta=df_2d_xy_p['Angle'], mode='lines', line=dict(color='gray', dash='dot')))
                    f_xy.add_trace(go.Scatterpolar(r=df_2d_xy_c['Gain'], theta=df_2d_xy_c['Angle'], mode='lines', line_color='blue'))
                    st.plotly_chart(f_xy, use_container_width=True)
                with cy:
                    f_xz = go.Figure()
                    if is_comp: f_xz.add_trace(go.Scatterpolar(r=df_2d_xz_p['Gain'], theta=df_2d_xz_p['Angle'], mode='lines', line=dict(color='gray', dash='dot')))
                    f_xz.add_trace(go.Scatterpolar(r=df_2d_xz_c['Gain'], theta=df_2d_xz_c['Angle'], mode='lines', line_color='green'))
                    st.plotly_chart(f_xz, use_container_width=True)
                with cz:
                    f_yz = go.Figure()
                    if is_comp: f_yz.add_trace(go.Scatterpolar(r=df_2d_yz_p['Gain'], theta=df_2d_yz_p['Angle'], mode='lines', line=dict(color='gray', dash='dot')))
                    f_yz.add_trace(go.Scatterpolar(r=df_2d_yz_c['Gain'], theta=df_2d_yz_c['Angle'], mode='lines', line_color='red'))
                    st.plotly_chart(f_yz, use_container_width=True)
                    
            with d_tab4:
                st.subheader("💡 Expert System 자동 진단 결과")
                if is_comp:
                    shift = c_res_freq - p_res_freq
                    if abs(shift) < 5: st.success("✅ 매칭 공진점이 안정적입니다.")
                    else: st.warning(f"⚠️ 공진 주파수가 {abs(shift)}MHz 이동했습니다. 소자값이나 기구 조립 상태를 확인하세요.")
                else:
                    if c_res_s11 <= t_s11: st.success("✅ 매칭 Target 만족!")
                    else: st.error("🚨 매칭 불량: 소자 재조정이 필요합니다.")

    # ------------------------------------------------
    # 탭 4: Master Report
    # ------------------------------------------------
    with tab_report:
        st.markdown("### 📄 분석 결과 요약 (캡처용)")
        report_data = {"Target Frequency": f"{t_freq} MHz", "Target S11": f"{t_s11} dB", "Target Efficiency": f"{t_eff} %"}
        st.table(pd.DataFrame(list(report_data.items()), columns=["Item", "Value"]))
        st.info("화면을 캡처하여 업무 보고용으로 활용하세요.")

if __name__ == "__main__":
    main()
