import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re

# 1. 스트림릿 페이지 설정
st.set_page_config(layout="wide")

# 2. 대표 타이틀
st.title("📡 안테나 성능 분석 시스템")
st.markdown("---")

# ---------------------------------------------------------
# 🛠️ [사이드바 영역] 프로젝트 타겟 스펙 및 파일 업로드
# ---------------------------------------------------------
st.sidebar.subheader("🔌 [1] S-Parameter 스펙 설정")
target_s11 = st.sidebar.number_input("목표 최대 S11 (dB)", value=-10.0, step=1.0)
target_vswr = st.sidebar.number_input("목표 최대 VSWR", value=2.0, step=0.1)

st.sidebar.markdown("---")
st.sidebar.subheader("📡 [2] 방사성능 스펙 설정")
target_eff = st.sidebar.number_input("목표 최소 효율 (%)", value=40.0, step=1.0)
target_avg_gain = st.sidebar.number_input("목표 최소 평균 이득 (dBi)", value=-4.0, step=0.1)

st.sidebar.markdown("---")
st.sidebar.header("📂 계측 데이터 파일 업로드")
st.sidebar.caption("💡 폰에서 파일이 회색으로 잠기는 현상을 방지하기 위해 확장자 제한을 해제했습니다. 자유롭게 터치하세요.")
# 모바일 차단 버그 해결을 위해 type 제한 완전 제거
s1p_file = st.sidebar.file_uploader("🔌 0. S-Parameter 데이터 (.s1p)")
summary_file = st.sidebar.file_uploader("📊 1. 방사 효율 및 이득 요약 (.csv/xlsx)")
raw_file = st.sidebar.file_uploader("📂 2. 각도별 Raw Data (.csv/xlsx)")

st.sidebar.markdown("---")
st.sidebar.header("🗺️ 대시보드 분석 메뉴")
menu_selection = st.sidebar.radio(
    "이동할 분석 단계를 선택하세요",
    [
        "🔍 임피던스 및 방사 인과관계 진단",
        "Passive 성능 분석", 
        "교차 편파 분석", 
        "방사패턴 분석(성능)", 
        "방사패턴 분석(RAW)", 
        "📋 엔지니어 종합 진단 리포트"
    ]
)

st.write(f"현재 뷰어 모드: **{menu_selection}**")
st.markdown("---")


# ---------------------------------------------------------
# 🔍 [메뉴 0순위] 임피던스 및 방사 인과관계 진단
# ---------------------------------------------------------
if menu_selection == "🔍 임피던스 및 방사 인과관계 진단":
    st.subheader("🔍 임피던스 매칭(S11/VSWR)과 방사 성능(효율)의 상호 인과관계 정밀 추적")
    st.write("안테나의 성능 저하 원인이 에너지를 받지 못하는 '회로 매칭 불량'인지, 에너지는 받았으나 뿜지 못하는 '기구물 차폐'인지 감별합니다.")
    
    if s1p_file is not None and summary_file is not None:
        try:
            # S1P 파일 파싱 엔진
            s1p_lines = s1p_file.read().decode("utf-8").splitlines()
            freq_list_s1p, s11_list_s1p, vswr_list_s1p = [], [], []
            
            unit_multiplier = 1.0 
            for line in s1p_lines:
                clean_line = line.strip().lower()
                if clean_line.startswith('#'):
                    if "mhz" in clean_line: unit_multiplier = 1.0
                    elif "ghz" in clean_line: unit_multiplier = 1000.0
                    elif "khz" in clean_line: unit_multiplier = 0.001
                    elif "hz" in clean_line: unit_multiplier = 0.000001
                    break
            
            for line in s1p_lines:
                clean_line = line.strip()
                if not clean_line or clean_line.startswith('!') or clean_line.startswith('#'): continue
                tokens = clean_line.split()
                if len(tokens) >= 3:
                    try:
                        f_val = float(tokens[0]) * unit_multiplier
                        val1 = float(tokens[1])
                        val2 = float(tokens[2])
                        s11_db = val1 if val1 <= 0 else 20 * np.log10(max(1e-5, np.sqrt(val1**2 + val2**2)))
                        if s11_db > 0: s11_db = -s11_db
                        gamma = 10**(s11_db / 20.0)
                        vswr_val = (1.0 + gamma) / (1.0 - gamma) if gamma < 0.999 else 99.0
                        
                        freq_list_s1p.append(f_val)
                        s11_list_s1p.append(s11_db)
                        vswr_list_s1p.append(vswr_val)
                    except: pass
                    
            df_s1p = pd.DataFrame({"freq": freq_list_s1p, "s11": s11_list_s1p, "vswr": vswr_list_s1p}).sort_values("freq").reset_index(drop=True)

            # 챔버 요약 데이터 로드 엔진
            if summary_file.name.lower().endswith('.csv'): df_sum_raw = pd.read_csv(summary_file, header=None)
            else: df_sum_raw = pd.read_excel(summary_file, header=None, engine='openpyxl')
            header_row_idx = None
            for idx in range(len(df_sum_raw)):
                row_vals = [str(v).strip().lower() for v in df_sum_raw.iloc[idx].values]
                if "no." in row_vals and "freq.[mhz]" in row_vals:
                    header_row_idx = idx
                    break
            df_clean_data = df_sum_raw.iloc[(header_row_idx+1):].dropna(subset=[0]).reset_index(drop=True)
            cham_freqs = pd.to_numeric(df_clean_data[1], errors='coerce').astype(float).values
            cham_effs = pd.to_numeric(df_clean_data[15], errors='coerce').astype(float).values
            cham_avg_gains = pd.to_numeric(df_clean_data[16], errors='coerce').astype(float).values
            
            valid_m = ~np.isnan(cham_freqs)
            cham_freqs, cham_effs, cham_avg_gains = cham_freqs[valid_m], cham_effs[valid_m], cham_avg_gains[valid_m]
            df_cham = pd.DataFrame({"freq": cham_freqs, "eff": cham_effs, "avg_gain": cham_avg_gains}).sort_values("freq").reset_index(drop=True)

            # 차트 축 제어판
            st.markdown("#### 🎛️ 차트 디스플레이 격자 범위 및 단위(Step) 설정")
            col_ctrl1, col_ctrl2 = st.columns(2)
            with col_ctrl1: 
                scale_s11 = st.slider("S11 차트 Y축 범위 설정 (dB)", -60.0, 10.0, (-45.0, 10.0), step=5.0)
                step_s11 = st.slider("S11 격자 주 단위 선택 (주 격리 폭)", 1.0, 10.0, 2.0, step=1.0)
            with col_ctrl2: 
                scale_vswr = st.slider("VSWR 차트 Y축 범위 설정", 1.0, 25.0, (1.0, 11.0), step=1.0)
                step_vswr = st.slider("VSWR 격자 주 단위 선택", 0.5, 5.0, 1.0, step=0.5)

            # 직관적인 통합 체크박스 스위치 배치
            show_markers_toggle = st.checkbox("🚩 마커 보기", value=False)
            st.markdown("---")
            
            # 마커 리스트 전광판용 텍스트 빌딩 및 반올림 연산
            all_s11_markers_y, all_vswr_markers_y = [], []
            marker_symbols_labels = [] 
            
            s11_embed_text = "<b>[Tr1] S11 Marker List</b><br>"
            vswr_embed_text = "<b>[Tr2] VSWR Marker List</b><br>"
            board_rows = [] 
            
            for m_idx, cf in enumerate(df_cham["freq"].values):
                idx_s1p = (df_s1p["freq"] - cf).abs().argmin()
                r_s11 = round(float(df_s1p.iloc[idx_s1p]["s11"]), 2)
                r_vswr = round(float(df_s1p.iloc[idx_s1p]["vswr"]), 2)
                
                all_s11_markers_y.append(r_s11)
                all_vswr_markers_y.append(r_vswr)
                
                m_label = f"m{m_idx+1}"
                marker_symbols_labels.append(m_label)
                
                s11_embed_text += f"{m_label}  {cf:.1f} MHz  {r_s11:.2f} dB<br>"
                vswr_embed_text += f"{m_label}  {cf:.1f} MHz  {r_vswr:.2f}<br>"
                
                board_rows.append({
                    "마커": m_label, "주파수": f"{cf:.0f} MHz",
                    "S11 반사손실": f"{r_s11:.2f} dB", "VSWR 전압비": f"{r_vswr:.2f}"
                })
            df_board = pd.DataFrame(board_rows)

            col_main_graph, col_main_table = st.columns([5, 4])
            with col_main_graph:
                st.markdown(f"**📉 S-Parameter 데이터 곡선 (💡 좌측 상단 체크박스로 마커 동시 On/Off 연동)**")
                
                # 1) S11 그래프 그리기
                fig_s11_plot = go.Figure()
                fig_s11_plot.add_trace(go.Scatter(x=df_s1p["freq"], y=df_s1p["s11"], mode='lines', name='S11 궤적', line=dict(color='#d62728', width=2.5)))
                
                # 체크박스 스위치 온오프 상태에 따라 정밀 동기화 노출
                if show_markers_toggle:
                    fig_s11_plot.add_trace(go.Scatter(
                        x=df_cham["freq"], y=all_s11_markers_y, mode='markers+text', name='마커 포인트',
                        marker=dict(color='black', size=11, symbol='triangle-down'), 
                        text=marker_symbols_labels, textposition='top center', textfont=dict(color='black', size=11, family='Arial')
                    ))
                    fig_s11_plot.add_annotation(
                        xref="paper", yref="paper", x=0.02, y=0.95, showarrow=False, align="left",
                        text=s11_embed_text, font=dict(size=12, color="blue", family="Courier New"),
                        bgcolor="rgba(255, 255, 255, 0.9)", bordercolor="rgba(0,0,0,0.15)", borderwidth=1, borderpad=6
                    )
                
                fig_s11_plot.add_hline(y=target_s11, line_width=2, line_dash="dash", line_color="Purple", annotation_text=f"Target {target_s11}dB", annotation_position="top left")
                fig_s11_plot.update_layout(
                    xaxis_title="Frequency (MHz)", yaxis_title="S11 Return Loss (dB)", height=350, margin=dict(l=10, r=10, t=10, b=10),
                    yaxis=dict(range=[scale_s11[0], scale_s11[1]], tickmode='linear', tick0=scale_s11[0], dtick=step_s11),
                    showlegend=False
                )
                st.plotly_chart(fig_s11_plot, use_container_width=True)
                
                # 2) VSWR 그래프 그리기
                fig_vswr_plot = go.Figure()
                fig_vswr_plot.add_trace(go.Scatter(x=df_s1p["freq"], y=df_s1p["vswr"], mode='lines', name='VSWR 궤적', line=dict(color='#bcbd22', width=2.5)))
                
                if show_markers_toggle:
                    fig_vswr_plot.add_trace(go.Scatter(
                        x=df_cham["freq"], y=all_vswr_markers_y, mode='markers+text', name='마커 포인트',
                        marker=dict(color='black', size=11, symbol='triangle-down'), 
                        text=marker_symbols_labels, textposition='top center', textfont=dict(color='black', size=11, family='Arial')
                    ))
                    fig_vswr_plot.add_annotation(
                        xref="paper", yref="paper", x=0.02, y=0.95, showarrow=False, align="left",
                        text=vswr_embed_text, font=dict(size=12, color="blue", family="Courier New"),
                        bgcolor="rgba(255, 255, 255, 0.9)", bordercolor="rgba(0,0,0,0.15)", borderwidth=1, borderpad=6
                    )
                
                fig_vswr_plot.add_hline(y=target_vswr, line_width=2, line_dash="dash", line_color="Red", annotation_text=f"Target {target_vswr}", annotation_position="top left")
                fig_vswr_plot.update_layout(
                    xaxis_title="Frequency (MHz)", yaxis_title="VSWR", height=350, margin=dict(l=10, r=10, t=10, b=10),
                    yaxis=dict(range=[scale_vswr[0], scale_vswr[1]], tickmode='linear', tick0=scale_vswr[0], dtick=step_vswr),
                    showlegend=False
                )
                st.plotly_chart(fig_vswr_plot, use_container_width=True)
                
            with col_main_table:
                st.markdown(f"**🔌 네트워크 분석기 S-Parameter (Marker Table)** <br><small>Spec Limit: S11 ≤ {target_s11}dB ｜ VSWR ≤ {target_vswr}</small>", unsafe_allow_html=True)
                st.dataframe(df_board, use_container_width=True, hide_index=True, height=295)
                
                st.markdown(f"**📡 챔버 방사성능 (Eff, Avg)** <br><small>Spec Limit: Eff ≥ {target_eff}% ｜ Avg Gain ≥ {target_avg_gain}dBi</small>", unsafe_allow_html=True)
                display_cham_df = df_cham.copy()
                display_cham_df["freq"] = display_cham_df["freq"].map(lambda x: f"{x:.0f} MHz")
                display_cham_df["eff"] = display_cham_df["eff"].map(lambda x: f"{round(x, 2):.2f} %")
                display_cham_df["avg_gain"] = display_cham_df["avg_gain"].map(lambda x: f"{round(x, 2):.2f} dBi")
                display_cham_df.columns = ["주파수 대역", "방사 효율 (%)", "평균 이득 (Average dBi)"]
                st.dataframe(display_cham_df, use_container_width=True, hide_index=True, height=350)

            # --- 분석 리포트 매핑 ---
            st.markdown("---")
            st.subheader("🔍 임피던스 매칭-공간 방사 결합 디버깅 분석 리포트")
            
            fail_s_param_bands, fail_radiation_bands = [], []
            for i in range(len(df_cham)):
                f_mhz = df_cham.iloc[i]["freq"]
                idx_s = (df_s1p["freq"] - f_mhz).abs().argmin()
                
                if df_s1p.iloc[idx_s]["s11"] > target_s11 or df_s1p.iloc[idx_s]["vswr"] > target_vswr:
                    fail_s_param_bands.append(f"{f_mhz:.0f}MHz")
                if df_cham.iloc[i]["eff"] < target_eff or df_cham.iloc[i]["avg_gain"] < target_avg_gain:
                    fail_radiation_bands.append(f"{f_mhz:.0f}MHz")

            if not fail_radiation_bands:
                st.success("✅ **[안테나 무선 전 구간 분석 최적화 만족]**\n\n모든 마커 주파수 대역에서 S-Parameter 회로 반사 조건과 챔버 공간 방사 효율 규격을 완벽하게 통과했습니다.")
            else:
                st.warning("📊 **[전체 주파수 매칭-방사 인과관계 추적 추이 분석]**")
                if fail_s_param_bands:
                    st.error(f"❌ **회로 매칭 불량 구간 발견:** 대역 [{', '.join(fail_s_param_bands)}] 은 S-Parameter 규격 한계치를 초과했습니다.\n\n해당 구간 성능 저하의 주원인은 **회로 입력단 임피던스 불평형**이므로, 매칭 회로 튜닝을 선행 하십시오.")
                
                struct_interference_bands = [b for b in fail_radiation_bands if b not in fail_s_param_bands]
                if struct_interference_bands:
                    st.warning(f"🚨 **기구물 차폐 및 무선 전력 흡수 구간 발견:** 대역 [{', '.join(struct_interference_bands)}] 은 회로 매칭이 정상임에도 최종 효율이 미달되었습니다.\n\n해당 구간은 방사체 소자와 **주변 금속 간의 기구적 이격 거리를 재조정**하십시오.")
                           
        except Exception as e: st.error(f"⚠️ S1p 연동 결합 오류 발생: {e}")
    else:
        st.info("📱 임피던스 결합 진단 분석을 시작하려면 왼쪽 사이드바에 데이터를 업로드해 주세요.")


# ---------------------------------------------------------
# 📊 [메뉴 1] Passive 성능 분석
# ---------------------------------------------------------
elif menu_selection == "Passive 성능 분석":
    if summary_file is not None:
        try:
            if summary_file.name.lower().endswith('.csv'): df_sum_raw = pd.read_csv(summary_file, header=None)
            else: df_sum_raw = pd.read_excel(summary_file, header=None, engine='openpyxl')
                
            header_row_idx = None
            for idx in range(len(df_sum_raw)):
                row_vals = [str(v).strip().lower() for v in df_sum_raw.iloc[idx].values]
                if "no." in row_vals and "freq.[mhz]" in row_vals:
                    header_row_idx = idx
                    break
            df_clean_data = df_sum_raw.iloc[(header_row_idx+1):].dropna(subset=[0]).reset_index(drop=True)
            freqs = pd.to_numeric(df_clean_data[1], errors='coerce').astype(float).values
            effs = pd.to_numeric(df_clean_data[15], errors='coerce').astype(float).values
            avg_gains = pd.to_numeric(df_clean_data[16], errors='coerce').astype(float).values
            peak_gains = pd.to_numeric(df_clean_data[17], errors='coerce').astype(float).values
            
            valid_mask = ~np.isnan(freqs)
            freqs, effs, avg_gains, peak_gains = freqs[valid_mask], effs[valid_mask], avg_gains[valid_mask], peak_gains[valid_mask]

            st.subheader("📋 1. 전 주파수 대역 타겟 스펙 자동 판정 요약 (Total PwrSum 기준)")
            summary_rows = []
            pass_count = 0
            for i in range(len(freqs)):
                is_pass = (effs[i] >= target_eff) and (avg_gains[i] >= target_avg_gain)
                status_str = "✅ PASS" if is_pass else "❌ FAIL"
                if is_pass: pass_count += 1
                summary_rows.append({"주파수 (MHz)": f"{freqs[i]:.0f} MHz", "Eff. (%)": f"{round(effs[i], 2):.2f} %", "Avg (dBi)": f"{round(avg_gains[i], 2):.2f} dBi", "Peak (dBi)": f"{round(peak_gains[i], 2):.2f} dBi", "스펙 검사": status_str})
            st.table(pd.DataFrame(summary_rows))

            st.subheader("📈 2. 주파수별 안테나 성능 가변 추이")
            sort_idx = np.argsort(freqs)
            c_freqs, c_effs, c_avg_gains = freqs[sort_idx], effs[sort_idx], avg_gains[sort_idx]
            col_graph1, col_graph2 = st.columns(2)
            with col_graph1:
                st.markdown("**[종합 방사 효율 추이]**")
                fig_eff = go.Figure()
                fig_eff.add_trace(go.Scatter(x=c_freqs, y=c_effs, mode='lines+markers', name='Actual Efficiency', line=dict(color='#1f77b4', width=3)))
                fig_eff.add_trace(go.Scatter(x=[c_freqs[0], c_freqs[-1]], y=[target_eff, target_eff], mode='lines', name='Target Spec', line=dict(color='red', dash='dash')))
                fig_eff.update_layout(xaxis_title="Frequency (MHz)", yaxis_title="Efficiency (%)", height=320)
                st.plotly_chart(fig_eff, use_container_width=True)
            with col_graph2:
                st.markdown("**[평균 이득(Average Gain) 추이]**")
                fig_avg = go.Figure()
                fig_avg.add_trace(go.Scatter(x=c_freqs, y=c_avg_gains, mode='lines+markers', name='Actual Avg Gain', line=dict(color='#2ca02c', width=3)))
                fig_avg.add_trace(go.Scatter(x=[c_freqs[0], c_freqs[-1]], y=[target_avg_gain, target_avg_gain], mode='lines', name='Target Spec', line=dict(color='red', dash='dash')))
                fig_avg.update_layout(xaxis_title="Frequency (MHz)", yaxis_title="Average Gain (dBi)", height=320)
                st.plotly_chart(fig_avg, use_container_width=True)

            st.markdown("---")
            st.subheader("🔍 Passive 성능 실시간 분석 리포트")
            fail_count = len(freqs) - pass_count
            min_eff_idx, max_eff_idx = np.argmin(effs), np.argmax(effs)
            st.info(f"▶ **대역 판정 결과:** 전체 {len(freqs)}개 대역 중 **{pass_count}개 합격 / {fail_count}개 불합격** 상태입니다.\n\n▶ **성능 극점 추적:** 최저 효율 대역은 **{freqs[min_eff_idx]:.0f} MHz ({effs[min_eff_idx]:.2f} %)** 이며, 최고 효율 대역은 **{freqs[max_eff_idx]:.0f} MHz ({effs[max_eff_idx]:.2f} %)** 입니다.")
        except Exception as e: st.error(f"⚠️ 요약 파일 해석 중 오류 발생: {e}")
    else: st.info("📱 '1. 방사 효율 및 이득 요약 업로드'를 진행해 주세요.")


# ---------------------------------------------------------
# ⚖️ [메뉴 2] 교차 편파 분석
# ---------------------------------------------------------
elif menu_selection == "교차 편파 분석":
    if summary_file is not None:
        try:
            if summary_file.name.lower().endswith('.csv'): df_sum_raw = pd.read_csv(summary_file, header=None)
            else: df_sum_raw = pd.read_excel(summary_file, header=None, engine='openpyxl')
            header_row_idx = None
            for idx in range(len(df_sum_raw)):
                row_vals = [str(v).strip().lower() for v in df_sum_raw.iloc[idx].values]
                if "no." in row_vals and "freq.[mhz]" in row_vals:
                    header_row_idx = idx
                    break
            df_clean_data = df_sum_raw.iloc[(header_row_idx+1):].dropna(subset=[0]).reset_index(drop=True)
            freqs = pd.to_numeric(df_clean_data[1], errors='coerce').astype(float).values
            h_effs = pd.to_numeric(df_clean_data[3], errors='coerce').astype(float).values
            v_effs = pd.to_numeric(df_clean_data[9], errors='coerce').astype(float).values
            total_effs = pd.to_numeric(df_sum_raw.iloc[(header_row_idx+1):, 15].reset_index(drop=True))
            
            total_effs = pd.to_numeric(total_effs, errors='coerce').astype(float).values
            valid_mask = ~np.isnan(freqs)
            freqs, h_effs, v_effs, total_effs = freqs[valid_mask], h_effs[valid_mask], v_effs[valid_mask], total_effs[valid_mask]

            st.subheader("⚖️ 주파수별 수평(H-Pol) vs 수직(V-Pol) 편파 격리도 분석")
            fig_pol = go.Figure()
            fig_pol.add_trace(go.Scatter(x=freqs, y=total_effs, mode='lines+markers', name='Total Power-Sum', line=dict(color='purple', width=3)))
            fig_pol.add_trace(go.Scatter(x=freqs, y=h_effs, mode='lines+markers', name='Theta-Pol (Horizontal)', line=dict(color='#1f77b4', width=2)))
            fig_pol.add_trace(go.Scatter(x=freqs, y=v_effs, mode='lines+markers', name='Phi-Pol (Vertical)', line=dict(color='#ff7f0e', width=2)))
            fig_pol.update_layout(xaxis_title="Frequency (MHz)", yaxis_title="Efficiency (%)", height=380)
            st.plotly_chart(fig_pol, use_container_width=True)

            pol_rows, excellent_xpd_bands, bad_xpd_bands = [], [], []
            for i in range(len(freqs)):
                dom_pol = "Horizontal" if h_effs[i] > v_effs[i] else "Vertical"
                diff_db = abs(10 * np.log10(max(1e-3, h_effs[i])) - 10 * np.log10(max(1e-3, v_effs[i])))
                if diff_db >= 15.0: excellent_xpd_bands.append(f"{freqs[i]:.0f}MHz")
                elif diff_db < 5.0: bad_xpd_bands.append(f"{freqs[i]:.0f}MHz")
                pol_rows.append({"주파수": f"{freqs[i]:.0f} MHz", "종합 효율 (%)": f"{total_effs[i]:.2f} %", "H 효율 (%)": f"{h_effs[i]:.2f} %", "V 효율 (%)": f"{v_effs[i]:.2f} %", "우세 편향": dom_pol, "편파 격리 수준 (XPD, dB)": f"{round(diff_db, 2):.2f} dB"})
            st.table(pd.DataFrame(pol_rows))

            st.markdown("---")
            st.subheader("🔍 편파 격리도(XPD) 실시간 분석 리포트")
            ex_txt = ", ".join(excellent_xpd_bands) if excellent_xpd_bands else "없음"
            bd_txt = ", ".join(bad_xpd_bands) if bad_xpd_bands else "없음"
            st.warning(f"▶ **✅ 우수한 편파 순도 대역 (XPD ≥ 15dB):** [{ex_txt}]\n특정 편파로 에너지가 집중되어 통신 품질이 좋습니다.\n\n▶ **❌ 편파 열화 대역 (XPD < 5dB):** [{bd_txt}]\nH와 V 성분이 혼재하여 편파성을 잃어버렸습니다.")
        except Exception as e: st.error(f"⚠️ 편파 분석 오류: {e}")
    else: st.info("📱 왼쪽 사이드바에서 '1. 방사 효율 및 이득 요약 업로드'를 먼저 올려주세요.")


# ---------------------------------------------------------
# 🎯 [메뉴 3] 방사패턴 분석(성능)
# ---------------------------------------------------------
elif menu_selection == "방사패턴 분석(성능)":
    if raw_file is not None:
        try:
            if raw_file.name.lower().endswith('.csv'): df_raw_table = pd.read_csv(raw_file, header=None)
            else: df_raw_table = pd.read_excel(raw_file, header=None, engine='openpyxl')
            parsed_data = {}
            for r_idx in range(len(df_raw_table) - 2):
                cell_f = str(df_raw_table.iloc[r_idx, 0]).strip()
                cell_type = str(df_raw_table.iloc[r_idx, 2]).strip().lower()
                if ("mhz" in cell_f.lower()) and ("power-sum" in cell_type or "pwrsum" in cell_type or "total" in cell_type) and ("gain" in cell_type):
                    extracted_nums = re.findall(r"\d+\.\d+|\d+", cell_f)
                    if not extracted_nums: continue
                    f_key = float(extracted_nums[0])
                    phi_row = df_raw_table.iloc[r_idx + 1].dropna().tolist()
                    phi_angles = [float(str(p).split('=')[1]) if "ph=" in str(p).lower() else float(p) for p in phi_row]
                    theta_angles, gain_rows = [], []
                    
                    curr_row = r_idx + 2
                    while curr_row < len(df_raw_table):
                        n_c0 = str(df_raw_table.iloc[curr_row, 0]).strip()
                        if "mhz" in n_c0.lower(): break
                        if "th=" in n_c0.lower():
                            t_deg = float(n_c0.lower().split('=')[1])
                            r_d = pd.to_numeric(df_raw_table.iloc[curr_row, 1:(1+len(phi_angles))], errors='coerce').astype(float).values
                            theta_angles.append(t_deg)
                            gain_rows.append(r_d)
                        curr_row += 1
                    if gain_rows: parsed_data[f_key] = {'theta': theta_angles, 'phi': phi_angles, 'matrix': np.array(gain_rows)}

            if parsed_data:
                available_keys = sorted(list(parsed_data.keys()))
                option_mapping = {f"✨ {k:.0f} MHz 종합 방사패턴 관찰": k for k in available_keys}
                selected_display = st.selectbox("패턴 주파수 선택", list(option_mapping.keys()))
                selected_key = option_mapping[selected_display]
                active_block = parsed_data[selected_key]
                matrix, t_angles, p_angles = active_block['matrix'], active_block['theta'], active_block['phi']
                
                scale_range = st.slider("2D 패턴 다이내믹 레인지(Scale) 설정 (dBi)", -50.0, 15.0, (-40.0, 10.0), step=5.0)
                st.markdown("### 🔮 3D 파워 합산 성능 입체 패턴")
                offset_matrix = matrix - np.min(matrix) + 10.0
                T, P = np.meshgrid(np.radians(t_angles), np.radians(p_angles), indexing='ij')
                X, Y, Z = offset_matrix * np.sin(T) * np.cos(P), offset_matrix * np.sin(T) * np.sin(P), offset_matrix * np.cos(T)
                if len(t_angles) == 1:
                    X, Y, Z = np.vstack([X, X * 1.01]), np.vstack([Y, Y * 1.01]), np.vstack([Z, Z + 0.1])
                    plot_color = np.vstack([matrix, matrix])
                else: plot_color = matrix
                fig_3d_perf = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, surfacecolor=plot_color, colorscale='Jet')])
                fig_3d_perf.update_layout(margin=dict(l=0, r=0, t=10, b=10), height=450)
                st.plotly_chart(fig_3d_perf, use_container_width=True)
                
                st.markdown("---")
                st.markdown("### 🎯 2D 평면 컷(Cut) 방사 성능 디버깅 (장비 뷰어 완벽 동기화)")
                col_xy, col_zx, col_yz = st.columns(3)
                
                t_arr, p_arr = np.array(t_angles), np.array(p_angles)
                
                # 💡 [핵심 보정] Theta 180도를 360도로 펼치기 위한 보정 배열
                # t_arr이 [0, 15, ..., 180]이면, 360 - t_arr[::-1][1:] 은 [195, ..., 345, 360]이 됩니다.
                if max(t_arr) <= 180:
                    theta_full_360 = np.concatenate([t_arr, 360 - t_arr[::-1][1:]])
                else:
                    theta_full_360 = t_arr

                with col_xy:
                    target_t_idx = (np.abs(t_arr - 90.0)).argmin()
                    gains_xy = matrix[target_t_idx, :]
                    max_xy, pk_xy_idx, n_pts = np.max(gains_xy), np.argmax(gains_xy), len(gains_xy)
                    l_idx, r_idx = pk_xy_idx, pk_xy_idx
                    for _ in range(n_pts):
                        n_i = (l_idx - 1) % n_pts
                        if gains_xy[n_i] < max_xy - 3.0: break
                        l_idx = n_i
                    for _ in range(n_pts):
                        n_i = (r_idx + 1) % n_pts
                        if gains_xy[n_i] < max_xy - 3.0: break
                        r_idx = n_i
                    hpbw_xy = (p_angles[r_idx] - p_angles[l_idx]) % 360
                    if hpbw_xy == 0: hpbw_xy = 360.0
                    
                    fig_xy = go.Figure()
                    fig_xy.add_trace(go.Scatterpolar(r=gains_xy, theta=p_angles, mode='lines', line=dict(color='#FF1493', width=2), name='XY-Cut'))
                    fig_xy.update_layout(polar=dict(angularaxis=dict(direction="counterclockwise", rotation=0, period=360), radialaxis=dict(range=[scale_range[0], scale_range[1]], ticksuffix=" dBi")), height=320, margin=dict(l=20, r=20, t=20, b=20), title=dict(text="<b>H-Cut (Theta=90)</b>", x=0.5))
                    st.plotly_chart(fig_xy, use_container_width=True)
                    st.caption(f"ℹ️ **Peak**: {round(max_xy, 2):.2f} dBi ｜ **3dB 빔폭(HPBW)**: {hpbw_xy:.1f}°")

                with col_zx:
                    idx_0 = (np.abs(p_arr - 0.0)).argmin()
                    idx_180 = (np.abs(p_arr - 180.0)).argmin()
                    
                    if max(t_arr) <= 180:
                        # 반대편 Phi 각도의 데이터를 가져와서 결합
                        gains_zx_full = np.concatenate([matrix[:, idx_0], matrix[::-1, idx_180][1:]])
                    else:
                        gains_zx_full = matrix[:, idx_0]
                        
                    fig_zx = go.Figure()
                    fig_zx.add_trace(go.Scatterpolar(r=gains_zx_full, theta=theta_full_360, mode='lines', line=dict(color='#1f77b4', width=2), name='ZX-Cut'))
                    fig_zx.update_layout(polar=dict(angularaxis=dict(direction="clockwise", rotation=90, period=360), radialaxis=dict(range=[scale_range[0], scale_range[1]], ticksuffix=" dBi")), height=320, margin=dict(l=20, r=20, t=20, b=20), title=dict(text="<b>E1-Cut (Phi=0)</b>", x=0.5))
                    st.plotly_chart(fig_zx, use_container_width=True)
                    st.caption(f"ℹ️ **Peak**: {round(np.max(gains_zx_full), 2):.2f} dBi")

                with col_yz:
                    idx_90 = (np.abs(p_arr - 90.0)).argmin()
                    # 90도의 반대편은 270도
                    idx_270 = (np.abs(p_arr - 270.0)).argmin()
                    
                    if max(t_arr) <= 180:
                        # 반대편 Phi 각도의 데이터를 가져와서 결합
                        gains_yz_full = np.concatenate([matrix[:, idx_90], matrix[::-1, idx_270][1:]])
                    else:
                        gains_yz_full = matrix[:, idx_90]
                        
                    fig_yz = go.Figure()
                    fig_yz.add_trace(go.Scatterpolar(r=gains_yz_full, theta=theta_full_360, mode='lines', line=dict(color='#2ca02c', width=2), name='YZ-Cut'))
                    fig_yz.update_layout(polar=dict(angularaxis=dict(direction="clockwise", rotation=90, period=360), radialaxis=dict(range=[scale_range[0], scale_range[1]], ticksuffix=" dBi")), height=320, margin=dict(l=20, r=20, t=20, b=20), title=dict(text="<b>E2-Cut (Phi=90)</b>", x=0.5))
                    st.plotly_chart(fig_yz, use_container_width=True)
                    st.caption(f"ℹ️ **Peak**: {round(np.max(gains_yz_full), 2):.2f} dBi")

                st.markdown("---")
                st.subheader("🔍 방사패턴 형상 및 공간 사각지대 실시간 분석 리포트")
                xy_ripple = np.max(gains_xy) - np.min(gains_xy)
                pattern_type_str = "✅ 우수한 전방향성 안테나" if xy_ripple <= 5.0 else ("🎯 지향성 안테나 특성 감지" if xy_ripple >= 15.0 else "📊 일반 준무지향성 방사 구조 패턴")
                hpbw_guide_str = "⚠️ 빔 집중도 저하" if hpbw_xy >= 90.0 else ("⚠️ 통신 사각지대 위험" if hpbw_xy <= 30.0 else "✅ 적정 무선 커버리지 조건 만족")
                idx_0, idx_180_for_fb = (np.abs(np.array(p_angles) - 0.0)).argmin(), (np.abs(np.array(p_angles) - 180.0)).argmin()
                fb_ratio = gains_xy[idx_0] - gains_xy[idx_180_for_fb]
                fb_guide_str = "❌ 후방 누설" if fb_ratio < 3.0 else "✅ 전방 집중 우수"
                total_cells = matrix.size
                null_density = (np.sum(matrix <= -15.0) / total_cells) * 100.0

                col_main_graph, col_main_table = st.columns(2)
                with col_main_graph: st.success(f"▶ **[A] 형상:** {pattern_type_str}\n\n▶ **[B] 빔폭:** {hpbw_guide_str}\n\n▶ **[C] 전후방비:** {fb_guide_str}")
                with col_main_table:
                    if null_density >= 20.0: st.error(f"🚨 **[D] 음영 위험 포착 (Null 밀도: {null_density:.1f}%):** 기구물 간섭 분석을 하세요.")
                    else: st.info(f"📊 **[D] 음영 안정적 구조 (Null 밀도: {null_density:.1f}%)**")
            else: st.error("⚠️ 데이터 파싱 오류")
        except Exception as e: st.error(f"⚠️ 에러 발생: {e}")
    else: st.info("📱 '2. 각도별 Raw Data 업로드'를 진행해 주세요.")


# ---------------------------------------------------------
# 🔮 [메뉴 4] 방사패턴 분석(RAW)
# ---------------------------------------------------------
elif menu_selection == "방사패턴 분석(RAW)":
    if raw_file is not None:
        try:
            if raw_file.name.lower().endswith('.csv'): df_raw_table = pd.read_csv(raw_file, header=None)
            else: df_raw_table = pd.read_excel(raw_file, header=None, engine='openpyxl')
                
            parsed_data = {}
            for r_idx in range(len(df_raw_table) - 2):
                cell_f = str(df_raw_table.iloc[r_idx, 0]).strip()
                cell_type = str(df_raw_table.iloc[r_idx, 2]).strip().lower()
                if ("mhz" in cell_f.lower()) and ("power-sum" in cell_type or "pwrsum" in cell_type or "total" in cell_type) and ("gain" in cell_type):
                    extracted_nums = re.findall(r"\d+\.\d+|\d+", cell_f)
                    if not extracted_nums: continue
                    f_key = float(extracted_nums[0])
                    phi_row = df_raw_table.iloc[r_idx + 1].dropna().tolist()
                    phi_angles = [float(str(p).split('=')[1]) if "ph=" in str(p).lower() else float(p) for p in phi_row]
                    theta_angles, gain_rows = [], []
                    
                    curr_row = r_idx + 2
                    while curr_row < len(df_raw_table):
                        next_cell_0 = str(df_raw_table.iloc[curr_row, 0]).strip()
                        if "mhz" in next_cell_0.lower() or "pol" in str(df_raw_table.iloc[curr_row, 2]).lower(): break
                        if "th=" in next_cell_0.lower():
                            try:
                                t_deg = float(next_cell_0.lower().split('=')[1])
                                r_data = pd.to_numeric(df_raw_table.iloc[curr_row, 1:(1+len(phi_angles))], errors='coerce').astype(float).values
                                if not np.isnan(r_data).all():
                                    theta_angles.append(t_deg)
                                    gain_rows.append(r_data)
                            except: pass
                        curr_row += 1
                    if gain_rows: parsed_data[f_key] = {'theta': theta_angles, 'phi': phi_angles, 'matrix': np.array(gain_rows)}

            if parsed_data:
                available_keys = sorted(list(parsed_data.keys()))
                option_mapping = {f"✨ {k:.0f} MHz 생데이터 입체 디버깅": k for k in available_keys}
                selected_display = st.selectbox("정밀 관찰 주파수 선택", list(option_mapping.keys()))
                selected_key = option_mapping[selected_display]
                active_block = parsed_data[selected_key]
                matrix, t_angles, p_angles = active_block['matrix'], active_block['theta'], active_block['phi']
                
                st.markdown("#### 🎛️ 각도 관심 영역(ROI) 및 이득 스케일(Scale) 설정")
                col_s1, col_s2, col_s3 = st.columns(3)
                with col_s1:
                    if min(t_angles) == max(t_angles): th_range = (float(min(t_angles)), float(max(t_angles)))
                    else: th_range = st.slider("Theta (수직각) 분석 범위", float(min(t_angles)), float(max(t_angles)), (float(min(t_angles)), float(max(t_angles))), step=1.0)
                with col_s2: ph_range = st.slider("Phi (수평각) 분석 범위", float(min(p_angles)), float(max(p_angles)), (float(min(p_angles)), float(max(p_angles))), step=5.0)
                with col_s3: scale_range_raw = st.slider("방사 차트 스케일 범위 (dBi)    ", -50.0, 15.0, (-40.0, 10.0), step=5.0)
                
                t_idxs = [idx for idx, t in enumerate(t_angles) if th_range[0] <= t <= th_range[1]]
                p_idxs = [idx for idx, p in enumerate(p_angles) if ph_range[0] <= p <= ph_range[1]]
                
                if not t_idxs or not p_idxs: st.warning("⚠️ 범위 내에 데이터가 없습니다.")
                else:
                    sub_matrix = matrix[np.ix_(t_idxs, p_idxs)]
                    sub_t, sub_p = [t_angles[i] for i in t_idxs], [p_angles[i] for i in p_idxs]
                    gains_raw_sweep = sub_matrix[0, :]
                    raw_max_gain, raw_peak_idx, n_raw_pts = np.max(gains_raw_sweep), np.argmax(gains_raw_sweep), len(gains_raw_sweep)
                    r_left, r_right = raw_peak_idx, raw_peak_idx
                    for _ in range(n_raw_pts):
                        n_idx = (r_left - 1) % n_raw_pts
                        if gains_raw_sweep[n_idx] < raw_max_gain - 3.0: break
                        r_left = n_idx
                    for _ in range(n_raw_pts):
                        n_idx = (r_right + 1) % n_raw_pts
                        if gains_raw_sweep[n_idx] < raw_max_gain - 3.0: break
                        r_right = n_idx
                    raw_hpbw_val = (sub_p[r_right] - sub_p[r_left]) % 360
                    if raw_hpbw_val == 0: raw_hpbw_val = 360.0

                    st.markdown("---")
                    col_plot3d, col_plot2d = st.columns(2)
                    with col_plot3d:
                        st.markdown("**🔮 필터링 매트릭스 기반 3D 형상 복원**")
                        offset_matrix = sub_matrix - np.min(matrix) + 10.0
                        T, P = np.meshgrid(np.radians(sub_t), np.radians(sub_p), indexing='ij')
                        X, Y, Z = offset_matrix * np.sin(T) * np.cos(P), offset_matrix * np.sin(T) * np.sin(P), offset_matrix * np.cos(T)
                        if len(sub_t) == 1:
                            X, Y, Z = np.vstack([X, X * 1.01]), np.vstack([Y, Y * 1.01]), np.vstack([Z, Z + 0.1])
                            plot_color = np.vstack([sub_matrix, sub_matrix])
                        else: plot_color = sub_matrix
                        fig_3d = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, surfacecolor=plot_color, colorscale='Jet')])
                        fig_3d.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=400)
                        st.plotly_chart(fig_3d, use_container_width=True)
                        
                    with col_plot2d:
                        st.markdown("**🎯 필터링 구간 기준 2D Polar 매핑 곡선**")
                        fig_polar_raw = go.Figure()
                        is_theta_sweep = min(sub_p) == max(sub_p)
                        
                        if is_theta_sweep:
                            lbl_axis = "수직각 (Theta)"
                            sub_t_arr = np.array(sub_t)
                            
                            # 반대편 Phi의 데이터를 가져와서 완전한 원형으로 구성
                            opposite_phi = (sub_p[0] + 180) % 360
                            idx_opp = (np.abs(np.array(p_angles) - opposite_phi)).argmin()
                            
                            if max(sub_t_arr) <= 180:
                                plot_theta = np.concatenate([sub_t_arr, 360 - sub_t_arr[::-1][1:]])
                                opp_gains = matrix[np.ix_(t_idxs, [idx_opp])][:, 0]
                                plot_r = np.concatenate([gains_raw_sweep, opp_gains[::-1][1:]])
                            else:
                                plot_theta = sub_t_arr
                                plot_r = gains_raw_sweep
                                
                            plot_rotation = 90
                        else:
                            lbl_axis = "수평각 (Phi)"
                            plot_theta = sub_p
                            plot_r = gains_raw_sweep
                            plot_rotation = 0
                            
                        fig_polar_raw.add_trace(go.Scatterpolar(r=plot_r, theta=plot_theta, mode='lines+markers', line=dict(color='#FF1493', width=2.5), name='단면 타겟 선', hovertemplate=f'<b>{lbl_axis}</b>: %{{theta}}°<br><b>이득 (Gain)</b>: %{{r:.2f}} dBi<extra></extra>'))
                        fig_polar_raw.update_layout(polar=dict(angularaxis=dict(rotation=plot_rotation, direction="clockwise", period=360), radialaxis=dict(range=[scale_range_raw[0], scale_range_raw[1]], ticksuffix=" dBi")), height=400)
                        st.plotly_chart(fig_polar_raw, use_container_width=True)
                        st.caption(f"ℹ️ **구간 패턴 참고 정보** ｜ **Peak Gain**: {round(raw_max_gain, 2):.2f} dBi ｜ **빔 중심 각도**: {sub_p[raw_peak_idx]:.1f}° ｜ **3dB 빔폭 (HPBW)**: {raw_hpbw_val:.1f}°")
                        
                    st.markdown("---")
                    st.subheader("🔍 방사패턴 형상 및 공간 사각지대 실시간 분석 리포트")
                    xy_ripple_raw = np.max(gains_raw_sweep) - np.min(gains_raw_sweep)
                    p_type_raw_str = "✅ 우수한 전방향성 안테나" if xy_ripple_raw <= 5.0 else ("🎯 지향성 안테나 특성 감지" if xy_ripple_raw >= 15.0 else "📊 일반 준무지향성 영역 방사 패턴")
                    p_hpbw_raw_str = "⚠️ 빔 집중도 저하: 반사판 구조를 키우세요." if raw_hpbw_val >= 90.0 else ("⚠️ 통신 사각지대 위험: 무지향 구조 변경 필요" if raw_hpbw_val <= 30.0 else "✅ 지정 구간 내 안정적 커버리지 형성 확보")
                    
                    idx_0_r, idx_180_r = (np.abs(np.array(sub_p) - 0.0)).argmin(), (np.abs(np.array(sub_p) - 180.0)).argmin()
                    fb_ratio_raw = gains_raw_sweep[idx_0_r] - gains_raw_sweep[idx_180_r]
                    p_fb_raw_str = "❌ 후방 방사 누설 감지: 차폐재 보강 필요" if fb_ratio_raw < 3.0 else "✅ 후방 에너지 차폐 전방 집중 유효"

                    total_cells_raw = sub_matrix.size
                    null_cells_raw = np.sum(sub_matrix <= -15.0)
                    null_density_raw = (null_cells_raw / total_cells_raw) * 100.0
                    
                    col_i1_r, col_i2_r = st.columns(2)
                    with col_i1_r: st.success(f"▶ **[A] 형상:** {p_type_raw_str} (구간 Ripple: {xy_ripple_raw:.2f} dB)\n\n▶ **[B] 빔폭:** {p_hpbw_raw_str} (HPBW: {raw_hpbw_val:.1f}°)\n\n▶ **[C] 전후방비:** {p_fb_raw_str} (F/B Ratio: {fb_ratio_raw:.2f} dB)")
                    with col_i2_r:
                        if null_density_raw >= 20.0: st.error(f"🚨 **[D] 음영 위험 수준 포착 (Null 밀도: {null_density_raw:.1f}%):** 기구물 간섭을 즉시 제거하세요.")
                        elif null_density_raw <= 5.0: st.success(f"✅ **[D] 음영 사각지대 안전 수준 (Null 밀도: {null_density_raw:.1f}%):** 사각지대 안전 수준 충족.")
                        else: st.info(f"📊 **[D] 일반 공간 음영 밀도 (Null 밀도: {null_density_raw:.1f}%):** 사각지대가 안정적으로 케어됩니다.")
        except Exception as e: st.error(f"⚠️ 연산 중 에러 발생: {e}")
    else: st.info("📱 공식 챔버 패턴 Raw 파일을 업로드해 주세요.")


# ---------------------------------------------------------
# 📋 [메뉴 6] 엔지니어 종합 진단 리포트
# ---------------------------------------------------------
elif menu_selection == "📋 엔지니어 종합 진단 리포트":
    if summary_file is not None and raw_file is not None:
        try:
            if summary_file.name.lower().endswith('.csv'): df_sum_raw = pd.read_csv(summary_file, header=None)
            else: df_sum_raw = pd.read_excel(summary_file, header=None, engine='openpyxl')
            header_row_idx = None
            for idx in range(len(df_sum_raw)):
                row_vals = [str(v).strip().lower() for v in df_sum_raw.iloc[idx].values]
                if "no." in row_vals and "freq.[mhz]" in row_vals:
                    header_row_idx = idx
                    break
            df_clean_data = df_sum_raw.iloc[(header_row_idx+1):].dropna(subset=[0]).reset_index(drop=True)
            freqs = pd.to_numeric(df_clean_data[1], errors='coerce').astype(float).values
            effs = pd.to_numeric(df_clean_data[15], errors='coerce').astype(float).values
            avg_gains = pd.to_numeric(df_clean_data[16], errors='coerce').astype(float).values
            h_effs = pd.to_numeric(df_clean_data[3], errors='coerce').astype(float).values
            v_effs = pd.to_numeric(df_clean_data[9], errors='coerce').astype(float).values
            valid_mask = ~np.isnan(freqs)
            freqs, effs, avg_gains, h_effs, v_effs = freqs[valid_mask], effs[valid_mask], avg_gains[valid_mask], h_effs[valid_mask], v_effs[valid_mask]

            if raw_file.name.lower().endswith('.csv'): df_raw_table = pd.read_csv(raw_file, header=None)
            else: df_raw_table = pd.read_excel(raw_file, header=None, engine='openpyxl')
            parsed_data = {}
            for r_idx in range(len(df_raw_table) - 2):
                cell_f = str(df_raw_table.iloc[r_idx, 0]).strip()
                cell_type = str(df_raw_table.iloc[r_idx, 2]).strip().lower()
                if ("mhz" in cell_f.lower()) and ("power-sum" in cell_type or "pwrsum" in cell_type or "total" in cell_type) and ("gain" in cell_type):
                    extracted_nums = re.findall(r"\d+\.\d+|\d+", cell_f)
                    if not extracted_nums: continue
                    f_key = float(extracted_nums[0])
                    phi_row = df_raw_table.iloc[r_idx + 1].dropna().tolist()
                    phi_angles = [float(str(p).split('=')[1]) if "ph=" in str(p).lower() else float(p) for p in phi_row]
                    theta_angles, gain_rows = [], []
                    curr_row = r_idx + 2
                    while curr_row < len(df_raw_table):
                        next_cell_0 = str(df_raw_table.iloc[curr_row, 0]).strip()
                        if "mhz" in next_cell_0.lower() or "pol" in str(df_raw_table.iloc[curr_row, 2]).lower(): break
                        if "th=" in next_cell_0.lower():
                            try:
                                t_deg = float(next_cell_0.lower().split('=')[1])
                                r_data = pd.to_numeric(df_raw_table.iloc[curr_row, 1:(1+len(phi_angles))], errors='coerce').astype(float).values
                                if not np.isnan(r_data).all():
                                    theta_angles.append(t_deg)
                                    gain_rows.append(r_data)
                            except: pass
                        curr_row += 1
                    if gain_rows: parsed_data[f_key] = {'theta': theta_angles, 'phi': phi_angles, 'matrix': np.array(gain_rows)}

            st.subheader("📋 안테나 방사 성능 종합 진단 성적서 (Master Report)")
            pass_count = sum((effs >= target_eff) & (avg_gains >= target_avg_gain))
            fail_count = len(freqs) - pass_count

            total_xpd_db, bad_xpd_bands, excellent_xpd_bands = [], [], []
            for i in range(len(freqs)):
                diff_db = abs(10 * np.log10(max(1e-3, h_effs[i])) - 10 * np.log10(max(1e-3, v_effs[i])))
                total_xpd_db.append(diff_db)
                if diff_db >= 15.0: excellent_xpd_bands.append(f"{freqs[i]:.0f}M")
                elif diff_db < 5.0: bad_xpd_bands.append(f"{freqs[i]:.0f}M")
            avg_xpd_db = np.mean(total_xpd_db)
            xpd_status = "✅ 우수한 편파 순도" if avg_xpd_db >= 15.0 else ("⚠️ 편파 왜곡 주의" if avg_xpd_db >= 5.0 else "❌ 편파 붕괴")

            all_ripples, all_hpbws, all_fbs, all_nulls = [], [], [], []
            for f_k in parsed_data.keys():
                mat = parsed_data[f_k]['matrix']
                p_ang = parsed_data[f_k]['phi']
                g_xy = mat[0, :]
                all_ripples.append(np.max(g_xy) - np.min(g_xy))
                mx, p_i, n_p = np.max(g_xy), np.argmax(g_xy), len(g_xy)
                l_i, r_i = p_i, p_i
                for _ in range(n_p):
                    n_idx = (l_i - 1) % n_p
                    if g_xy[n_idx] < mx - 3.0: break
                    l_i = n_idx
                for _ in range(n_p):
                    n_idx = (r_i + 1) % n_p
                    if g_xy[n_idx] < mx - 3.0: break
                    r_i = n_idx
                all_hpbws.append((p_ang[r_i] - p_ang[l_i]) % 360)
                i0, i180 = (np.abs(np.array(p_ang) - 0.0)).argmin(), (np.abs(np.array(p_ang) - 180.0)).argmin()
                all_fbs.append(g_xy[i0] - g_xy[i180])
                all_nulls.append((np.sum(mat <= -15.0) / mat.size) * 100.0)

            m_ripple, m_hpbw, m_fb, m_null = np.mean(all_ripples), np.mean(all_hpbws), np.mean(all_fbs), np.mean(all_nulls)
            omni_directional_status = "✅ 우수한 전방향성 안테나" if m_ripple <= 5.0 else ("🎯 지향성 안테나 특성 감지" if m_ripple >= 15.0 else "📊 일반 무지향성 방사 분포")
            hpbw_status = "⚠️ 빔 집중도 저하: 반사판 보강 요망" if m_hpbw >= 90.0 else ("⚠️ 통신 사각지대 위험: 무지향 구조 변경 필요" if m_hpbw <= 30.0 else "✅ 서비스 최적 빔 가이드폭 만족")

            report_template_text = f"""
====================================================================================================
                        📋 안테나 방사 성능 및 무선 품질 종합 디버깅 진단 성적서
====================================================================================================

1. PASSIVE EVALUATION (성능 규격 평가 종합)
   ▶ 판정 요약: 총 {len(freqs)}개 마커 주파수 대역 중 [{pass_count}개 SPEC PASS] / [{fail_count}개 SPEC FAIL]
   ▶ 디버깅 리스크 대상: {'Low Band 대역 효율 저하, 0순위 인과관계 메뉴 진단 요망' if fail_count > 0 else '전 대역 목표 규격 만족 상태 도달 완료'}

2. ⚖️ CROSS-POLARIZATION DISCRIMINATION (교차 편파 특성 검사)
   ▶ 연산 지표: 전 주파수 평균 격리도 수준 = [{round(avg_xpd_db, 2):.2f} dB] ➡️ 판정 결과: [{xpd_status}]

3. 🎯 RADIATION PATTERN & COVERAGE (방사 패턴 매트릭스 형상 자동 식별 분석)
   ▶ [A] 빔 전방향성 판정 (Ripple): 전 대역 평균 2D 패턴 리플 수치 = [{round(m_ripple, 2):.2f} dB] ➡️ {omni_directional_status}
   ▶ [B] 3dB 빔폭 커버리지 (HPBW): 전 대역 평균 수평 면적 빔폭 수치 = [{round(m_hpbw, 1):.1f}°] ➡️ {hpbw_status}
   ▶ [C] 전후방 방사 분포비 (F/B Ratio): 전 대역 평균 에너지 격리 지표 = [{round(m_fb, 2):.2f} dB]
   ▶ [D] 3D 구면 사각지대 밀도 (Null Density): 전 사방 안테나 총 음영 밀도 = [{round(m_null, 1):.1f} %]

====================================================================================================
"""
            st.code(report_template_text, language="text")
        except Exception as e: st.error(f"⚠️ 리포트 생성 실패: {e}")
    else: st.info("📱 종합 리포트 조회를 위해 요약 파일과 Raw 파일을 모두 업로드해 주세요.")
