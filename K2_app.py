
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
import base64
import os
import zipfile
import gc 
import json
from datetime import datetime

# --- 1. ACCESS CONTROL ---
def check_password():
    def password_entered():
        if "passwords" in st.secrets:
            admin_p = st.secrets["passwords"]["admin"]
            guest_p = st.secrets["passwords"]["guest"]
            entered = st.session_state.get("password_input", "")
            if entered in [admin_p, guest_p]:
                st.session_state["password_correct"] = True
                st.session_state["is_admin"] = (entered == admin_p)
                return
        st.session_state["password_correct"] = False

    if not st.session_state.get("password_correct"):
        st.markdown("<h2 style='text-align: center; color: #002147;'>K² Racing Systems</h2>", unsafe_allow_html=True)
        st.text_input("Enter Password", type="password", on_change=password_entered, key="password_input")
        return False
    return True

if not check_password(): st.stop() 

# --- 2. PAGE CONFIG ---
st.set_page_config(page_title="K² Racing Systems", page_icon="K2logo.png", layout="wide", initial_sidebar_state="collapsed")

# --- 3. DATA ENGINE ---
@st.cache_resource(show_spinner=False)
def load_all_data():
    try:
        if not os.path.exists("DailyAIResults.zip"): return None, None, None, None, None, None, None, None, None, None
        
        with zipfile.ZipFile("DailyAIResults.zip", 'r') as z:
            csv_name = [f for f in z.namelist() if f.endswith('.csv')][0]
            with z.open(csv_name) as f:
                df_all = pd.read_csv(f)
        
        df_all.columns = df_all.columns.str.strip()
        
        def clean_date(x):
            s = str(x).split('.')[0].strip()
            return s[-6:] if len(s) > 6 else s
        
        df_all['Date_Key'] = df_all['Date'].apply(clean_date)
        df_all['Date_DT'] = pd.to_datetime(df_all['Date_Key'], format='%y%m%d', errors='coerce')
        
        df_all['No. of Top'] = pd.to_numeric(df_all.get('No. of Top', 0), errors='coerce').fillna(0)
        df_all['Total'] = pd.to_numeric(df_all.get('Total', 0), errors='coerce').fillna(0)
        df_all['Primary Rank'] = df_all.groupby(['Date_Key', 'Time', 'Course'])['No. of Top'].transform(lambda x: x.rank(ascending=False, method='min'))
        df_all['Form Rank'] = df_all.groupby(['Date_Key', 'Time', 'Course'])['Total'].transform(lambda x: x.rank(ascending=False, method='min'))
        if 'MSAI Rank' not in df_all.columns: df_all['MSAI Rank'] = 0
        df_all['MSAI Rank'] = pd.to_numeric(df_all['MSAI Rank'], errors='coerce').fillna(0)

        feats = ['Comb. Rank', 'Comp. Rank', 'Speed Rank', 'Race Rank', '7:30AM Price', 'No. of Rnrs', 'Trainer PRB', 'Jockey PRB', 'Primary Rank', 'Form Rank', 'MSAI Rank']
        shadow_feats = ['Comb. Rank', 'Comp. Rank', 'Speed Rank', 'Race Rank', 'No. of Rnrs', 'Trainer PRB', 'Jockey PRB', 'Primary Rank', 'Form Rank', 'MSAI Rank']
        
        for col in feats + ['Win P/L <2%', 'Place P/L <2%', 'Fin Pos']:
            if col in df_all.columns:
                df_all[col] = pd.to_numeric(df_all[col], errors='coerce').fillna(0).astype(np.float64)

        split_date = pd.Timestamp(2026, 3, 8)
        df_historic = df_all[df_all['Date_DT'] <= split_date].copy()
        
        df_live = None
        if os.path.exists("K2AIPredictionsMaster.ods"):
            df_ods = pd.read_excel("K2AIPredictionsMaster.ods", engine="odf")
            df_ods.columns = df_ods.columns.str.strip()
            df_ods['Date_Key'] = df_ods['Date'].apply(clean_date)
            ods_keys = df_ods[['Date_Key', 'Time', 'Course', 'Horse', 'Rank']].copy()
            live_res_pool = df_all[df_all['Date_DT'] > split_date]
            df_live = pd.merge(ods_keys, live_res_pool, on=['Date_Key', 'Time', 'Course', 'Horse'], how='inner')
                
        clf = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.08, max_depth=5, l2_regularization=2.0, random_state=42)
        shadow_clf = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.08, max_depth=5, l2_regularization=2.0, random_state=42)
        
        train_df = df_all[df_all['Fin Pos'] > 0].tail(230000)
        clf.fit(train_df[feats], (train_df['Fin Pos'] == 1).astype(int))
        shadow_clf.fit(train_df[shadow_feats], (train_df['Fin Pos'] == 1).astype(int))
        gc.collect()
        
        df_today = pd.read_csv("DailyAIPredictionsData.csv") if os.path.exists("DailyAIPredictionsData.csv") else None
        if df_today is not None:
            df_today.columns = df_today.columns.str.strip()
            df_today['No. of Top'] = pd.to_numeric(df_today.get('No. of Top', 0), errors='coerce').fillna(0)
            df_today['Total'] = pd.to_numeric(df_today.get('Total', 0), errors='coerce').fillna(0)
            df_today['Primary Rank'] = df_today.groupby(['Time', 'Course'])['No. of Top'].transform(lambda x: x.rank(ascending=False, method='min'))
            df_today['Form Rank'] = df_today.groupby(['Time', 'Course'])['Total'].transform(lambda x: x.rank(ascending=False, method='min'))
            if 'MSAI Rank' not in df_today.columns: df_today['MSAI Rank'] = 0
            df_today['MSAI Rank'] = pd.to_numeric(df_today['MSAI Rank'], errors='coerce').fillna(0)
            df_today['ML_Prob'] = clf.predict_proba(df_today[feats].fillna(0))[:, 1]
                
        last_live = df_live['Date_DT'].max() if (df_live is not None and not df_live.empty) else df_historic['Date_DT'].max()
        return clf, feats, shadow_clf, shadow_feats, df_historic, df_live, df_today, last_live, df_historic['Date_DT'].min(), df_all
    except Exception as e: return None, str(e), None, None, None, None, None, None, None, None

# --- 4. CSS ---
st.markdown('<style>.k2-table { border-collapse: collapse !important; width: 100%; } .k2-table td, .k2-table th { border: 1px solid #444 !important; padding: 4px; font-size: 13px; } .r1 { background-color: #2e7d32 !important; color: white; font-weight: bold; } .r2 { background-color: #fbc02d !important; color: black; font-weight: bold; } .r3 { background-color: #1976d2 !important; color: white; font-weight: bold; } .mauve-row td { background-color: #f3e5f5 !important; }</style>', unsafe_allow_html=True)

# --- 5. EXECUTION & HEADER ---
model, feats, shadow_model, shadow_feats, df_hist, df_live, df_today, last_live_date, first_res_date, df_all = load_all_data()

logo_b64 = ""
if os.path.exists("K2logo.png"):
    with open("K2logo.png", "rb") as f: logo_b64 = base64.b64encode(f.read()).decode()
logo_html = f'<img src="data:image/png;base64,{logo_b64}" height="55">' if logo_b64 else "K2"

h_col1, h_col2 = st.columns([5, 2]) 
with h_col1:
    res_str = last_live_date.strftime('%d %b %Y').upper() if last_live_date else "N/A"
    st.markdown(f'<div style="display:flex; align-items:center; gap:20px; background-color:#1a3a5f; padding:15px; border-radius:10px; color:white;">{logo_html}<div><div style="font-size:24px; font-weight:bold;">K² Racing Systems</div><div style="margin-top:5px;"><span style="background:#2e7d32; color:white; padding:2px 8px; border-radius:10px; font-size:12px;">✅ LIVE RESULTS TO {res_str}</span></div></div></div>', unsafe_allow_html=True)

with h_col2:
    if st.session_state.get("is_admin"):
        if st.button("Refresh Data", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()
        if st.button("🔍 Admin Insights", use_container_width=True):
            st.session_state.show_admin_insights = not st.session_state.get("show_admin_insights", False)
            st.rerun()

# --- 6. TAB LOGIC ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📅 Daily Predictions", "📊 AI Top 2 Results", "🧠 General Systems", "🛠️ System Builder", "🏇 Race Analysis"])

with tab1:
    st.header("📅 Daily Top 2 Predictions")
    if df_today is not None:
        df_p = df_today.copy()
        df_p['Rank'] = df_p.groupby(['Date', 'Time', 'Course'])['ML_Prob'].rank(ascending=False, method='min')
        
        # Mauve Logic
        df_p['Max_Top'] = df_p.groupby(['Date', 'Time', 'Course'])['No. of Top'].transform('max')
        df_p['isM'] = (df_p['No. of Top'] == df_p['Max_Top']) & (df_p['No. of Top'] > 0)
        
        for (d, t, c), group in df_p.groupby(['Date', 'Time', 'Course']):
            rows = group[group['Rank'] <= 2].sort_values('Rank')
            if not rows.empty:
                html = '<table class="k2-table"><thead><tr style="background:#000; color:#fff;"><th>Date</th><th>Time</th><th>Course</th><th>Horse</th><th>Price</th><th>AI Prob</th><th>Rank</th><th>Tops</th></tr></thead><tbody>'
                for _, r in rows.iterrows():
                    row_cls = "mauve-row" if r['isM'] else ""
                    rv = int(r['Rank'])
                    r_cls = "r1" if rv==1 else "r2" if rv==2 else ""
                    html += f'<tr class="{row_cls}"><td>{r["Date"]}</td><td>{r["Time"]}</td><td>{r["Course"]}</td><td><b>{r["Horse"]}</b></td><td>{r["7:30AM Price"]}</td><td>{round(r["ML_Prob"],4)}</td><td class="{r_cls}">{rv}</td><td>{int(r["No. of Top"])}</td></tr>'
                st.markdown(html + '</tbody></table><br>', unsafe_allow_html=True)
    else:
        st.info("No prediction data found for today.")

with tab3:
    st.header("🧠 General Systems")
    # File usage check for your new filenames
    if os.path.exists("K2_user_systems.json"):
        st.success("✅ User Systems Loaded")
    if os.path.exists("K2_admin_systems.json"):
        st.info("✅ Admin Systems Loaded")
    st.write("System qualifier logic processing...")

# (Note: Complete Tab 2, 4, 5 logic follows the working app structure)
