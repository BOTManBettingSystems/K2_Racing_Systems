import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
import base64
import os
import zipfile
import gc 
import json
from datetime import datetime, timedelta
import extra_streamlit_components as stx

# ---  PAGE CONFIG ---
st.set_page_config(page_title="K² Racing Systems", page_icon="K2logo.png", layout="wide", initial_sidebar_state="expanded")

# --- Initialize Cookie Manager (Must be at the top level) ---
cookie_manager = stx.CookieManager(key="k2_cookie_manager")

# --- 1. ACCESS CONTROL & LOGGING ---
def log_performance(action_name):
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / (1024 * 1024)
        cpu_pct = psutil.cpu_percent(interval=None)
        
        log_file = "K2_performance_log.csv"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        file_exists = os.path.exists(log_file)
        with open(log_file, mode='a', newline='', encoding='utf-8') as f:
            if not file_exists:
                f.write("Timestamp,Action,RAM_Used_MB,CPU_Percent\n")
            f.write(f"{timestamp},{action_name},{mem_mb:.1f},{cpu_pct:.1f}\n")
    except ImportError:
        pass

def log_login(role):
    log_file = "K2_login_log.csv"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    client_ip = "Unknown"
    try:
        if hasattr(st, 'context') and hasattr(st.context, 'headers'):
            client_ip = st.context.headers.get("X-Forwarded-For", "Unknown").split(",")[0]
    except Exception:
        pass

    file_exists = os.path.exists(log_file)
    with open(log_file, mode='a', newline='', encoding='utf-8') as f:
        if not file_exists:
            f.write("Timestamp,Role,IP_Address\n")
        f.write(f"{timestamp},{role},{client_ip}\n")
        
def check_password():
    auth_cookie = cookie_manager.get(cookie="k2_auth")
    admin_p = os.environ.get("ADMIN_PASS", st.secrets.get("passwords", {}).get("admin", "fallback1"))
    guest_p = os.environ.get("GUEST_PASS", st.secrets.get("passwords", {}).get("guest", "fallback2"))

    if auth_cookie in ["admin_active", "guest_active"]:
        st.session_state["password_correct"] = True
        st.session_state["is_admin"] = (auth_cookie == "admin_active")
        return True

    def password_entered():
        entered = st.session_state.get("password_input", "")
        if entered in [admin_p, guest_p]:
            st.session_state["password_correct"] = True
            st.session_state["is_admin"] = (entered == admin_p)
            
            role = "Admin" if st.session_state["is_admin"] else "Guest"
            log_login(role)
            
            cookie_val = "admin_active" if st.session_state["is_admin"] else "guest_active"
            expire_date = datetime.now() + timedelta(days=30)
            cookie_manager.set("k2_auth", cookie_val, expires_at=expire_date)
            
            if "password_input" in st.session_state:
                del st.session_state["password_input"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.markdown("<h2 style='text-align: center; color: #002147;'>K² Racing Systems</h2>", unsafe_allow_html=True)
        st.text_input("Enter Password", type="password", on_change=password_entered, key="password_input")
        return False
        
    return st.session_state.get("password_correct", False)

if not check_password(): st.stop() 

if "show_admin_insights" not in st.session_state:
    st.session_state.show_admin_insights = False

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
        
        def optimize_memory(df):
            for col in df.select_dtypes(include=['float64']).columns:
                df[col] = pd.to_numeric(df[col], downcast='float')
            for col in df.select_dtypes(include=['int64']).columns:
                df[col] = pd.to_numeric(df[col], downcast='integer')
            for col in ['Course', 'Race Type', 'H/Cap', 'Price Bracket']:
                if col in df.columns:
                    df[col] = df[col].astype('category')
            return df
        
        df_all = optimize_memory(df_all)
        
        def clean_date(x):
            s = str(x).split('.')[0].strip()
            if len(s) > 6: s = s[-6:]
            return s
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
                
        # --- TWO-BRAIN ARCHITECTURE INTEGRATION ---
        # 1. The Sorters (High Complexity)
        clf = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.08, max_depth=5, l2_regularization=2.0, random_state=42)
        shadow_clf = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.08, max_depth=5, l2_regularization=2.0, random_state=42)
        # 2. The Pricer (Low Complexity / Calibrated)
        cal_clf = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, max_depth=3, l2_regularization=15.0, min_samples_leaf=250, random_state=42)
        
        train_df = df_all[df_all['Fin Pos'] > 0].tail(230000)
        
        clf.fit(train_df[feats], (train_df['Fin Pos'] == 1).astype(int))
        shadow_clf.fit(train_df[shadow_feats], (train_df['Fin Pos'] == 1).astype(int))
        cal_clf.fit(train_df[feats], (train_df['Fin Pos'] == 1).astype(int))
        
        gc.collect()
                
        last_live = df_live['Date_DT'].max() if (df_live is not None and not df_live.empty) else datetime.now()
        first_hist = df_historic['Date_DT'].min() if not df_historic.empty else datetime(2024,1,1)
        
        log_performance("Full ML Retrain & History Load")
        return clf, feats, shadow_clf, shadow_feats, cal_clf, df_historic, df_live, last_live, first_hist, df_all
    except Exception as e: return None, str(e), None, None, None, None, None, None, None, None

def prep_system_builder_data(_df, _model, feats, _shadow_model=None, shadow_feats=None, _cal_model=None, is_live_today=False):
    b_df = _df.copy()
    b_df.columns = b_df.columns.str.strip()
    
    if 'Date_Key' not in b_df.columns and 'Date' in b_df.columns:
        b_df['Date_Key'] = b_df['Date'].astype(str).str.split('.').str[0].str.strip()
        b_df['Date_Key'] = b_df['Date_Key'].apply(lambda s: s[-6:] if len(s) > 6 else s)
    
    if 'Date_DT' not in b_df.columns and 'Date_Key' in b_df.columns:
        b_df['Date_DT'] = pd.to_datetime(b_df['Date_Key'], format='%y%m%d', errors='coerce')
        
    if not is_live_today:
        b_df = b_df[b_df.get('Fin Pos', 0) > 0].copy()

    # --- 🛡️ PREDICTION VAULT INTEGRATION ---
    vault_file = "K2_Prediction_Vault.csv"
    vault_keys = ['Date_Key', 'Time', 'Course', 'Horse']

    if 'Time' in b_df.columns: b_df['Time'] = b_df['Time'].astype(str).str.split('.').str[0].str.strip()
    if 'Course' in b_df.columns: b_df['Course'] = b_df['Course'].astype(str).str.strip().str.title()
    if 'Horse' in b_df.columns: b_df['Horse'] = b_df['Horse'].astype(str).str.strip().str.title()

    if os.path.exists(vault_file):
        vault_df = pd.read_csv(vault_file)
        vault_df['Date_Key'] = vault_df['Date_Key'].astype(str)
        vault_df['Time'] = vault_df['Time'].astype(str).str.split('.').str[0].str.strip()
        vault_df['Course'] = vault_df['Course'].astype(str).str.strip().str.title()
        vault_df['Horse'] = vault_df['Horse'].astype(str).str.strip().str.title()
        
        b_df = pd.merge(b_df, vault_df[vault_keys + ['ML_Prob', 'Shadow_Prob']], on=vault_keys, how='left')
        b_df['ML_Prob'] = pd.to_numeric(b_df['ML_Prob'], errors='coerce')
        b_df['Shadow_Prob'] = pd.to_numeric(b_df['Shadow_Prob'], errors='coerce')
    else:
        b_df['ML_Prob'] = np.nan
        b_df['Shadow_Prob'] = np.nan

    # --- CALCULATE MISSING AI PREDICTIONS ---
    missing_mask = b_df['ML_Prob'].isna()
    if missing_mask.any():
        missing_df = b_df[missing_mask].copy()
        b_df.loc[missing_mask, 'ML_Prob'] = _model.predict_proba(missing_df[feats].fillna(0))[:, 1]
        
        if _shadow_model is not None and shadow_feats is not None:
            missing_shadow = [f for f in shadow_feats if f not in missing_df.columns]
            if not missing_shadow:
                b_df.loc[missing_mask, 'Shadow_Prob'] = _shadow_model.predict_proba(missing_df[shadow_feats].fillna(0))[:, 1]
            else:
                b_df.loc[missing_mask, 'Shadow_Prob'] = 0
        else:
            b_df.loc[missing_mask, 'Shadow_Prob'] = 0
            
    if 'Shadow_Prob' not in b_df.columns:
        b_df['Shadow_Prob'] = 0

    if is_live_today:
        b_df['Pure Rank'] = b_df.groupby(['Time', 'Course'])['Shadow_Prob'].rank(ascending=False, method='min')
    else:
        b_df['Pure Rank'] = b_df.groupby(['Date_Key', 'Time', 'Course'])['Shadow_Prob'].rank(ascending=False, method='min')
        
    b_df['7:30AM Price'] = pd.to_numeric(b_df.get('7:30AM Price', 0), errors='coerce')
    b_df['BSP'] = pd.to_numeric(b_df.get('BSP', 0), errors='coerce')
    b_df['Age'] = pd.to_numeric(b_df.get('Age', 0), errors='coerce').fillna(0)
    
    b_df['No. of Top'] = pd.to_numeric(b_df.get('No. of Top', 0), errors='coerce').fillna(0)
    b_df['Total'] = pd.to_numeric(b_df.get('Total', 0), errors='coerce').fillna(0)
    
    if not is_live_today:
        b_df['Win P/L <2%'] = pd.to_numeric(b_df.get('Win P/L <2%', 0), errors='coerce')
        b_df['Place P/L <2%'] = pd.to_numeric(b_df.get('Place P/L <2%', 0), errors='coerce')
        b_df['Fin Pos'] = pd.to_numeric(b_df.get('Fin Pos', 0), errors='coerce')
        b_df['Is_Win'] = np.where(b_df['Win P/L <2%'] > 0, 1, 0)
        b_df['Is_Place'] = np.where((b_df['Fin Pos'] >= 1) & (b_df['Fin Pos'] <= 3), 1, 0)
    
    b_df['No. of Rnrs'] = pd.to_numeric(b_df.get('No. of Rnrs', 0), errors='coerce')
    
    b_df['Trainer PRB Rank'] = 0
    b_df['Jockey PRB Rank'] = 0
    b_df['Primary Rank'] = 0
    b_df['Form Rank'] = 0
    
    if 'Trainer PRB' in b_df.columns:
        b_df['Trainer PRB Rank'] = b_df.groupby(['Date_Key', 'Time', 'Course'])['Trainer PRB'].transform(lambda x: x.rank(ascending=False, method='min'))
    if 'Jockey PRB' in b_df.columns:
        b_df['Jockey PRB Rank'] = b_df.groupby(['Date_Key', 'Time', 'Course'])['Jockey PRB'].transform(lambda x: x.rank(ascending=False, method='min'))
        
    b_df['Primary Rank'] = b_df.groupby(['Date_Key', 'Time', 'Course'])['No. of Top'].transform(lambda x: x.rank(ascending=False, method='min'))
    b_df['Form Rank'] = b_df.groupby(['Date_Key', 'Time', 'Course'])['Total'].transform(lambda x: x.rank(ascending=False, method='min'))
    
    if is_live_today:
        b_df['Rank'] = b_df.groupby(['Time', 'Course'])['ML_Prob'].rank(ascending=False, method='min')
        b_df['Rank2_Prob'] = b_df.groupby(['Time', 'Course'])['ML_Prob'].transform(lambda x: x.nlargest(2).iloc[-1] if len(x) > 1 else 0)
    else:
        b_df['Rank'] = b_df.groupby(['Date_Key', 'Time', 'Course'])['ML_Prob'].rank(ascending=False, method='min')
        b_df['Rank2_Prob'] = b_df.groupby(['Date_Key', 'Time', 'Course'])['ML_Prob'].transform(lambda x: x.nlargest(2).iloc[-1] if len(x) > 1 else 0)
        
    b_df['Prob Gap'] = b_df['ML_Prob'] - b_df['Rank2_Prob']
    b_df['User Value'] = pd.to_numeric(b_df.get('Value', 0), errors='coerce')
        
    bins = [-1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 15.0, 20.0, 50.0, 100.0, 1000.0]
    labels = ["<1.0", "1.0-2.0", "2.01-3.0", "3.01-4.0", "4.01-5.0", "5.01-6.0", "6.01-7.0", "7.01-8.0", "8.01-9.0", "9.01-10.0", "10.01-11.0", "11.01-15.0", "15.01-20.0", "20.01-50.0", "50.01-100.0", "100.01+"]
    b_df['Price Bracket'] = pd.cut(b_df['7:30AM Price'], bins=bins, labels=labels, right=True)
    b_df['Price Bracket'] = b_df['Price Bracket'].cat.add_categories('Unknown').fillna('Unknown')
    
    # --- TRUE VALUE PRICING (THE PRICER BRAIN) ---
    if _cal_model is not None:
        b_df['True_AI_Prob'] = _cal_model.predict_proba(b_df[feats].fillna(0))[:, 1]
        b_df['True_Value_Price'] = np.where(b_df['True_AI_Prob'] > 0.001, 1.0 / b_df['True_AI_Prob'], 1000.0)
        
        # Override the old hallucinatory Value Price with the true calibrated price
        b_df['Value Price'] = b_df['True_Value_Price']
        
        b_df['Market_Price'] = np.where(b_df['BSP'] > 0, b_df['BSP'], b_df['7:30AM Price'])
        b_df['Value_Edge_Perc'] = ((b_df['Market_Price'] / b_df['True_Value_Price']) - 1) * 100
        
        # Base initial brackets (dynamically overwritten later in the builder if sliders used)
        v_bins = [-np.inf, 0.0, 10.0, np.inf]
        v_labels = ['1. Negative Edge (< 0%)', '2. Fair Value (0% to 10%)', '3. Deep Value (> 10%)']
        b_df['Edge Bracket'] = pd.cut(b_df['Value_Edge_Perc'], bins=v_bins, labels=v_labels)
        b_df['Edge Bracket'] = b_df['Edge Bracket'].cat.add_categories('Unknown').fillna('Unknown')
    else:
        b_df['Value Price'] = 1 / b_df['ML_Prob'] # Fallback
        b_df['Value_Edge_Perc'] = 0.0
        b_df['Edge Bracket'] = 'Unknown'
        
    return b_df

@st.cache_data(show_spinner=False)
def load_daily_data(_model, feats, _shadow_model, shadow_feats, _cal_model):
    try:
        df_today = pd.read_csv("DailyAIPredictionsData.csv") if os.path.exists("DailyAIPredictionsData.csv") else None
        if df_today is not None and not df_today.empty:
            df_today.columns = df_today.columns.str.strip()
            df_today['No. of Top'] = pd.to_numeric(df_today.get('No. of Top', 0), errors='coerce').fillna(0)
            df_today['Total'] = pd.to_numeric(df_today.get('Total', 0), errors='coerce').fillna(0)
            df_today['Primary Rank'] = df_today.groupby(['Time', 'Course'])['No. of Top'].transform(lambda x: x.rank(ascending=False, method='min'))
            df_today['Form Rank'] = df_today.groupby(['Time', 'Course'])['Total'].transform(lambda x: x.rank(ascending=False, method='min'))
            if 'MSAI Rank' not in df_today.columns: df_today['MSAI Rank'] = 0
            df_today['MSAI Rank'] = pd.to_numeric(df_today['MSAI Rank'], errors='coerce').fillna(0)

            missing_feats = [f for f in feats if f not in df_today.columns]
            if not missing_feats and _model is not None:
                df_today = prep_system_builder_data(df_today, _model, feats, _shadow_model, shadow_feats, _cal_model, is_live_today=True)
        
        log_performance("Daily Data Processed")
        return df_today
    except Exception as e:
        return None

@st.cache_resource(show_spinner=False)
def get_prepped_history(_df, _model, feats, _shadow_model, shadow_feats, _cal_model):
    if _df is None or _df.empty: return None
    return prep_system_builder_data(_df, _model, feats, _shadow_model, shadow_feats, _cal_model, is_live_today=False)

@st.cache_data(show_spinner=False)
def load_ods_master():
    if os.path.exists("K2SystemsMaster.ods"):
        return pd.read_excel("K2SystemsMaster.ods", engine="odf")
    return None

# --- 4. CSS ---
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem !important; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] { background-color: rgba(0,0,0,0) !important; color: #1a3a5f !important; }
    button[data-testid="stSidebarCollapse"] { background-color: #1a3a5f !important; color: white !important; border-radius: 5px !important; margin-left: 10px !important; }
    .main .block-container { overflow-x: hidden; -webkit-overflow-scrolling: touch; }
    .k2-table-container { overflow-x: auto; width: 100%; -webkit-overflow-scrolling: touch; }
    .k2-table { border-collapse: collapse !important; width: 100% !important; min-width: 850px !important; table-layout: fixed !important; margin-bottom: 0px !important; }
    .k2-table th, .k2-table td { border: 1px solid #444 !important; padding: 3px 4px !important; font-size: 12.5px !important; white-space: nowrap !important; overflow: hidden !important; text-overflow: ellipsis !important; }
    .k2-table td.r1 { background-color: #2e7d32 !important; color: white !important; font-weight: bold !important; }
    .k2-table td.r2 { background-color: #fbc02d !important; color: black !important; font-weight: bold !important; }
    .k2-table td.r3 { background-color: #1976d2 !important; color: white !important; font-weight: bold !important; }
    .mauve-row td { background-color: #f3e5f5 !important; color: black !important; }
    .k2-table tr:hover td { background-color: #aec6cf !important; color: black !important; }
    .k2-table thead th { background-color: #000 !important; color: white !important; text-transform: uppercase; letter-spacing: 0.5px; white-space: normal !important; line-height: 1.2 !important; vertical-align: bottom !important; }
    .left-head { text-align: left !important; padding-left: 10px !important; }
    .left-text { text-align: left !important; padding-left: 10px !important; }
    .center-text { text-align: center !important; }
    .pos-val { color: #2e7d32 !important; font-weight: bold !important; }
    .neg-val { color: #d32f2f !important; font-weight: bold !important; }
</style>
""", unsafe_allow_html=True)

# --- 5. EXECUTION & HEADER ---
model, feats, shadow_model, shadow_feats, cal_model, df_hist, df_live, last_live_date, first_res_date, df_all = load_all_data()
df_today = load_daily_data(model, feats, shadow_model, shadow_feats, cal_model)
df_all_prepped = get_prepped_history(df_all, model, feats, shadow_model, shadow_feats, cal_model)

if 'expanded_races' not in st.session_state: st.session_state.expanded_races = set()

logo_b64 = ""
if os.path.exists("K2logo.png"):
    with open("K2logo.png", "rb") as f: logo_b64 = base64.b64encode(f.read()).decode()
logo_html = '<img src="data:image/png;base64,' + logo_b64 + '" height="55">' if logo_b64 else "K2"

h_col1, h_col2 = st.columns([5, 2.5]) 
with h_col1:
    res_str = last_live_date.strftime('%d %b %Y').upper() if last_live_date else "08 MAR 2026"
    header_box = '<div style="display:flex; align-items:center; gap:20px; background-color:#1a3a5f; padding:15px; border-radius:10px; color:white;">' + logo_html + '<div>'
    header_box += '<div style="font-size:24px; font-weight:bold;">K² Racing Systems</div>'
    header_box += '<div style="margin-top:5px;"><span style="background:#2e7d32; color:white; padding:2px 8px; border-radius:10px; font-size:12px;">✅ LIVE RESULTS TO ' + res_str + '</span></div>'
    header_box += '</div></div>'
    st.markdown(header_box, unsafe_allow_html=True)

with h_col2:
    if st.session_state.get("is_admin"):
        st.markdown('<div style="margin-top:5px;"></div>', unsafe_allow_html=True) 
        
        b1, b2 = st.columns(2)
        with b1:
            if st.button("🔄 Quick Refresh", help="Reloads today's races and systems instantly", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
        with b2:
            if st.button("⚠️ Re-Train", help="Retrains AI and extracts history (~5 mins)", use_container_width=True):
                st.cache_resource.clear()
                st.cache_data.clear()
                st.rerun()
        
        btn_label = "🔙 Return to Dashboard" if st.session_state.get("show_admin_insights") else "🔍 Admin Insights"
        if st.button(btn_label, key="admin_toggle_btn", use_container_width=True):
            st.session_state.show_admin_insights = not st.session_state.get("show_admin_insights", False)
            st.rerun()

@st.cache_data(show_spinner=False)
def prep_dashboard_data(_df, _model, feats, perf_mode, d_start, d_end, p_min, p_max):
    mask = (_df['7:30AM Price'] >= p_min) & (_df['7:30AM Price'] <= p_max)
    if d_start and d_end:
        mask &= (_df['Date_DT'].dt.date >= d_start) & (_df['Date_DT'].dt.date <= d_end)
        
    res_data = _df[mask & (_df['Fin Pos'] > 0)].copy()
    if res_data.empty: return res_data
    
    if perf_mode == "Live":
        res_data['AI_R'] = pd.to_numeric(res_data.get('Rank', 0), errors='coerce')
    else:
        res_data['AI_S'] = _model.predict_proba(res_data[feats].fillna(0))[:, 1]
        res_data['AI_R'] = res_data.groupby(['Date_Key', 'Time', 'Course'])['AI_S'].rank(ascending=False, method='first')
    return res_data


def clean_csv_df(df_in):
    if df_in is None or df_in.empty: return df_in
    df_out = df_in.copy()
    float_cols = [c for c in df_out.select_dtypes(include=['float', 'float32', 'float64']).columns if 'Prob' not in c]
    df_out[float_cols] = df_out[float_cols].round(2)
    return df_out


# -------------------------------------------------------------------------
# VIEW CONTROLLER
# -------------------------------------------------------------------------

if st.session_state.get("is_admin") and st.session_state.get("show_admin_insights"):
    st.header("🔍 Admin Data Insights (Multi-Factor Analysis)")
    st.markdown("Combine multiple data elements to discover highly profitable 'Golden Rules' hidden in your historical data.")
    
    if df_all_prepped is not None and not df_all_prepped.empty:
        ins_df = df_all_prepped
        
        i_col1, i_col2, i_col3 = st.columns([1.5, 1.5, 1])
        with i_col1:
            race_types_avail = ["All"] + sorted([str(x) for x in ins_df['Race Type'].dropna().unique() if str(x).strip()])
            race_filter = st.selectbox("Analyze Race Type:", race_types_avail)
        with i_col2:
            target_metric = st.selectbox(
                "Sort Results By:", 
                ["Logical Grouping (By Factor)", "Win P/L", "Win ROI (%)", "Win S/R (%)", "Place P/L", "Place ROI (%)", "Place S/R (%)"], 
                index=0
            )
        with i_col3:
            min_bets = st.number_input("Minimum Bets (Sample Size):", min_value=5, max_value=2000, value=25, step=5)
        
        analysis_cols = ['Rank', 'Comb. Rank', 'Speed Rank', 'Race Rank', 'No. of Top', 'Primary Rank', 'Form Rank', 'Class', 'Class Move', 'PRB Rank', 'Trainer PRB Rank', 'Jockey PRB Rank', 'MSAI Rank', 'Price Bracket', 'Pure Rank', 'Edge Bracket']
        avail_cols = [c for c in analysis_cols if c in ins_df.columns]
        
        selected_factors = st.multiselect("Select Factors to Combine (Choose 1 to 4):", avail_cols, default=['No. of Top', 'Speed Rank'])
        
        if race_filter != "All":
            ins_df = ins_df[ins_df['Race Type'] == race_filter]
        
        st.markdown("---")
        
        if not selected_factors:
            st.warning("⚠️ Please select at least one factor from the dropdown above to generate insights.")
        else:
            st.markdown(f"### 🏆 System Analysis for {race_filter} Races")
            
            grp = ins_df.groupby(selected_factors, observed=False).agg(
                Bets=('Horse', 'count'),
                Wins=('Is_Win', 'sum'),
                Profit=('Win P/L <2%', 'sum'),
                Places=('Is_Place', 'sum'),
                Place_Profit=('Place P/L <2%', 'sum')
            ).reset_index()
            
            grp = grp[grp['Bets'] >= min_bets] 
            
            if 'Price Bracket' in selected_factors:
                grp = grp[grp['Price Bracket'] != 'Unknown']
            
            if not grp.empty:
                grp['Strike Rate (%)'] = (grp['Wins'] / grp['Bets']) * 100
                grp['Win ROI (%)'] = (grp['Profit'] / grp['Bets']) * 100
                grp['Place SR (%)'] = (grp['Places'] / grp['Bets']) * 100
                grp['Place ROI (%)'] = (grp['Place_Profit'] / grp['Bets']) * 100
                grp['Total P/L'] = grp['Profit'] + grp['Place_Profit']
                
                if target_metric == "Logical Grouping (By Factor)":
                    ascending_sorts = []
                    for factor in selected_factors:
                        if factor == 'No. of Top': ascending_sorts.append(False) 
                        else: ascending_sorts.append(True)  
                    grp = grp.sort_values(by=selected_factors, ascending=ascending_sorts)
                else:
                    sort_map = {
                        "Win P/L": "Profit", "Win ROI (%)": "Win ROI (%)", "Win S/R (%)": "Strike Rate (%)",
                        "Place P/L": "Place_Profit", "Place ROI (%)": "Place ROI (%)", "Place S/R (%)": "Place SR (%)"
                    }
                    sort_col = sort_map.get(target_metric, 'Profit')
                    grp = grp.sort_values(by=sort_col, ascending=False).head(100)
                
                html_table = """
                <style>
                    .builder-table { border-collapse: collapse; width: 100%; font-size: 14px; font-family: sans-serif; }
                    .builder-table th, .builder-table td { border: 1px solid #ccc; padding: 4px; text-align: center; }
                    .left-align { text-align: left !important; padding-left: 8px !important; }
                    .divider { border-left: 3px solid #1a3a5f !important; }
                </style>
                <div style="overflow-x: auto; width: 100%;">
                <table class="builder-table" style="min-width: 800px;">
                    <thead><tr style="background-color: #1a3a5f; color: white;">
                """
                
                for factor in selected_factors: html_table += f"<th>{factor}</th>"
                html_table += """
                        <th class="divider">Total Bets</th><th>Wins</th><th>Win P/L</th><th>Win S/R</th><th>Win ROI</th>
                        <th class="divider">Places</th><th>Plc P/L</th><th>Plc S/R</th><th>Total P/L</th>
                    </tr></thead><tbody>
                """
                for _, r in grp.iterrows():
                    html_table += "<tr>"
                    for factor in selected_factors:
                        val = r[factor]
                        if isinstance(val, float): val = int(val) if val.is_integer() else f"{val:.1f}"
                        html_table += f"<td style='color:#1a3a5f; font-weight:bold;'>{val}</td>"
                        
                    p_color = "#2e7d32" if r['Profit'] > 0 else "#d32f2f"
                    r_color = "#2e7d32" if r['Win ROI (%)'] > 0 else "#d32f2f"
                    pp_color = "#2e7d32" if r['Place_Profit'] > 0 else "#d32f2f"
                    t_color = "#2e7d32" if r['Total P/L'] > 0 else "#d32f2f"
                        
                    html_table += f"""
                        <td class="divider">{int(r['Bets'])}</td><td>{int(r['Wins'])}</td>
                        <td style="color:{p_color}; font-weight:bold;">£{r['Profit']:.2f}</td><td>{r['Strike Rate (%)']:.1f}%</td>
                        <td style="color:{r_color}; font-weight:bold;">{r['Win ROI (%)']:.1f}%</td>
                        <td class="divider">{int(r['Places'])}</td><td style="color:{pp_color}; font-weight:bold;">£{r['Place_Profit']:.2f}</td>
                        <td>{r['Place SR (%)']:.1f}%</td><td style="color:{t_color}; font-weight:bold;">£{r['Total P/L']:.2f}</td>
                    </tr>"""
                html_table += "</tbody></table></div>"
                st.markdown(html_table, unsafe_allow_html=True)
            else: st.info(f"No combinations found with at least {min_bets} bets.")
    else: st.warning("No data available.")

else:
    st.sidebar.markdown("### 🧭 Main Menu")
    page = st.sidebar.radio(
        "Select an Option:",
        ["📅 Daily Predictions", "📊 AI Top 2 Results", "🧠 General Systems", "🛠️ System Builder", "🏇 Race Analysis", "🧪 Acid Test Environment"]
    )

    # =========================================================================
    # 📅 PAGE 1: DAILY PREDICTIONS
    # =========================================================================
    if page == "📅 Daily Predictions":
        st.header("📅 Daily Top 2 Predictions")
        if df_today is not None and not df_today.empty:
            if 'ML_Prob' in df_today.columns:
                df_p = df_today.copy()
                df_p.columns = df_p.columns.str.strip()
                df_p['Rank'] = df_p.groupby(['Date', 'Time', 'Course'])['ML_Prob'].rank(ascending=False, method='min')
                
                if 'No. of Top' in df_p.columns:
                    df_p['No. of Top'] = pd.to_numeric(df_p['No. of Top'], errors='coerce').fillna(0)
                    df_p['Max_Top'] = df_p.groupby(['Date', 'Time', 'Course'])['No. of Top'].transform('max')
                    df_p['isM'] = (df_p['No. of Top'] == df_p['Max_Top']) & (df_p['No. of Top'] > 0)
                else: df_p['isM'] = False
                
                df_p = df_p.sort_values(by=['Date', 'Time', 'Course', 'Rank'])
                
                ideal_csv_cols = ['Date', 'Time', 'Course', 'Horse', '7:30AM Price', 'ML_Prob', 'Rank', 'No. of Top']
                existing_csv_cols = [c for c in ideal_csv_cols if c in df_p.columns]
                csv_out = df_p[df_p['Rank'] <= 2][existing_csv_cols].copy()
                csv_out = clean_csv_df(csv_out)
                
                timestamp = datetime.now().strftime('%d%m%y_%H%M%S')
                file_name = f"K2_AIPredictions_{timestamp}.csv"
                
                col_dl, col_spacer, col_col = st.columns([1, 3, 0.5])
                with col_dl: st.download_button("Download CSV", csv_out.to_csv(index=False), file_name)
                with col_col:
                    if st.button("Collapse All"): 
                        st.session_state.expanded_races = set()
                        st.rerun()

                w = ["10%", "10%", "12%", "31%", "12%", "12%", "8%", "5%"]
                h_col1, h_col2 = st.columns([19, 1], gap="small")
                with h_col1:
                    header = '<div style="overflow-x: auto; width: 100%;"><table class="k2-table" style="min-width: 800px;"><thead><tr>'
                    header += f'<th style="width:{w[0]};" class="left-head">Date</th><th style="width:{w[1]};" class="left-head">Time</th><th style="width:{w[2]};" class="left-head">Course</th><th style="width:{w[3]};" class="left-head">Horse</th><th style="width:{w[4]};" class="left-head">Price</th><th style="width:{w[5]};" class="left-head">AI Prob</th><th style="width:{w[6]};" class="left-head">Rank</th><th style="width:{w[7]};" class="left-head">Tops</th>'
                    header += '</tr></thead></table></div>'
                    st.markdown(header, unsafe_allow_html=True)

                for (d, t, c), group in df_p.groupby(['Date', 'Time', 'Course'], sort=False):
                    race_id = str(d) + " " + str(t) + " " + str(c)
                    is_expanded = race_id in st.session_state.expanded_races
                    rows = group if is_expanded else group[group['Rank'] <= 2]
                    
                    st.markdown('<div style="margin-top:2px;"></div>', unsafe_allow_html=True)
                    t_col, b_col = st.columns([19, 1], gap="small")
                    with t_col:
                        html = '<div style="overflow-x: auto; width: 100%;"><table class="k2-table" style="min-width: 800px;"><tbody>'
                        for _, r in rows.iterrows():
                            row_cls = "mauve-row" if r['isM'] else ""
                            rv = int(r['Rank'])
                            r_cls = "r1" if rv==1 else "r2" if rv==2 else "r3" if rv==3 else ""
                            html += f'<tr class="{row_cls}"><td style="width:{w[0]};" class="center-text">{r["Date"]}</td><td style="width:{w[1]};" class="center-text">{r["Time"]}</td><td style="width:{w[2]};" class="left-text">{r["Course"]}</td><td style="width:{w[3]};" class="left-text"><b>{r["Horse"]}</b></td><td style="width:{w[4]};" class="center-text">{round(r["7:30AM Price"], 2)}</td><td style="width:{w[5]};" class="center-text">{round(r["ML_Prob"], 4)}</td><td style="width:{w[6]};" class="{r_cls} center-text">{rv}</td><td style="width:{w[7]};" class="center-text">{int(r["No. of Top"])}</td></tr>'
                        st.markdown(html + '</tbody></table></div>', unsafe_allow_html=True)
                    with b_col:
                        if st.button("-" if is_expanded else "+", key="btn_"+race_id):
                            if is_expanded: st.session_state.expanded_races.remove(race_id)
                            else: st.session_state.expanded_races.add(race_id)
                            st.rerun()

    # =========================================================================
    # 📊 PAGE 2: AI TOP 2 RESULTS
    # =========================================================================
    elif page == "📊 AI Top 2 Results":
        if "perf_mode" not in st.session_state: st.session_state.perf_mode = "Live"
        st.markdown('<div class="filter-area">', unsafe_allow_html=True)
        cb1, cb2, cd = st.columns([1, 1, 2])
        if cb1.button("Recent Results (Live)", type="primary" if st.session_state.perf_mode == "Live" else "secondary"): 
            st.session_state.perf_mode = "Live"; st.rerun()
        if cb2.button("Historical Data", type="primary" if st.session_state.perf_mode == "Legacy" else "secondary"): 
            st.session_state.perf_mode = "Legacy"; st.rerun()
        
        ytd_start = datetime(2026, 3, 9).date()
        if st.session_state.perf_mode == "Live":
            df_s = df_live
            f_end = last_live_date.date() if last_live_date else datetime.now().date()
            d_range = cd.date_input("Live Range (Since 9th March)", [ytd_start, f_end], min_value=ytd_start)
        else:
            df_s = df_hist
            f_start = first_res_date.date() if first_res_date else datetime(2024,1,1)
            d_range = cd.date_input("Historical Range (To 8th March)", [f_start, datetime(2026, 3, 8).date()], max_value=datetime(2026, 3, 8).date())
        
        price_options = {"All": (0.0, 1000.0), "Up to 5": (0.0, 5.0), "5 to 10": (5.01, 10.0), "10 to 15": (10.01, 15.0), "15 to 25": (15.01, 25.0), "25 to 50": (25.01, 50.0), "50 to 100": (50.01, 100.0), "> 100": (100.01, 1000.0)}
        sel_range = st.radio("Price Range Quick-Select:", list(price_options.keys()), index=0, horizontal=True)
        start_p, end_p = price_options[sel_range]
        p_range = st.slider("Fine-Tune Price Filter", 0.0, 1000.0, (float(start_p), float(end_p)))
        st.markdown('</div>', unsafe_allow_html=True)
        
        if df_s is not None and not df_s.empty:
            d_start = d_range[0] if len(d_range) > 0 else None
            d_end = d_range[1] if len(d_range) == 2 else d_start
            
            master_tab2_df = prep_dashboard_data(df_s, model, feats, st.session_state.perf_mode, d_start, d_end, p_range[0], p_range[1])
            audit_totals = {'Win': 0.0, 'Place': 0.0}
            
            def render_pick_card(label, data, is_main_cat=False):
                if data.empty: return
                p1_count = len(data[data['AI_R'] == 1])
                p2_count = len(data[data['AI_R'] == 2])
                total_runs = p1_count + p2_count
                st.markdown(f'''<div style="border: 1px solid #444; background-color: #f9f9f9; padding: 10px; border-radius: 4px; margin-bottom: 15px;"><div style="border-left: 6px solid #1a3a5f; padding-left: 10px; margin-bottom: 10px; font-weight: bold; font-size: 14px;">{label} <span style="font-weight: normal; color: #666;">(Total: {total_runs})</span></div>''', unsafe_allow_html=True)
                cols = st.columns(2)
                for i in range(1, 3):
                    pick = data[data['AI_R'] == i]
                    pr = len(pick)
                    with cols[i-1]:
                        if pr > 0:
                            wpl = float(pick['Win P/L <2%'].sum())
                            ppl = float(pick['Place P/L <2%'].sum())
                            wsr = (len(pick[pick['Fin Pos']==1])/pr)*100
                            psr = (len(pick[pick['Fin Pos'] <= 3])/pr)*100
                            wroi, proi = (wpl/pr)*100, (ppl/pr)*100
                            if is_main_cat: audit_totals['Win'] += wpl; audit_totals['Place'] += ppl
                            wpl_cls = "pos-val" if wpl >= 0 else "neg-val"
                            ppl_cls = "pos-val" if ppl >= 0 else "neg-val"
                            wroi_cls = "pos-val" if wroi >= 0 else "neg-val"
                            proi_cls = "pos-val" if proi >= 0 else "neg-val"
                            bg, tx = ('#2e7d32', 'white') if i==1 else ('#fbc02d', 'black')
                            
                            c_box = f'''<div class="pick-box" style="border: 1px solid #ccc; background: white;"><div style="background:{bg}; color:{tx}; text-align:center; font-weight:bold; font-size:12px; padding:3px;">PICK {i} ({pr})</div><div style="padding:8px; font-size:11px; line-height:1.6;"><b>Win P&L:</b> <span class="{wpl_cls}">{round(wpl, 2)}</span> | <b>S/R%:</b> {round(wsr, 1)} | <b>ROI%:</b> <span class="{wroi_cls}">{round(wroi, 1)}%</span><br><b>Place P&L:</b> <span class="{ppl_cls}">{round(ppl, 2)}</span> | <b>S/R%:</b> {round(psr, 1)} | <b>ROI%:</b> <span class="{proi_cls}">{round(proi, 1)}%</span></div></div>'''
                            st.markdown(c_box, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div style="background:#1a3a5f; color:white; padding:8px 15px; font-weight:bold; border-radius:4px; margin-bottom:10px;">TOTAL SYSTEM BREAKDOWN</div>', unsafe_allow_html=True)
            render_pick_card("TOTAL SYSTEM", master_tab2_df, is_main_cat=True)
            
            sc1, sc2 = st.columns(2)
            with sc1: render_pick_card("TOTAL NON-HANDICAP", master_tab2_df[master_tab2_df['H/Cap'].astype(str).str.strip() == 'N'])
            with sc2: render_pick_card("TOTAL HANDICAP", master_tab2_df[master_tab2_df['H/Cap'].astype(str).str.strip() == 'Y'])
            
            for rt in ['A/W', 'Chase', 'Hurdle', 'Turf']:
                st.markdown(f'<div style="background:#1a3a5f; color:white; padding:8px 15px; font-weight:bold; border-radius:4px; margin-top:20px; margin-bottom:10px;">{rt} CATEGORY BREAKDOWN</div>', unsafe_allow_html=True)
                render_pick_card(rt+" Aggregated", master_tab2_df[master_tab2_df['Race Type'].astype(str).str.strip() == rt])
                sc1, sc2 = st.columns(2)
                with sc1: render_pick_card(rt+" Non-Handicap", master_tab2_df[(master_tab2_df['Race Type'].astype(str).str.strip() == rt) & (master_tab2_df['H/Cap'].astype(str).str.strip() == 'N')])
                with sc2: render_pick_card(rt+" Handicap", master_tab2_df[(master_tab2_df['Race Type'].astype(str).str.strip() == rt) & (master_tab2_df['H/Cap'].astype(str).str.strip() == 'Y')])

            if st.session_state.perf_mode == "Live":
                st.markdown('<div style="background:#1a3a5f; color:white; padding:6px 15px; font-weight:bold; border-radius:4px; margin-top:20px;">🏆 LIVE TRACK PERFORMANCE RANKINGS</div>', unsafe_allow_html=True)
                track_stats = []
                for course in df_s['Course'].unique():
                    c_data = df_s[(df_s['Course'] == course) & (df_s['Fin Pos'] > 0)]
                    c_runs = len(c_data)
                    if c_runs >= 2:
                        c_win = float(c_data['Win P/L <2%'].sum())
                        c_roi = (c_win / c_runs) * 100
                        track_stats.append({'Course': course, 'Picks': c_runs, 'Win P/L': c_win, 'ROI%': c_roi})
                if track_stats:
                    top_tracks = pd.DataFrame(track_stats).sort_values('ROI%', ascending=False).head(5)
                    st.table(top_tracks.set_index('Course'))


    # =========================================================================
    # 🧠 PAGE 3: GENERAL SYSTEMS
    # =========================================================================
    elif page == "🧠 General Systems":
        st.header("🧠 General Systems")
        smart_view = st.radio("Select View:", ["📅 Today's Qualifiers", "📊 Live Performance (Master file)"], horizontal=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        if smart_view == "📅 Today's Qualifiers":
            s_col, p_col = st.columns(2)
            with s_col: sort_pref = st.radio("Sort Qualifiers By:", ["System Name (Morning Review)", "Time (Live Racing)"], horizontal=True)
            with p_col:
                if st.session_state.get("is_admin"): pool_choice = st.radio("System Pool (Admin Only):", ["Public", "Admin Secret", "Combined"], horizontal=True)
                else: pool_choice = "Public"
            
            st.markdown("<br>", unsafe_allow_html=True)
            try:
                if df_today is not None and not df_today.empty:
                    if 'ML_Prob' not in df_today.columns:
                        st.warning("Data Check: 'ML_Prob' could not be calculated. Please check your source CSV for missing columns.")
                    else:
                        all_today_picks = []
                        t_df = df_today
                        saved_systems = {}
                        if pool_choice in ["Public", "Combined"] and os.path.exists("K2_user_systems.json"):
                            with open("K2_user_systems.json", "r") as f:
                                try: saved_systems.update(json.load(f))
                                except: pass
                        if pool_choice in ["Admin Secret", "Combined"] and os.path.exists("K2_admin_systems.json"):
                            with open("K2_admin_systems.json", "r") as f:
                                try: saved_systems.update(json.load(f))
                                except: pass

                        if saved_systems:
                            for s_name, s_data in saved_systems.items():
                                s_mask = (
                                    t_df['Race Type'].isin(s_data.get('race_types', [])) &
                                    t_df['H/Cap'].isin(s_data.get('hcap_types', [])) &
                                    (t_df['7:30AM Price'] >= s_data.get('price_min', 0.0)) &
                                    (t_df['7:30AM Price'] <= s_data.get('price_max', 1000.0)) &
                                    (t_df['Prob Gap'] >= s_data.get('min_prob_gap', -100.0))
                                )
                                
                                ai_filt = s_data.get('ai_rank_filter', "Any")
                                if ai_filt == "Any" and s_data.get('rank_1_only', False): ai_filt = "AI Rank 1 Only"
                                if ai_filt == "AI Rank 1 Only": s_mask &= (t_df['Rank'] == 1)
                                elif ai_filt == "Top 2 Only": s_mask &= (t_df['Rank'] <= 2)
                                elif ai_filt == "NOT AI Rank 1": s_mask &= (t_df['Rank'] > 1)
                                elif ai_filt == "NOT Top 3": s_mask &= (t_df['Rank'] > 3)
                                
                                months = s_data.get('months', ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
                                if len(months) < 12:
                                    month_map = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6, "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}
                                    sel_m_nums = [month_map[m] for m in months]
                                    s_mask &= t_df['Date_DT'].dt.month.isin(sel_m_nums)

                                vf = s_data.get('value_filter', "Off")
                                if vf in ["Value vs 7:30AM Price", "Value vs BSP", "AI Value vs 7:30AM", "AI Value vs BSP"]: 
                                    s_mask &= (t_df['7:30AM Price'] > t_df['Value Price'])
                                elif vf in ["My Value vs 7:30AM", "My Value vs BSP"]: 
                                    s_mask &= (t_df['7:30AM Price'] > t_df['User Value'])
                                elif vf in ["NOT AI Value vs 7:30AM", "NOT AI Value vs BSP"]:
                                    s_mask &= (t_df['7:30AM Price'] < t_df['Value Price'])
                                elif vf in ["NOT My Value vs 7:30AM", "NOT My Value vs BSP"]:
                                    s_mask &= (t_df['7:30AM Price'] < t_df['User Value'])

                                rnrs = s_data.get('rnrs', [])
                                r_m = pd.Series(False, index=t_df.index)
                                if "2-7" in rnrs: r_m |= (t_df['No. of Rnrs'] >= 2) & (t_df['No. of Rnrs'] <= 7)
                                if "8-12" in rnrs: r_m |= (t_df['No. of Rnrs'] >= 8) & (t_df['No. of Rnrs'] <= 12)
                                if "13-16" in rnrs: r_m |= (t_df['No. of Rnrs'] >= 13) & (t_df['No. of Rnrs'] <= 16)
                                if ">16" in rnrs: r_m |= (t_df['No. of Rnrs'] > 16)
                                if not r_m.any() and not rnrs: r_m = pd.Series(True, index=t_df.index)
                                s_mask &= r_m

                                if 'Class' in t_df.columns and s_data.get('classes'): s_mask &= t_df['Class'].isin(s_data['classes'])
                                if 'Class Move' in t_df.columns and s_data.get('cm'): s_mask &= t_df['Class Move'].isin(s_data['cm'])
                                if 'Age' in t_df.columns: s_mask &= (t_df['Age'] >= s_data.get('age_min', 1)) & (t_df['Age'] <= s_data.get('age_max', 20))

                                irish_setting = s_data.get('irish', "Any")
                                irish_col = 'Irish?' if 'Irish?' in t_df.columns else 'Irish' if 'Irish' in t_df.columns else None
                                if irish_col and irish_setting != "Any":
                                    t_irish_series = t_df[irish_col].astype(str).str.strip().str.upper()
                                    if irish_setting == "Y (Yes)": s_mask &= (t_irish_series == 'Y')
                                    elif irish_setting == "No (Blank)": s_mask &= (t_irish_series != 'Y')

                                if 'Sex' in t_df.columns and s_data.get('sex'): s_mask &= t_df['Sex'].astype(str).str.strip().str.lower().isin([s.lower() for s in s_data['sex']])
                                if 'Course' in t_df.columns and s_data.get('courses'): s_mask &= t_df['Course'].astype(str).str.strip().isin(s_data['courses'])

                                ranks = s_data.get('ranks', {})
                                for col_name, setting in ranks.items():
                                    if setting != "Any" and col_name in t_df.columns:
                                        num_col = pd.to_numeric(t_df[col_name], errors='coerce')
                                        if setting == "Rank 1": s_mask &= (num_col == 1)
                                        elif setting == "Top 2": s_mask &= (num_col <= 2)
                                        elif setting == "Top 3": s_mask &= (num_col <= 3)
                                        elif setting == "Not Top 3": s_mask &= ((num_col > 3) | (num_col == 0))

                                sys_df = t_df[s_mask].copy()
                                if not sys_df.empty:
                                    sys_df['System Name'] = s_name
                                    all_today_picks.append(sys_df)

                        if all_today_picks:
                            final_df = pd.concat(all_today_picks, ignore_index=True)
                            if pool_choice == "Public": ideal_base_cols = ["Date", "Time", "Course", "Horse", "7:30AM Price", "ML_Prob", "Rank", "No. of Top", "System Name"]
                            else: ideal_base_cols = ["Date", "Time", "Course", "Horse", "7:30AM Price", "ML_Prob", "Rank", "Primary Rank", "Pure Rank", "No. of Top", "System Name"]
                            
                            existing_cols = [c for c in ideal_base_cols if c in final_df.columns]
                            final_df = final_df[existing_cols]
                            
                            if sort_pref == "System Name (Morning Review)": final_df = final_df.sort_values(by=["System Name", "Date", "Time", "Course"], ascending=[True, True, True, True])
                            else: final_df = final_df.sort_values(by=["Date", "Time", "Course", "System Name"], ascending=[True, True, True, True])

                            unique_systems = final_df['System Name'].unique()
                            palette = ["#e8f4f8", "#f8e8e8", "#e8f8e8", "#f8f4e8", "#f4e8f8", "#e8f8f8"]
                            sys_color_map = {sys: palette[i % len(palette)] for i, sys in enumerate(unique_systems)}

                            clean_final_df = clean_csv_df(final_df)
                            csv_data = clean_final_df.to_csv(index=False).encode('utf-8')
                            timestamp = datetime.now().strftime('%d%m%y_%H%M%S')
                            dl_label = "📥 Download Admin Picks to CSV" if pool_choice != "Public" else "📥 Download General Picks to CSV"
                            st.download_button(dl_label, csv_data, f"K2_{pool_choice}_Systems_{timestamp}.csv", "text/csv", key="dl_smart")
                            st.write("")

                            html_table = """<style>.contiguous-table { border-collapse: collapse; width: 100%; font-size: 14px; font-family: sans-serif; } .contiguous-table th, .contiguous-table td { border: 1px solid #ccc; padding: 4px; text-align: left; } .contiguous-table tr:hover { background-color: #0000FF !important; color: white !important; }</style><div style="overflow-x: auto; width: 100%;"><table class="contiguous-table" style="min-width: 900px;"><thead><tr>"""
                            for col in existing_cols: html_table += f"<th>{col}</th>"
                            html_table += "</tr></thead><tbody>"

                            for _, row in final_df.iterrows():
                                bg_color = sys_color_map.get(row.get("System Name"), "")
                                row_style = f"background-color: {bg_color}; color: black;" if bg_color else ""
                                html_table += f"<tr style='{row_style}'>"
                                for col in existing_cols:
                                    val = row[col]
                                    if isinstance(val, float):
                                        if col == "ML_Prob": val = f"{val*100:.1f}%"
                                        elif col in ["Rank", "No. of Top", "Primary Rank", "Pure Rank"]: val = f"{int(val)}"
                                        else: val = f"{val:.2f}"
                                    html_table += f"<td>{val}</td>"
                                html_table += "</tr>"

                            html_table += "</tbody></table></div>"
                            st.markdown(html_table, unsafe_allow_html=True)
                        else: st.info(f"No systems selections found today for the '{pool_choice}' pool.")
                else: st.info("No data available for today's races.")
            except Exception as e: st.error(f"Error loading General Systems: {e}")
                
        else:
            st.markdown("### 📈 Live Performance (Master file)")
            if st.session_state.get("is_admin"):
                perf_file_choice = st.radio("Select Master File to Analyze:", ["Public (K2SystemsMaster.ods)", "Admin (K2AdminMaster.ods)"], horizontal=True)
                target_ods = "K2AdminMaster.ods" if "Admin" in perf_file_choice else "K2SystemsMaster.ods"
            else:
                target_ods = "K2SystemsMaster.ods"
                
            if os.path.exists(target_ods):
                try:
                    df_smart_master = pd.read_excel(target_ods, engine="odf")
                    df_smart_master.columns = df_smart_master.columns.str.strip()
                    if all(c in df_smart_master.columns for c in ['Date', 'Time', 'Course', 'Horse']) and df_all is not None:
                        sys_col_found = None
                        for col in df_smart_master.columns:
                            if col.lower() in ['system name', 'system', 'system_name', 'systems']:
                                sys_col_found = col
                                break
                        
                        def clean_d(x):
                            s = str(x).split('.')[0].strip()
                            return s[-6:] if len(s) > 6 else s
                            
                        df_smart_master['Date_Key'] = df_smart_master['Date'].apply(clean_d)
                        df_smart_master['Time'] = df_smart_master['Time'].astype(str).str.split('.').str[0].str.strip()
                        df_a = df_all.copy()
                        df_a['Time'] = df_a['Time'].astype(str).str.split('.').str[0].str.strip()
                        
                        df_smart_master['Course'] = df_smart_master['Course'].astype(str).str.strip().str.title()
                        df_a['Course'] = df_a['Course'].astype(str).str.strip().str.title()
                        df_smart_master['Horse'] = df_smart_master['Horse'].astype(str).str.strip().str.title()
                        df_a['Horse'] = df_a['Horse'].astype(str).str.strip().str.title()
                        
                        merged_smart = pd.merge(df_smart_master, df_a, on=['Date_Key', 'Time', 'Course', 'Horse'], how='inner')
                        merged_smart['Fin Pos'] = pd.to_numeric(merged_smart['Fin Pos'], errors='coerce')
                        merged_smart = merged_smart[merged_smart['Fin Pos'] > 0]
                        
                        if not merged_smart.empty:
                            if sys_col_found is None:
                                merged_smart['System Name'] = 'All Systems Combined'
                                sys_col_found = 'System Name'
                            else:
                                merged_smart['System Name'] = merged_smart[sys_col_found]
                                sys_col_found = 'System Name'

                            merged_smart['Win P/L <2%'] = pd.to_numeric(merged_smart['Win P/L <2%'], errors='coerce').fillna(0)
                            merged_smart['Place P/L <2%'] = pd.to_numeric(merged_smart['Place P/L <2%'], errors='coerce').fillna(0)
                            merged_smart['Is_Win'] = np.where(merged_smart['Win P/L <2%'] > 0, 1, 0)
                            merged_smart['Is_Place'] = np.where((merged_smart['Fin Pos'] >= 1) & (merged_smart['Fin Pos'] <= 3), 1, 0)
                            
                            merged_smart['Date_DT'] = pd.to_datetime(merged_smart['Date_Key'], format='%y%m%d', errors='coerce')
                            merged_smart['Month_Yr'] = merged_smart['Date_DT'].dt.strftime('%Y - %b')
                            current_month_str = datetime.now().strftime('%Y - %b')
                            
                            all_time = merged_smart.groupby(sys_col_found, observed=False).agg(
                                Bets=('Horse', 'count'), Wins=('Is_Win', 'sum'), Win_Profit=('Win P/L <2%', 'sum'), Places=('Is_Place', 'sum'), Place_Profit=('Place P/L <2%', 'sum')
                            ).reset_index()
                            all_time['Period'] = 'All Time'
                            
                            curr_month_df = merged_smart[merged_smart['Month_Yr'] == current_month_str]
                            if not curr_month_df.empty:
                                curr_month = curr_month_df.groupby(sys_col_found, observed=False).agg(
                                    Bets=('Horse', 'count'), Wins=('Is_Win', 'sum'), Win_Profit=('Win P/L <2%', 'sum'), Places=('Is_Place', 'sum'), Place_Profit=('Place P/L <2%', 'sum')
                                ).reset_index()
                                curr_month['Period'] = current_month_str
                            else:
                                curr_month = all_time.copy()
                                curr_month['Period'] = current_month_str
                                curr_month[['Bets', 'Wins', 'Win_Profit', 'Places', 'Place_Profit']] = 0

                            combined = pd.concat([all_time, curr_month], ignore_index=True)
                            combined['Strike Rate (%)'] = np.where(combined['Bets'] > 0, (combined['Wins'] / combined['Bets'] * 100), 0)
                            combined['Place SR (%)'] = np.where(combined['Bets'] > 0, (combined['Places'] / combined['Bets'] * 100), 0)
                            combined['Win ROI (%)'] = np.where(combined['Bets'] > 0, (combined['Win_Profit'] / combined['Bets'] * 100), 0)
                            combined['Total P/L'] = combined['Win_Profit'] + combined['Place_Profit']
                            
                            combined['SortKey'] = np.where(combined['Period'] == 'All Time', 1, 2)
                            combined = combined.sort_values(by=[sys_col_found, 'SortKey']).drop('SortKey', axis=1)

                            html_table = """<style>.builder-table { border-collapse: collapse; width: 100%; font-size: 14px; font-family: sans-serif; margin-top: 15px; } .builder-table th, .builder-table td { border: 1px solid #ccc; padding: 6px; text-align: center; } .builder-table tr:hover { background-color: #0000FF !important; color: white !important; } .left-align { text-align: left !important; padding-left: 8px !important; }</style><div style="overflow-x: auto; width: 100%;"><table class="builder-table" style="min-width: 1000px;"><thead><tr style="background-color: #1a3a5f; color: white;"><th class="left-align">System Name</th><th class="left-align">Period</th><th>Bets</th><th>Wins</th><th>Win P/L</th><th>Win SR</th><th>Places</th><th>Plc P/L</th><th>Plc SR</th><th>Total P/L</th></tr></thead><tbody>"""
                            
                            unique_sys = combined[sys_col_found].unique()
                            palette = ["#e8f4f8", "#f8e8e8", "#e8f8e8", "#f8f4e8", "#f4e8f8", "#e8f8f8"]
                            bg_colors = {sys: palette[i % len(palette)] for i, sys in enumerate(unique_sys)}

                            last_sys = None
                            for _, row in combined.iterrows():
                                if last_sys is not None and last_sys != row[sys_col_found]:
                                    html_table += '<tr><td colspan="10" style="border: none !important; background-color: white !important; height: 15px; padding: 0 !important;"></td></tr>'
                                last_sys = row[sys_col_found]
                                bg = bg_colors[row[sys_col_found]]
                                b_s = "<b>" if row['Period'] == 'All Time' else ""
                                b_e = "</b>" if row['Period'] == 'All Time' else ""
                                w_color = "#2e7d32" if row['Win_Profit'] > 0 else "#d32f2f" if row['Win_Profit'] < 0 else "black"
                                p_color = "#2e7d32" if row['Place_Profit'] > 0 else "#d32f2f" if row['Place_Profit'] < 0 else "black"
                                t_color = "#2e7d32" if row['Total P/L'] > 0 else "#d32f2f" if row['Total P/L'] < 0 else "black"
                                
                                html_table += f"""<tr style="background-color: {bg};"><td class="left-align"><b>{row[sys_col_found]}</b></td><td class="left-align">{b_s}{row['Period']}{b_e}</td><td>{row['Bets']}</td><td>{row['Wins']}</td><td style="color:{w_color}; font-weight:bold;">£{row['Win_Profit']:.2f}</td><td>{row['Strike Rate (%)']:.2f}%</td><td>{row['Places']}</td><td style="color:{p_color}; font-weight:bold;">£{row['Place_Profit']:.2f}</td><td>{row['Place SR (%)']:.2f}%</td><td style="color:{t_color}; font-weight:bold;">£{row['Total P/L']:.2f}</td></tr>"""
                            html_table += "</tbody></table></div>"
                            st.markdown(html_table, unsafe_allow_html=True)
                        else: st.warning("Found the file, but none of the picks had a matched race result in the database.")
                    else: st.error("The file is missing one of the required columns: Date, Time, Course, or Horse.")
                except Exception as e: st.error(f"Error processing {target_ods}: {e}")
            else: 
                if st.session_state.get("is_admin") and "Admin" in perf_file_choice:
                    st.info("To see Admin performance tracking, please upload 'K2AdminMaster.ods' to the root folder.")
                else:
                    st.info("To see live performance tracking, please upload 'K2SystemsMaster.ods' to the root folder.")

    # =========================================================================
    # 🛠️ PAGE 4: SYSTEM BUILDER
    # =========================================================================
    elif page == "🛠️ System Builder":
        if "form_reset_counter" not in st.session_state:
            st.session_state.form_reset_counter = 0

        c_title, c_reset = st.columns([4, 1])
        with c_title:
            st.header("🛠️ Mini System Builder")
        with c_reset:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔄 Reset Filters", use_container_width=True):
                if 'tab4_results' in st.session_state:
                    del st.session_state['tab4_results']
                st.session_state['force_reset'] = True
                st.session_state.form_reset_counter += 1
                st.rerun()

        if df_all_prepped is not None and not df_all_prepped.empty:
            b_df = df_all_prepped

            rt_opts = b_df['Race Type'].dropna().unique().tolist()
            hcap_opts = b_df['H/Cap'].dropna().unique().tolist()
            rnr_opts = ["2-7", "8-12", "13-16", ">16"]
            class_opts = sorted([int(x) for x in b_df['Class'].dropna().unique() if str(x).isdigit() or isinstance(x, (int, float))]) if 'Class' in b_df.columns else []
            cm_opts = [x for x in b_df['Class Move'].dropna().unique() if x in ['U', 'D', 'S']] if 'Class Move' in b_df.columns else []
            sex_opts = ["c", "f", "g", "m", "h", "r", "x"]
            course_opts = sorted([str(x).strip() for x in b_df['Course'].dropna().unique() if str(x).strip()])
            
            all_months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            ai_rank_opts = ["Any", "AI Rank 1 Only", "Top 2 Only", "NOT AI Rank 1", "NOT Top 3"]
            value_opts = ["Off", "AI Value vs 7:30AM", "AI Value vs BSP", "My Value vs 7:30AM", "My Value vs BSP", "NOT AI Value vs 7:30AM", "NOT AI Value vs BSP", "NOT My Value vs 7:30AM", "NOT My Value vs BSP"]
            irish_opts = ["Any", "Y (Yes)", "No (Blank)"]
            rank_opts = ["Any", "Rank 1", "Top 2", "Top 3", "Top 4", "Top 5", "Not Top 3"]
            
            defaults = {
                'ui_months': all_months, 'ui_courses': [], 'ui_race_types': rt_opts, 'ui_hcap_types': hcap_opts,
                'ui_price_min': 0.0, 'ui_price_max': 1000.0, 'ui_prob_gap': -100.0, 'ui_min_edge': -100.0, 'ui_bracket_1': 0.0, 'ui_bracket_2': 10.0,
                'ui_rnrs': rnr_opts, 'ui_classes': class_opts, 'ui_cm': cm_opts,
                'ui_ai_rank_filter': "Any", 'ui_sex': sex_opts, 'ui_value_filter': "Off",
                'ui_irish_f': "Any", 'ui_age_range': (1, 20),
                'ui_comb_f': "Any", 'ui_comp_f': "Any", 'ui_speed_f': "Any", 'ui_race_f': "Any", 'ui_primary_f': "Any",
                'ui_msai_f': "Any", 'ui_prb_f': "Any", 'ui_tprb_f': "Any", 'ui_jprb_f': "Any", 'ui_form_f': "Any", 'ui_pure_f': "Any",
                'ui_group_cols': ['Edge Bracket', 'Race Type', 'H/Cap']
            }
            
            if st.session_state.get('force_reset', False):
                for k, v in defaults.items(): st.session_state[k] = v
                st.session_state['force_reset'] = False
            else:
                for k, v in defaults.items():
                    if k not in st.session_state: st.session_state[k] = v

            def load_sys_to_ui(sys_data):
                st.session_state['ui_months'] = sys_data.get('months', all_months)
                st.session_state['ui_courses'] = [c for c in sys_data.get('courses', []) if c in course_opts]
                st.session_state['ui_race_types'] = [r for r in sys_data.get('race_types', []) if r in rt_opts]
                st.session_state['ui_hcap_types'] = [h for h in sys_data.get('hcap_types', []) if h in hcap_opts]
                st.session_state['ui_price_min'] = float(sys_data.get('price_min', 0.0))
                st.session_state['ui_price_max'] = float(sys_data.get('price_max', 1000.0))
                st.session_state['ui_prob_gap'] = float(sys_data.get('min_prob_gap', -1.0)) * 100
                st.session_state['ui_min_edge'] = float(sys_data.get('min_edge_perc', -100.0))
                st.session_state['ui_bracket_1'] = float(sys_data.get('bracket_1', 0.0))
                st.session_state['ui_bracket_2'] = float(sys_data.get('bracket_2', 10.0))
                st.session_state['ui_rnrs'] = sys_data.get('rnrs', rnr_opts)
                st.session_state['ui_classes'] = [c for c in sys_data.get('classes', []) if c in class_opts]
                st.session_state['ui_cm'] = [c for c in sys_data.get('cm', []) if c in cm_opts]
                
                ai_filt = sys_data.get('ai_rank_filter', "Any")
                if ai_filt == "Any" and sys_data.get('rank_1_only', False): ai_filt = "AI Rank 1 Only"
                st.session_state['ui_ai_rank_filter'] = ai_filt
                
                st.session_state['ui_sex'] = [s for s in sys_data.get('sex', []) if s in sex_opts]
                st.session_state['ui_value_filter'] = sys_data.get('value_filter', "Off")
                st.session_state['ui_irish_f'] = sys_data.get('irish', "Any")
                st.session_state['ui_age_range'] = (int(sys_data.get('age_min', 1)), int(sys_data.get('age_max', 20)))
                
                ranks = sys_data.get('ranks', {})
                st.session_state['ui_comb_f'] = ranks.get('Comb. Rank', 'Any')
                st.session_state['ui_comp_f'] = ranks.get('Comp. Rank', 'Any')
                st.session_state['ui_speed_f'] = ranks.get('Speed Rank', 'Any')
                st.session_state['ui_race_f'] = ranks.get('Race Rank', 'Any')
                st.session_state['ui_primary_f'] = ranks.get('Primary Rank', 'Any')
                st.session_state['ui_msai_f'] = ranks.get('MSAI Rank', 'Any')
                st.session_state['ui_prb_f'] = ranks.get('PRB Rank', 'Any')
                st.session_state['ui_tprb_f'] = ranks.get('Trainer PRB Rank', 'Any')
                st.session_state['ui_jprb_f'] = ranks.get('Jockey PRB Rank', 'Any')
                st.session_state['ui_form_f'] = ranks.get('Form Rank', 'Any')
                st.session_state['ui_pure_f'] = ranks.get('Pure Rank', 'Any')
                
                st.session_state.form_reset_counter += 1

            def get_ai_idx():
                val = st.session_state.get('ui_ai_rank_filter', 'Any')
                return ai_rank_opts.index(val) if val in ai_rank_opts else 0
            
            def get_val_idx():
                val = st.session_state.get('ui_value_filter', 'Off')
                return value_opts.index(val) if val in value_opts else 0
                
            def get_irish_idx():
                val = st.session_state.get('ui_irish_f', 'Any')
                return irish_opts.index(val) if val in irish_opts else 0
                
            def get_r_idx(key_name):
                val = st.session_state.get(key_name, 'Any')
                return rank_opts.index(val) if val in rank_opts else 0

            with st.form(f"builder_form_{st.session_state.form_reset_counter}"):
                st.markdown("### Core Filters")
                
                d_col, m_col = st.columns([1, 3])
                with d_col:
                    if 'Date_DT' in b_df.columns and not b_df['Date_DT'].dropna().empty:
                        min_d = b_df['Date_DT'].min().date()
                        max_d = b_df['Date_DT'].max().date()
                    else:
                        min_d = datetime(2024, 1, 1).date()
                        max_d = datetime.now().date()
                    date_range = st.date_input("Test Specific Period (From - To)", [min_d, max_d], min_value=min_d, max_value=max_d)
                
                with m_col:
                    selected_months = st.multiselect("Include Specific Months (Seasonal Filter)", all_months, default=st.session_state.get('ui_months', all_months))
                
                st.markdown("---")
                selected_courses = st.multiselect("🎯 Specific Course(s) [Leave blank to include ALL courses]", course_opts, default=st.session_state.get('ui_courses', []))
                st.markdown("<br>", unsafe_allow_html=True)

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    selected_race_types = st.multiselect("Race Type", rt_opts, default=st.session_state.get('ui_race_types', rt_opts))
                    selected_hcap = st.multiselect("Handicap Status", hcap_opts, default=st.session_state.get('ui_hcap_types', hcap_opts))
                with c2:
                    p_col1, p_col2 = st.columns(2)
                    with p_col1: price_min = st.number_input("Min Price (7:30AM)", 0.0, 1000.0, value=float(st.session_state.get('ui_price_min', 0.0)), step=0.5)
                    with p_col2: price_max = st.number_input("Max Price (7:30AM)", 0.0, 1000.0, value=float(st.session_state.get('ui_price_max', 1000.0)), step=0.5)
                    
                    e_col1, e_col2 = st.columns(2)
                    with e_col1: min_prob_gap = st.number_input("Min Prob Gap (%)", -100.0, 50.0, value=float(st.session_state.get('ui_prob_gap', -100.0)), step=0.5) / 100
                    with e_col2: min_edge_perc = st.number_input("Min Value Edge %", -100.0, 200.0, value=float(st.session_state.get('ui_min_edge', -100.0)), step=5.0)
                with c3:
                    selected_rnrs = st.multiselect("No. of Runners", rnr_opts, default=st.session_state.get('ui_rnrs', rnr_opts))
                    if class_opts:
                        selected_classes = st.multiselect("Class (1-6)", class_opts, default=st.session_state.get('ui_classes', class_opts))
                    else:
                        st.multiselect("Class (1-6)", ["Not Found in CSV"], disabled=True)
                        selected_classes = []
                with c4:
                    if cm_opts:
                        selected_cm = st.multiselect("Class Movement", cm_opts, default=st.session_state.get('ui_cm', cm_opts))
                    else:
                        st.multiselect("Class Movement", ["Not Found in CSV"], disabled=True)
                        selected_cm = []

                c5, c6, c7, c8 = st.columns(4)
                with c5:
                    ai_rank_filter = st.selectbox("AI Rank", ai_rank_opts, index=get_ai_idx())
                    selected_sex = st.multiselect("Horse Sex", sex_opts, default=st.session_state.get('ui_sex', sex_opts))
                with c6: 
                    value_filter = st.selectbox("Value Filter", value_opts, index=get_val_idx())
                with c7: 
                    irish_f = st.selectbox("Irish Race", irish_opts, index=get_irish_idx())
                with c8: 
                    age_min, age_max = st.slider("Horse Age Range", 1, 20, value=st.session_state.get('ui_age_range', (1, 20)))
                
                with st.expander("📊 Advanced Rank Filters", expanded=False):
                    r1_c1, r1_c2, r1_c3, r1_c4, r1_c5 = st.columns(5)
                    with r1_c1: comb_f = st.selectbox("Comb. Rank", rank_opts, index=get_r_idx('ui_comb_f'))
                    with r1_c2: comp_f = st.selectbox("Comp. Rank", rank_opts, index=get_r_idx('ui_comp_f'))
                    with r1_c3: speed_f = st.selectbox("Speed Rank", rank_opts, index=get_r_idx('ui_speed_f'))
                    with r1_c4: race_f = st.selectbox("Race Rank", rank_opts, index=get_r_idx('ui_race_f'))
                    with r1_c5: primary_f = st.selectbox("Primary Rank", rank_opts, index=get_r_idx('ui_primary_f'))
                    
                    r2_c1, r2_c2, r2_c3, r2_c4, r2_c5 = st.columns(5)
                    with r2_c1: msai_f = st.selectbox("MSAI Rank", rank_opts, index=get_r_idx('ui_msai_f'))
                    with r2_c2: prb_f = st.selectbox("PRB Rank", rank_opts, index=get_r_idx('ui_prb_f'))
                    with r2_c3: tprb_f = st.selectbox("Trainer PRB Rank", rank_opts, index=get_r_idx('ui_tprb_f'))
                    with r2_c4: jprb_f = st.selectbox("Jockey PRB Rank", rank_opts, index=get_r_idx('ui_jprb_f'))
                    with r2_c5: form_f = st.selectbox("Form Rank", rank_opts, index=get_r_idx('ui_form_f'))
                    
                    r3_c1, r3_c2, r3_c3, r3_c4, r3_c5 = st.columns(5)
                    with r3_c1: pure_f = st.selectbox("Pure Rank", rank_opts, index=get_r_idx('ui_pure_f'))

                with st.expander("🩻 Custom Value X-Ray Brackets", expanded=False):
                    v_c1, v_c2 = st.columns(2)
                    with v_c1: bracket_1 = st.slider("Bracket 1 Threshold (%)", -20.0, 20.0, value=float(st.session_state.get('ui_bracket_1', 0.0)), step=1.0)
                    with v_c2: bracket_2 = st.slider("Bracket 2 Threshold (%)", 0.0, 50.0, value=float(st.session_state.get('ui_bracket_2', 10.0)), step=1.0)
                    st.info(f"Dynamically splits the table into: '< {bracket_1}%', '{bracket_1}% to {bracket_2}%', and '> {bracket_2}%'.")
                
                st.markdown("---")
                st.markdown("### 📊 Dynamic Table Grouping")
                groupable_cols = ['Edge Bracket', 'Race Type', 'H/Cap', 'Price Bracket', 'Course', 'No. of Rnrs', 'Class', 'Class Move', 'Sex', 'Age', 'Comb. Rank', 'Comp. Rank', 'Speed Rank', 'Race Rank', 'Primary Rank', 'MSAI Rank', 'PRB Rank', 'Trainer PRB Rank', 'Jockey PRB Rank', 'Form Rank', 'Pure Rank', 'Irish?']
                groupable_cols = [c for c in groupable_cols if c in b_df.columns]
                
                groupable_cols.insert(0, 'Month/Year')
                
                ui_group_cols = st.multiselect(
                    "Select up to 3 factors to group the breakdown table by:", 
                    options=groupable_cols, 
                    default=st.session_state.get('ui_group_cols', ['Edge Bracket', 'Race Type', 'H/Cap']),
                    max_selections=3
                )
                
                submit_button = st.form_submit_button(label="🚀 Process Data")

            if st.session_state.get("is_admin"):
                st.markdown("---")
                st.markdown("### ⚙️ Admin Tools")
                
                c_vault1, c_vault2 = st.columns([3, 1])
                with c_vault1:
                    st.markdown("#### 🗄️ Update Prediction Vault")
                    st.markdown("<span style='font-size:12px;'>Append today's live predictions to the master history file.</span>", unsafe_allow_html=True)
                with c_vault2:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("Append Today to Vault 💾", use_container_width=True, type="primary"):
                        try:
                            if df_today is not None and not df_today.empty:
                                vault_file = "K2_Prediction_Vault.csv"
                                if os.path.exists(vault_file):
                                    v_df = pd.read_csv(vault_file)
                                    today_export = df_today[['Date_Key', 'Time', 'Course', 'Horse', 'ML_Prob', 'Shadow_Prob']].copy()
                                    today_export['Date_Key'] = today_export['Date_Key'].astype(str)
                                    today_export['Time'] = today_export['Time'].astype(str).str.split('.').str[0].str.strip()
                                    today_export['Course'] = today_export['Course'].astype(str).str.strip().str.title()
                                    today_export['Horse'] = today_export['Horse'].astype(str).str.strip().str.title()
                                    
                                    for col in ['ML_Prob', 'Shadow_Prob']:
                                        today_export[col] = pd.to_numeric(today_export[col], errors='coerce').round(6)
                                        
                                    updated_vault = pd.concat([v_df, today_export], ignore_index=True)
                                    updated_vault = updated_vault.drop_duplicates(subset=['Date_Key', 'Time', 'Course', 'Horse'], keep='last')
                                    updated_vault.to_csv(vault_file, index=False)
                                    st.success(f"✅ Vault Updated! Added {len(today_export)} new predictions.")
                                else:
                                    st.error("⚠️ Vault file not found. Please download the Master Vault from the sidebar and upload it to GitHub first.")
                            else:
                                st.warning("No daily predictions found to append.")
                        except Exception as e:
                            st.error(f"Error updating vault: {e}")

                st.markdown("---")
                st.markdown("#### 💻 Generate System Code (For GitHub)")
                c_name, c_btn = st.columns([3, 1])
                with c_name: new_sys_name = st.text_input("System Name:", placeholder="e.g., A/W MSAI Value")
                with c_btn:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("Generate JSON Code", use_container_width=True):
                        if new_sys_name:
                            sys_data = {
                                "race_types": selected_race_types, "hcap_types": selected_hcap, "price_min": price_min, "price_max": price_max, "min_prob_gap": min_prob_gap, "min_edge_perc": min_edge_perc, "bracket_1": bracket_1, "bracket_2": bracket_2, "rnrs": selected_rnrs, "classes": selected_classes, "cm": selected_cm, "sex": selected_sex, "courses": selected_courses, "ai_rank_filter": ai_rank_filter, "value_filter": value_filter, "irish": irish_f, "age_min": age_min, "age_max": age_max, "months": selected_months,
                                "ranks": {"Comb. Rank": comb_f, "Comp. Rank": comp_f, "Speed Rank": speed_f, "Race Rank": race_f, "Primary Rank": primary_f, "MSAI Rank": msai_f, "PRB Rank": prb_f, "Trainer PRB Rank": tprb_f, "Jockey PRB Rank": jprb_f, "Form Rank": form_f, "Pure Rank": pure_f}
                            }
                            st.code(f'"{new_sys_name}": {json.dumps(sys_data, indent=4)}', language="json")
                        else: st.error("Please enter a name for the system to generate code.")
                c_pub, c_sec = st.columns(2)
                with c_pub:
                    if os.path.exists("K2_user_systems.json"):
                        with st.expander("📖 View Active PUBLIC Systems"):
                            try:
                                with open("K2_user_systems.json", "r") as f: saved_dict = json.load(f)
                                if saved_dict:
                                    for s_key, s_data in list(saved_dict.items()):
                                        with st.popover(f"🔍 {s_key}", use_container_width=True):
                                            st.json(s_data)
                                            if st.button("Load 📥", key=f"load_pub_{s_key}", use_container_width=True):
                                                load_sys_to_ui(s_data)
                                                st.rerun()
                                else: st.write("No systems currently active.")
                            except Exception as e: st.error(f"Error reading public file: {e}")
                
                with c_sec:
                    if os.path.exists("K2_admin_systems.json"):
                        with st.expander("🕵️ View Active ADMIN SECRETS"):
                            try:
                                with open("K2_admin_systems.json", "r") as f: admin_dict = json.load(f)
                                if admin_dict:
                                    for s_key, s_data in list(admin_dict.items()):
                                        with st.popover(f"🔍 {s_key}", use_container_width=True):
                                            st.json(s_data)
                                            if st.button("Load 📥", key=f"load_sec_{s_key}", use_container_width=True):
                                                load_sys_to_ui(s_data)
                                                st.rerun()
                                else: st.write("No admin systems currently active.")
                            except Exception as e: st.error(f"Error reading admin file: {e}")
                st.markdown("---")

            if submit_button:
                st.session_state['ui_group_cols'] = ui_group_cols
                st.success("✅ System recalculated instantly!")

                mask = (b_df['Race Type'].isin(selected_race_types) & b_df['H/Cap'].isin(selected_hcap) & (b_df['7:30AM Price'] >= price_min) & (b_df['7:30AM Price'] <= price_max) & (b_df['Prob Gap'] >= min_prob_gap) & (b_df['Value_Edge_Perc'] >= min_edge_perc))
                
                if len(date_range) == 2: mask = mask & (b_df['Date_DT'].dt.date >= date_range[0]) & (b_df['Date_DT'].dt.date <= date_range[1])
                elif len(date_range) == 1: mask = mask & (b_df['Date_DT'].dt.date == date_range[0])
                
                if len(selected_months) < 12:
                    month_map = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6, "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}
                    sel_m_nums = [month_map[m] for m in selected_months]
                    mask = mask & b_df['Date_DT'].dt.month.isin(sel_m_nums)
                
                if ai_rank_filter == "AI Rank 1 Only": mask = mask & (b_df['Rank'] == 1)
                elif ai_rank_filter == "Top 2 Only": mask = mask & (b_df['Rank'] <= 2)
                elif ai_rank_filter == "NOT AI Rank 1": mask = mask & (b_df['Rank'] > 1)
                elif ai_rank_filter == "NOT Top 3": mask = mask & (b_df['Rank'] > 3)
                
                if value_filter in ["Value vs 7:30AM Price", "AI Value vs 7:30AM"]: mask = mask & (b_df['7:30AM Price'] > b_df['Value Price'])
                elif value_filter in ["Value vs BSP", "AI Value vs BSP"]: mask = mask & (b_df['BSP'] > b_df['Value Price'])
                elif value_filter == "My Value vs 7:30AM": mask = mask & (b_df['7:30AM Price'] > b_df['User Value'])
                elif value_filter == "My Value vs BSP": mask = mask & (b_df['BSP'] > b_df['User Value'])
                elif value_filter == "NOT AI Value vs 7:30AM": mask = mask & (b_df['7:30AM Price'] < b_df['Value Price'])
                elif value_filter == "NOT AI Value vs BSP": mask = mask & (b_df['BSP'] < b_df['Value Price'])
                elif value_filter == "NOT My Value vs 7:30AM": mask = mask & (b_df['7:30AM Price'] < b_df['User Value'])
                elif value_filter == "NOT My Value vs BSP": mask = mask & (b_df['BSP'] < b_df['User Value'])
                
                rnr_mask = pd.Series(False, index=b_df.index)
                if "2-7" in selected_rnrs: rnr_mask |= (b_df['No. of Rnrs'] >= 2) & (b_df['No. of Rnrs'] <= 7)
                if "8-12" in selected_rnrs: rnr_mask |= (b_df['No. of Rnrs'] >= 8) & (b_df['No. of Rnrs'] <= 12)
                if "13-16" in selected_rnrs: rnr_mask |= (b_df['No. of Rnrs'] >= 13) & (b_df['No. of Rnrs'] <= 16)
                if ">16" in selected_rnrs: rnr_mask |= (b_df['No. of Rnrs'] > 16)
                mask = mask & rnr_mask

                if 'Class' in b_df.columns and selected_classes: mask = mask & b_df['Class'].isin(selected_classes)
                if 'Class Move' in b_df.columns and selected_cm: mask = mask & b_df['Class Move'].isin(selected_cm)
                if 'Age' in b_df.columns: mask = mask & (b_df['Age'] >= age_min) & (b_df['Age'] <= age_max)
                if 'Sex' in b_df.columns and selected_sex: mask = mask & b_df['Sex'].astype(str).str.strip().str.lower().isin([s.lower() for s in selected_sex])
                if 'Course' in b_df.columns and selected_courses: mask = mask & b_df['Course'].astype(str).str.strip().isin(selected_courses)

                t_irish_col = 'Irish?' if 'Irish?' in b_df.columns else 'Irish' if 'Irish' in b_df.columns else None
                if t_irish_col and irish_f != "Any":
                    t_irish_series = b_df[t_irish_col].astype(str).str.strip().str.upper()
                    if irish_f == "Y (Yes)": mask = mask & (t_irish_series == 'Y')
                    elif irish_f == "No (Blank)": mask = mask & (t_irish_series != 'Y')

                def apply_rank_filter(df_mask, current_df, col_name, setting):
                    if setting != "Any" and col_name in current_df.columns:
                        num_col = pd.to_numeric(current_df[col_name], errors='coerce')
                        if setting == "Rank 1": return df_mask & (num_col == 1)
                        elif setting == "Top 2": return df_mask & (num_col <= 2)
                        elif setting == "Top 3": return df_mask & (num_col <= 3)
                        elif setting == "Top 4": return df_mask & (num_col <= 4)
                        elif setting == "Top 5": return df_mask & (num_col <= 5)
                        elif setting == "Not Top 3": return df_mask & ((num_col > 3) | (num_col == 0))
                    return df_mask

                mask = apply_rank_filter(mask, b_df, 'Comb. Rank', comb_f)
                mask = apply_rank_filter(mask, b_df, 'Comp. Rank', comp_f)
                mask = apply_rank_filter(mask, b_df, 'Speed Rank', speed_f)
                mask = apply_rank_filter(mask, b_df, 'Race Rank', race_f)
                mask = apply_rank_filter(mask, b_df, 'Primary Rank', primary_f)
                mask = apply_rank_filter(mask, b_df, 'MSAI Rank', msai_f)
                mask = apply_rank_filter(mask, b_df, 'PRB Rank', prb_f)
                mask = apply_rank_filter(mask, b_df, 'Trainer PRB Rank', tprb_f)
                mask = apply_rank_filter(mask, b_df, 'Jockey PRB Rank', jprb_f)
                mask = apply_rank_filter(mask, b_df, 'Form Rank', form_f)
                mask = apply_rank_filter(mask, b_df, 'Pure Rank', pure_f)
                
                df_filtered = b_df[mask].copy()

                if not df_filtered.empty:
                    # --- NEW: DYNAMIC EDGE BRACKETS ---
                    safe_b2 = max(bracket_1 + 0.1, float(bracket_2))
                    v_bins = [-np.inf, bracket_1, safe_b2, np.inf]
                    v_labels = [f'1. Negative Edge (< {bracket_1}%)', f'2. Fair Value ({bracket_1}% to {safe_b2:.1f}%)', f'3. Deep Value (> {safe_b2:.1f}%)']
                    df_filtered['Edge Bracket'] = pd.cut(df_filtered['Value_Edge_Perc'], bins=v_bins, labels=v_labels)
                    df_filtered['Edge Bracket'] = df_filtered['Edge Bracket'].cat.add_categories('Unknown').fillna('Unknown')
                    
                    if 'Date_DT' in df_filtered.columns:
                        df_filtered['Month/Year'] = df_filtered['Date_DT'].dt.strftime('%Y-%m (%b)')
                    else:
                        df_filtered['Month/Year'] = "Unknown"

                    actual_grp_cols = [c for c in ui_group_cols if c in df_filtered.columns]
                    if not actual_grp_cols: actual_grp_cols = ['Race Type']

                    breakdown = df_filtered.groupby(actual_grp_cols, observed=False).agg(
                        Bets=('Horse', 'count'), Wins=('Is_Win', 'sum'), Win_Profit=('Win P/L <2%', 'sum'), Places=('Is_Place', 'sum'), Place_Profit=('Place P/L <2%', 'sum')
                    ).reset_index()
                    
                    breakdown = breakdown[breakdown['Bets'] > 0]
                    breakdown['Strike Rate (%)'] = (breakdown['Wins'] / breakdown['Bets'] * 100).fillna(0)
                    breakdown['Place SR (%)'] = (breakdown['Places'] / breakdown['Bets'] * 100).fillna(0)
                    breakdown['Win ROI (%)'] = (breakdown['Win_Profit'] / breakdown['Bets'] * 100).fillna(0)
                    breakdown['Total P/L'] = breakdown['Win_Profit'] + breakdown['Place_Profit']
                    breakdown = breakdown.sort_values(by=actual_grp_cols)

                    total_bets_for_roi = breakdown['Bets'].sum()
                    total_pl_for_roi = breakdown['Total P/L'].sum()
                    total_roi_perc = (total_pl_for_roi / total_bets_for_roi * 100) if total_bets_for_roi > 0 else 0.0
                    
                    # --- ADVANCED METRICS CALCULATION ---
                    df_chrono = df_filtered.sort_values(by=['Date_DT', 'Time'])
                    
                    df_chrono['Is_Loss'] = (df_chrono['Win P/L <2%'] < 0).astype(int)
                    loss_blocks = df_chrono['Is_Loss'] * (df_chrono['Is_Loss'].groupby((df_chrono['Is_Loss'] != df_chrono['Is_Loss'].shift()).cumsum()).cumcount() + 1)
                    llr = loss_blocks.max() if not loss_blocks.empty else 0

                    df_chrono['Staked_PL'] = np.where(df_chrono['Win P/L <2%'] > 0, df_chrono['Win P/L <2%'], -1)
                    df_chrono['Bankroll'] = df_chrono['Staked_PL'].cumsum()
                    df_chrono['Peak'] = df_chrono['Bankroll'].cummax()
                    df_chrono['Drawdown'] = df_chrono['Bankroll'] - df_chrono['Peak']
                    max_dd = abs(df_chrono['Drawdown'].min()) if not df_chrono['Drawdown'].empty else 0.0

                    market_prices = np.where(df_chrono['BSP'] > 0, df_chrono['BSP'], df_chrono['7:30AM Price'])
                    valid_prices = market_prices[market_prices > 1.0]
                    expected_wins = np.sum(1.0 / valid_prices) if len(valid_prices) > 0 else 0.0
                    actual_wins = breakdown['Wins'].sum()
                    
                    a_e = actual_wins / expected_wins if expected_wins > 0 else 0.0
                    
                    expected_losses = total_bets_for_roi - expected_wins
                    actual_losses = total_bets_for_roi - actual_wins
                    
                    if expected_wins > 0 and expected_losses > 0:
                        chi_score = ((actual_wins - expected_wins)**2 / expected_wins) + ((actual_losses - expected_losses)**2 / expected_losses)
                    else:
                        chi_score = 0.0

                    returns = df_chrono['Staked_PL']
                    mean_return = returns.mean() if not returns.empty else 0.0
                    
                    downside_sq = np.where(returns < 0, returns**2, 0)
                    downside_dev = np.sqrt(np.mean(downside_sq)) if len(returns) > 0 else 0.0
                    
                    sortino = (mean_return / downside_dev) if downside_dev > 0 else (99.99 if mean_return > 0 else 0.0)
                    ulcer_index = np.sqrt(np.mean(df_chrono['Drawdown']**2)) if not df_chrono['Drawdown'].empty else 0.0

                    win_sr = (breakdown['Wins'].sum() / total_bets_for_roi * 100) if total_bets_for_roi > 0 else 0.0
                    plc_sr = (breakdown['Places'].sum() / total_bets_for_roi * 100) if total_bets_for_roi > 0 else 0.0

                    kpis = [
                        total_bets_for_roi, breakdown['Wins'].sum(), breakdown['Places'].sum(), 
                        breakdown['Win_Profit'].sum(), breakdown['Place_Profit'].sum(),
                        total_roi_perc
                    ]
                    
                    advanced_metrics = {
                        'win_sr': win_sr, 'plc_sr': plc_sr, 'llr': llr, 'max_dd': max_dd, 
                        'a_e': a_e, 'chi': chi_score, 'sortino': sortino, 'ulcer': ulcer_index
                    }

                    qual_html_out, csv_data_out, timestamp_out = "", None, ""
                    
                    if not df_filtered.empty:
                        historic_csv_out = clean_csv_df(df_filtered).to_csv(index=False).encode('utf-8') 
                    else:
                        historic_csv_out = None
                    
                    val_bsp_warning = value_filter in ["Value vs BSP", "AI Value vs BSP", "My Value vs BSP", "NOT AI Value vs BSP", "NOT My Value vs BSP"]

                    if df_today is not None and not df_today.empty:
                        t_df = df_today
                        
                        t_mask = (t_df['Race Type'].isin(selected_race_types) & t_df['H/Cap'].isin(selected_hcap) & (t_df['7:30AM Price'] >= price_min) & (t_df['7:30AM Price'] <= price_max) & (t_df['Prob Gap'] >= min_prob_gap) & (t_df['Value_Edge_Perc'] >= min_edge_perc))
                        
                        if len(selected_months) < 12:
                            t_mask = t_mask & t_df['Date_DT'].dt.month.isin(sel_m_nums)

                        if ai_rank_filter == "AI Rank 1 Only": t_mask = t_mask & (t_df['Rank'] == 1)
                        elif ai_rank_filter == "Top 2 Only": t_mask = t_mask & (t_df['Rank'] <= 2)
                        elif ai_rank_filter == "NOT AI Rank 1": t_mask = t_mask & (t_df['Rank'] > 1)
                        elif ai_rank_filter == "NOT Top 3": t_mask = t_mask & (t_df['Rank'] > 3)
                        
                        if value_filter in ["Value vs 7:30AM Price", "Value vs BSP", "AI Value vs 7:30AM", "AI Value vs BSP"]: 
                            t_mask = t_mask & (t_df['7:30AM Price'] > t_df['Value Price'])
                        elif value_filter in ["My Value vs 7:30AM", "My Value vs BSP"]: 
                            t_mask = t_mask & (t_df['7:30AM Price'] > t_df['User Value'])
                        elif value_filter in ["NOT AI Value vs 7:30AM", "NOT AI Value vs BSP"]:
                            t_mask = t_mask & (t_df['7:30AM Price'] < t_df['Value Price'])
                        elif value_filter in ["NOT My Value vs 7:30AM", "NOT My Value vs BSP"]:
                            t_mask = t_mask & (t_df['7:30AM Price'] < t_df['User Value'])
                        
                        t_rnr_mask = pd.Series(False, index=t_df.index)
                        if "2-7" in selected_rnrs: t_rnr_mask |= (t_df['No. of Rnrs'] >= 2) & (t_df['No. of Rnrs'] <= 7)
                        if "8-12" in selected_rnrs: t_rnr_mask |= (t_df['No. of Rnrs'] >= 8) & (t_df['No. of Rnrs'] <= 12)
                        if "13-16" in selected_rnrs: t_rnr_mask |= (t_df['No. of Rnrs'] >= 13) & (t_df['No. of Rnrs'] <= 16)
                        if ">16" in selected_rnrs: t_rnr_mask |= (t_df['No. of Rnrs'] > 16)
                        t_mask = t_mask & t_rnr_mask

                        if 'Class' in t_df.columns and selected_classes: t_mask = t_mask & t_df['Class'].isin(selected_classes)
                        if 'Class Move' in t_df.columns and selected_cm: t_mask = t_mask & t_df['Class Move'].isin(selected_cm)
                        if 'Age' in t_df.columns: t_mask = t_mask & (t_df['Age'] >= age_min) & (t_df['Age'] <= age_max)
                        if 'Sex' in t_df.columns and selected_sex: t_mask = t_mask & t_df['Sex'].astype(str).str.strip().str.lower().isin([s.lower() for s in selected_sex])
                        if 'Course' in t_df.columns and selected_courses: t_mask = t_mask & t_df['Course'].astype(str).str.strip().isin(selected_courses)

                        if t_irish_col and irish_f != "Any":
                            t_irish_series = t_df[t_irish_col].astype(str).str.strip().str.upper()
                            if irish_f == "Y (Yes)": t_mask = t_mask & (t_irish_series == 'Y')
                            elif irish_f == "No (Blank)": t_mask = t_mask & (t_irish_series != 'Y')

                        t_mask = apply_rank_filter(t_mask, t_df, 'Comb. Rank', comb_f)
                        t_mask = apply_rank_filter(t_mask, t_df, 'Comp. Rank', comp_f)
                        t_mask = apply_rank_filter(t_mask, t_df, 'Speed Rank', speed_f)
                        t_mask = apply_rank_filter(t_mask, t_df, 'Race Rank', race_f)
                        t_mask = apply_rank_filter(t_mask, t_df, 'Primary Rank', primary_f)
                        t_mask = apply_rank_filter(t_mask, t_df, 'MSAI Rank', msai_f)
                        t_mask = apply_rank_filter(t_mask, t_df, 'PRB Rank', prb_f)
                        t_mask = apply_rank_filter(t_mask, t_df, 'Trainer PRB Rank', tprb_f)
                        t_mask = apply_rank_filter(t_mask, t_df, 'Jockey PRB Rank', jprb_f)
                        t_mask = apply_rank_filter(t_mask, t_df, 'Form Rank', form_f)
                        t_mask = apply_rank_filter(t_mask, t_df, 'Pure Rank', pure_f)
                        
                        t_filtered = t_df[t_mask].copy()
                        
                        if not t_filtered.empty:
                            # --- NEW: DYNAMIC EDGE BRACKETS (TODAY'S QUALIFIERS) ---
                            t_filtered['Edge Bracket'] = pd.cut(t_filtered['Value_Edge_Perc'], bins=v_bins, labels=v_labels)
                            t_filtered['Edge Bracket'] = t_filtered['Edge Bracket'].cat.add_categories('Unknown').fillna('Unknown')
                            
                            t_filtered = t_filtered.sort_values(by=['Time', 'Course'])
                            dl_cols = [c for c in ['Date', 'Time', 'Course', 'Horse', '7:30AM Price', 'ML_Prob', 'Rank', 'Pure Rank'] if c in t_filtered.columns]
                            
                            clean_t_filtered = clean_csv_df(t_filtered[dl_cols])
                            csv_data_out = clean_t_filtered.to_csv(index=False).encode('utf-8')
                            
                            timestamp_out = datetime.now().strftime('%d%m%y_%H%M%S')
                            
                            qual_html_out = '<div style="overflow-x: auto; width: 100%;"><table class="builder-table" style="min-width: 700px;"><thead><tr style="background-color: #2e7d32; color: white;"><th class="center-text">Date</th><th class="center-text">Time</th><th class="left-align">Course</th><th class="left-align">Horse</th><th class="center-text">7:30AM Price</th><th class="center-text">Pure Rank</th></tr></thead><tbody>'
                            for _, q_row in t_filtered.iterrows(): qual_html_out += f"<tr><td class='center-text'>{q_row['Date']}</td><td class='center-text'>{q_row['Time']}</td><td class='left-align'>{q_row['Course']}</td><td class='left-align'><b>{q_row['Horse']}</b></td><td class='center-text'>{q_row['7:30AM Price']:.2f}</td><td class='center-text'><b>{int(q_row.get('Pure Rank', 0))}</b></td></tr>"
                            qual_html_out += "</tbody></table></div>"

                    html_table_out = '<style>.builder-table { border-collapse: collapse; width: 100%; font-size: 14px; font-family: sans-serif; } .builder-table th, .builder-table td { border: 1px solid #ccc; padding: 4px; text-align: center; } .builder-table tr:hover { background-color: #0000FF !important; color: white !important; } .left-align { text-align: left !important; padding-left: 8px !important; }</style>'
                    html_table_out += '<div style="overflow-x: auto; width: 100%;"><table class="builder-table" style="min-width: 900px;"><thead><tr style="background-color: #f0f2f6; color: black;">'
                    
                    for col in actual_grp_cols:
                        html_table_out += f'<th class="left-align">{col}</th>'
                        
                    html_table_out += '<th>Bets</th><th>Wins</th><th>Win P/L</th><th>Win SR</th><th>Places</th><th>Plc P/L</th><th>Plc SR</th><th>Total P/L</th></tr></thead><tbody>'
                    
                    for _, row in breakdown.iterrows(): 
                        t_col = "#2e7d32" if row['Total P/L'] >= 0 else "#d32f2f"
                        html_table_out += "<tr>"
                        
                        for col in actual_grp_cols:
                            html_table_out += f"<td class='left-align'>{row[col]}</td>"
                            
                        html_table_out += f"<td>{row['Bets']}</td><td>{row['Wins']}</td><td><b>£{row['Win_Profit']:.2f}</b></td><td>{row['Strike Rate (%)']:.2f}%</td><td>{row['Places']}</td><td><b>£{row['Place_Profit']:.2f}</b></td><td>{row['Place SR (%)']:.2f}%</td><td style='color:{t_col};'><b>£{row['Total P/L']:.2f}</b></td></tr>"
                    html_table_out += "</tbody></table></div>"

                    st.session_state['tab4_results'] = {
                        'kpis': kpis,
                        'adv_kpis': advanced_metrics,
                        'breakdown_html': html_table_out, 
                        'qual_html': qual_html_out, 
                        'csv': csv_data_out, 
                        'historic_csv': historic_csv_out,
                        'timestamp': timestamp_out, 
                        'val_warn': val_bsp_warning
                    }
                else: st.session_state['tab4_results'] = "empty"

            if 'tab4_results' in st.session_state:
                if st.session_state['tab4_results'] == "empty": st.warning("No bets found matching these exact criteria.")
                else:
                    res = st.session_state['tab4_results']
                    kpis = res['kpis']
                    adv = res['adv_kpis']
                    st.markdown("### System Preview Performance")
                    kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
                    kpi1.metric("Total Bets", int(kpis[0]))
                    kpi2.metric("Wins", int(kpis[1]))
                    kpi3.metric("Places", int(kpis[2]))
                    kpi4.metric("Win P/L", f"£{kpis[3]:.2f}")
                    kpi5.metric("Place P/L", f"£{kpis[4]:.2f}")
                    kpi6.metric("Total ROI", f"{kpis[5]:.2f}%")
                    
                    st.markdown(f"""
                        <div style='background-color: #f8f9fa; padding: 12px; border-radius: 5px; margin-top: 5px; font-size: 13px; line-height: 2.0;'>
                            <b style='color: #1a3a5f;'>Advanced Metrics:</b> &nbsp; 
                            Win S/R: <b>{adv['win_sr']:.1f}%</b> &nbsp;|&nbsp; 
                            Place S/R: <b>{adv['plc_sr']:.1f}%</b> &nbsp;|&nbsp; 
                            LLR <span title='Longest Losing Run: The maximum consecutive losing bets.' style='cursor:help; border-bottom:1px dotted #888; color:#1a3a5f;'>(?)</span>: <b style='color:#d32f2f;'>{int(adv['llr'])}</b> &nbsp;|&nbsp; 
                            Max DD <span title='Maximum Drawdown: The largest peak-to-trough drop at £1 stakes.' style='cursor:help; border-bottom:1px dotted #888; color:#1a3a5f;'>(?)</span>: <b style='color:#d32f2f;'>£{adv['max_dd']:.2f}</b><br>
                            A/E <span title='Actual vs Expected Wins: Values > 1.0 indicate the system beats market odds.' style='cursor:help; border-bottom:1px dotted #888; color:#1a3a5f;'>(?)</span>: <b>{adv['a_e']:.2f}</b> &nbsp;|&nbsp;
                            Chi Score <span title='Statistical significance: > 3.84 means we are 95% confident this profit is due to a real edge, not luck.' style='cursor:help; border-bottom:1px dotted #888; color:#1a3a5f;'>(?)</span>: <b>{adv['chi']:.2f}</b> &nbsp;|&nbsp;
                            Sortino <span title='Sortino Ratio: Measures profit relative to downside volatility. Higher is better (>1.0 is good).' style='cursor:help; border-bottom:1px dotted #888; color:#1a3a5f;'>(?)</span>: <b>{adv['sortino']:.2f}</b> &nbsp;|&nbsp;
                            Ulcer <span title='Ulcer Index (Flat Staking): 0-10 = Elite / Low Stress | 10-25 = Normal Variance | 25-50 = High Stress | 50+ = Unplayable Rollercoaster.' style='cursor:help; border-bottom:1px dotted #888; color:#1a3a5f;'>(?)</span>: <b>{adv['ulcer']:.2f}</b>
                        </div>
                    """, unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    if res.get('historic_csv'):
                        st.download_button(
                            label="📥 Download All Historic Selections (CSV)", 
                            data=res['historic_csv'], 
                            file_name=f"K2_Historic_System_Selections_{res['timestamp']}.csv", 
                            mime="text/csv", 
                            use_container_width=True
                        )

                    if res['qual_html'] != "":
                        st.markdown("<br>", unsafe_allow_html=True)
                        with st.expander("🔍 View Today's Qualifiers for this System", expanded=False):
                            if res['val_warn']: st.info("ℹ️ **Note:** You selected a 'vs BSP' value filter. Because today's BSP is not yet known, the live qualifiers are falling back to use the 7:30AM Price to check for Value.")
                            st.download_button(label="📥 Download Qualifiers to CSV", data=res['csv'], file_name=f"K2_System_Preview_{res['timestamp']}.csv", mime="text/csv")
                            st.write("") 
                            st.markdown(res['qual_html'], unsafe_allow_html=True)
                    elif df_today is not None and not df_today.empty:
                        st.markdown("<br>", unsafe_allow_html=True)
                        with st.expander("🔍 View Today's Qualifiers for this System", expanded=False):
                            st.info("There are no horses running today that match these exact system criteria.")

                    st.markdown("### Detailed Preview Breakdown")
                    st.markdown(res['breakdown_html'], unsafe_allow_html=True)
    # =========================================================================
    # 🏇 PAGE 5: RACE ANALYSIS
    # =========================================================================
    elif page == "🏇 Race Analysis":
        st.header("🏇 Race Analysis")
        
        st.markdown('''<style>
            div[data-testid="stButton"] button p {
                white-space: pre-wrap !important;
                text-align: center !important;
                line-height: 1.5 !important;
            }
        </style>''', unsafe_allow_html=True)
        
        if df_today is not None and not df_today.empty:
            ta_df = df_today
            
            ta_df['Time'] = ta_df['Time'].astype(str).str.strip()
            ta_df['Course'] = ta_df['Course'].astype(str).str.strip()
            
            if st.session_state.get('analysis_race'):
                sel_c = st.session_state.analysis_race['course']
                sel_t = st.session_state.analysis_race['time']
                
                race_info = ta_df[(ta_df['Course'] == sel_c) & (ta_df['Time'] == sel_t)]
                r_type_str = str(race_info['Race Type'].iloc[0]).strip() if not race_info.empty else "Unknown"
                r_hcap_str = "Hcap" if not race_info.empty and str(race_info['H/Cap'].iloc[0]).strip() == 'Y' else "Non-Hcap"
                
                st.markdown(f"### DETAILED RACE ANALYSIS: {sel_c} | {sel_t} | {r_type_str} ({r_hcap_str})")
                
                all_races_df = ta_df[['Time', 'Course']].drop_duplicates().sort_values(['Time', 'Course'])
                all_races = list(zip(all_races_df['Time'], all_races_df['Course']))
                curr_r_idx = all_races.index((sel_t, sel_c)) if (sel_t, sel_c) in all_races else -1
                
                prev_r = all_races[curr_r_idx - 1] if curr_r_idx > 0 else None
                next_r = all_races[curr_r_idx + 1] if curr_r_idx != -1 and curr_r_idx < len(all_races) - 1 else None
                
                meeting_races_df = ta_df[ta_df['Course'] == sel_c][['Time']].drop_duplicates().sort_values('Time')
                meeting_races = meeting_races_df['Time'].tolist()
                curr_m_idx = meeting_races.index(sel_t) if sel_t in meeting_races else -1
                
                prev_m = meeting_races[curr_m_idx - 1] if curr_m_idx > 0 else None
                next_m = meeting_races[curr_m_idx + 1] if curr_m_idx != -1 and curr_m_idx < len(meeting_races) - 1 else None

                nav_cols = st.columns(5)
                with nav_cols[0]:
                    if prev_r:
                        if st.button(f"⏪ <R ({prev_r[0]})", use_container_width=True):
                            st.session_state.analysis_race = {'course': prev_r[1], 'time': prev_r[0]}
                            st.rerun()
                with nav_cols[1]:
                    if prev_m:
                        if st.button(f"◀ <M ({prev_m})", type="primary", use_container_width=True):
                            st.session_state.analysis_race = {'course': sel_c, 'time': prev_m}
                            st.rerun()
                with nav_cols[2]:
                    if st.button("🔙 Back to Race Selection", use_container_width=True):
                        st.session_state.analysis_race = None
                        st.rerun()
                with nav_cols[3]:
                    if next_m:
                        if st.button(f"M> ({next_m}) ▶", type="primary", use_container_width=True):
                            st.session_state.analysis_race = {'course': sel_c, 'time': next_m}
                            st.rerun()
                with nav_cols[4]:
                    if next_r:
                        if st.button(f"R> ({next_r[0]}) ⏩", use_container_width=True):
                            st.session_state.analysis_race = {'course': next_r[1], 'time': next_r[0]}
                            st.rerun()
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                if 'ra_sort_by' not in st.session_state: st.session_state.ra_sort_by = "Pure Rank"
                if 'ra_sort_dir' not in st.session_state: st.session_state.ra_sort_dir = "Ascending (Low to High)"
                
                sc1, sc2 = st.columns([1.5, 3])
                with sc1:
                    sort_cols = ["Pure Rank", "7:30AM Price", "Speed Rank", "Comb. Rank", "Race Rank", "Race Rating", "Comp. Rank", "PRB Rank", "No. of Top", "Primary Rank", "Form Rank", "Value Price"]
                    
                    if r_type_str in ['A/W', 'Turf'] and 'MSAI Rank' in ta_df.columns:
                        sort_cols.insert(7, "MSAI Rank")
                    
                    try: current_sort_idx = sort_cols.index(st.session_state.ra_sort_by)
                    except ValueError: current_sort_idx = 0
                        
                    sort_by = st.selectbox("🔀 Sort Analysis By:", sort_cols, index=current_sort_idx)
                    st.session_state.ra_sort_by = sort_by 
                    
                with sc2:
                    st.markdown("<div style='margin-top: 32px;'></div>", unsafe_allow_html=True)
                    
                    dir_idx = 0 if "Ascending" in st.session_state.ra_sort_dir else 1
                    sort_dir = st.radio("Sort Direction:", ["Ascending (Low to High)", "Descending (High to Low)"], index=dir_idx, horizontal=True, label_visibility="collapsed")
                    st.session_state.ra_sort_dir = sort_dir 
                
                is_asc = "Ascending" in sort_dir
                
                race_df = ta_df[(ta_df['Course'] == sel_c) & (ta_df['Time'] == sel_t)].copy()
                
                if sort_by in race_df.columns:
                    race_df['sort_temp'] = pd.to_numeric(race_df[sort_by], errors='coerce').fillna(999 if is_asc else -1)
                    race_df = race_df.sort_values(by=['sort_temp', 'Rank'], ascending=[is_asc, True])
                else:
                    race_df = race_df.sort_values(by=['Pure Rank', 'Rank'], ascending=[True, True])
                
                st.markdown("<div style='margin-bottom: 10px;'></div>", unsafe_allow_html=True)
                
                show_msai = False
                if r_type_str in ['A/W', 'Turf']:
                    irish_col = 'Irish?' if 'Irish?' in race_df.columns else 'Irish' if 'Irish' in race_df.columns else None
                    is_irish = (race_df[irish_col].astype(str).str.strip().str.upper() == 'Y').any() if irish_col else False
                    
                    if not is_irish and 'MSAI Rank' in race_df.columns:
                        if pd.to_numeric(race_df['MSAI Rank'], errors='coerce').fillna(0).max() > 0:
                            show_msai = True
                
                def gv(r, c, num=False, default="-"):
                    v = r.get(c, default)
                    if pd.isna(v) or v == "": return default
                    if num:
                        try: return float(v)
                        except: return default
                    return v
                
                def rc(v):
                    try:
                        v = int(float(v))
                        if v == 1: return "r1"
                        if v == 2: return "r2"
                        if v == 3: return "r3"
                    except: pass
                    return ""

                def fmt_int(v):
                    try: return str(int(float(v)))
                    except: return "-"

                def fmt_2dp(v):
                    try: return f"{float(v):.2f}"
                    except: return "-"

                show_draw = r_type_str in ['A/W', 'Turf']
                form_colspan = 9 if show_draw else 8
                
                html = '<div style="overflow-x: auto; width: 100%;">'
                html += '<table class="k2-table" style="width:100%; min-width: 950px;"><thead><tr style="background-color: #1a3a5f; color: white;">'
                headers = ["Horse", "Value", "7:30am Price", "Speed Rank", "Comb. Rank", "Race Rank", "Race Rating", "Comp. Rank", "PRB Rank"]
                if show_msai: headers.append("MSAI Rank")
                
                headers.extend(["No. of Top", "Primary Rank", "Form Rank"])
                
                if 'No. of Top' in race_df.columns:
                    race_df['No. of Top'] = pd.to_numeric(race_df['No. of Top'], errors='coerce').fillna(0).astype(int)
                
                for h in headers: 
                    w_style = ' style="width: 9%;"' if h == "Horse" else ''
                    html += f'<th rowspan="2" class="{"left-head" if h == "Horse" else "center-text"}"{w_style}>{h}</th>'
                
                html += f'<th colspan="{form_colspan}" class="center-text" style="border-bottom: 1px dashed #ccc; letter-spacing: 2px; color: #a9bacd;">----------------------- FORM -----------------------</th>'
                html += '<th rowspan="2" class="center-text" style="background-color: #000;">Pure Rank</th></tr><tr style="background-color: #1a3a5f; color: white;">'
                
                form_headers = ["Ability", "Going", "Dist.", "Course/Sim", "Trainer", "Jockey"]
                if show_draw: form_headers.append("Draw")
                form_headers.extend(["Speed", "Total"])
                
                for h in form_headers: 
                    w_style_f = ' style="width: 8%;"' if h == "Dist." else ''
                    html += f'<th class="center-text"{w_style_f}>{h}</th>'
                html += '</tr></thead><tbody>'
                
                for _, r in race_df.iterrows():
                    vp, pr = gv(r,"Value Price",True), gv(r,"7:30AM Price",True)
                    sr, cr, rr, cpr, prb = gv(r,"Speed Rank"), gv(r,"Comb. Rank"), gv(r,"Race Rank"), gv(r,"Comp. Rank"), gv(r,"PRB Rank")
                    
                    pure_r = fmt_int(gv(r, "Pure Rank"))
                    no_top = fmt_int(gv(r, "No. of Top"))
                    prim_r = fmt_int(gv(r, "Primary Rank"))
                    form_r = fmt_int(gv(r, "Form Rank"))
                    
                    html += '<tr>'
                    html += f'<td class="left-text"><b>{gv(r, "Horse")}</b></td>'
                    html += f'<td class="center-text">{f"{vp:.2f}" if isinstance(vp, float) else vp}</td><td class="center-text">{f"{pr:.2f}" if isinstance(pr, float) else pr}</td>'
                    html += f'<td class="center-text {rc(sr)}">{sr}</td><td class="center-text {rc(cr)}">{cr}</td><td class="center-text {rc(rr)}">{rr}</td><td class="center-text">{gv(r, "Race Rating", default=0)}</td><td class="center-text {rc(cpr)}">{cpr}</td><td class="center-text {rc(prb)}">{prb}</td>'
                    
                    if show_msai:
                        msai = fmt_int(gv(r, "MSAI Rank"))
                        html += f'<td class="center-text {rc(msai)}">{msai}</td>'
                        
                    html += f'<td class="center-text">{no_top}</td>'
                    html += f'<td class="center-text {rc(prim_r)}">{prim_r}</td><td class="center-text {rc(form_r)}">{form_r}</td>'
                    
                    ab = fmt_2dp(gv(r, "Ability"))
                    go = fmt_2dp(gv(r, "Going"))
                    di = fmt_2dp(gv(r, "Distance"))
                    cs = fmt_2dp(gv(r, "Course/Sim"))
                    tr = fmt_2dp(gv(r, "TrainrF"))
                    jo = fmt_2dp(gv(r, "JockyF"))
                    dr = fmt_2dp(gv(r, "Draw"))
                    sp = fmt_2dp(gv(r, "Speed"))
                    ts = fmt_2dp(gv(r, "Total"))
                    
                    html += f'<td class="center-text">{ab}</td><td class="center-text">{go}</td><td class="center-text">{di}</td><td class="center-text">{cs}</td><td class="center-text">{tr}</td><td class="center-text">{jo}</td>'
                    if show_draw:
                        html += f'<td class="center-text">{dr}</td>'
                    html += f'<td class="center-text">{sp}</td>'
                    html += f'<td class="center-text" style="font-weight:bold;">{ts}</td>'
                    html += f'<td class="center-text {rc(pure_r)}" style="font-weight:bold;">{pure_r}</td>'
                    html += '</tr>'
                    
                html += "</tbody></table></div>"
                st.markdown(html, unsafe_allow_html=True)

                if st.session_state.get("is_admin"):
                    st.markdown("<br><hr>", unsafe_allow_html=True)
                    with st.expander("🧠 Admin: AI Decision X-Ray (Under the Hood)", expanded=False):
                        st.markdown("This shows the exact raw data the Machine Learning model used to calculate the probabilities for the Top 3 horses in this race.")
                        
                        top_3_df = race_df.nsmallest(3, 'Pure Rank').copy()
                        
                        if not top_3_df.empty:
                            xray_cols = ['Horse', 'Pure Rank'] + shadow_feats 
                            
                            valid_xray_cols = [c for c in xray_cols if c in top_3_df.columns]
                            xray_df = top_3_df[valid_xray_cols].set_index('Horse').T
                            
                            st.dataframe(
                                xray_df.style.format("{:.3f}", na_rep="-"),
                                use_container_width=True
                            )
                        else:
                            st.info("Not enough data to generate X-Ray.")
                
            else:
                st.markdown("### Race Selection")
                courses = sorted([str(x).strip() for x in ta_df['Course'].dropna().unique() if str(x).strip()])
                
                for course in courses:
                    st.markdown(f"<div style='border-left: 6px solid #1a3a5f; padding-left: 12px; margin-top: 15px; margin-bottom: 10px; font-weight: bold; font-size: 16px; color: #1a3a5f; text-transform: uppercase;'>{course}</div>", unsafe_allow_html=True)
                    
                    c_df = ta_df[ta_df['Course'] == course]
                    races = c_df[['Time', 'Race Type', 'H/Cap']].drop_duplicates().sort_values('Time')
                    
                    cols = st.columns(10)
                    for idx, (_, r_row) in enumerate(races.iterrows()):
                        r_time = str(r_row['Time']).strip()
                        r_type = str(r_row['Race Type']).strip()
                        r_hcap = "Hcap" if str(r_row['H/Cap']).strip() == 'Y' else "Non-Hcap"
                        
                        btn_text = f"{r_time}\n{r_type} | {r_hcap}"
                        
                        if cols[idx % 10].button(btn_text, key=f"nav_{course}_{r_time}", use_container_width=True):
                            st.session_state.analysis_race = {'course': course, 'time': r_time}
                            st.rerun()

        else:
            st.info("No data available for today's races.")
            
    # =========================================================================
    # 🧪 PAGE 6: ACID TEST ENVIRONMENT (OUT-OF-SAMPLE VALUE TEST)
    # =========================================================================
    elif page == "🧪 Acid Test Environment":
        st.header("🧪 The Acid Test: Out-of-Sample Value Validation")
        st.markdown("""
        **How it works:** This environment proves if our "True Value Price" can survive the future.
        1. Slices your historical data chronologically at "The Wall".
        2. Trains the Leashed AI strictly on the past to calculate True Probabilities.
        3. Identifies bets in the past AND the future where the Market Odds beat our True Price by your required Edge %.
        """)
        
        st.markdown("---")
        
        if df_all is not None and not df_all.empty:
            c_date, c_edge, c_btn = st.columns([2, 1.5, 1.5])
            
            with c_date:
                min_dt = df_all['Date_DT'].min().date()
                max_dt = df_all['Date_DT'].max().date()
                default_split = datetime(2026, 1, 1).date()
                if default_split < min_dt or default_split > max_dt:
                    default_split = min_dt + timedelta(days=(max_dt - min_dt).days * 0.7)
                
                split_date = st.date_input("Select 'The Wall' (Split Date):", value=default_split, min_value=min_dt, max_value=max_dt)
                
            with c_edge:
                test_edge = st.number_input("Test Minimum Value Edge %:", value=10.0, step=5.0)
                
            with c_btn:
                st.markdown("<br>", unsafe_allow_html=True)
                run_test = st.button("🚀 Run Blind Value Test", use_container_width=True, type="primary")

            if run_test:
                with st.spinner(f"Training Leashed AI and testing >{test_edge}% Edge blindly..."):
                    df_clean = df_all[df_all['Fin Pos'] > 0].copy()
                    
                    train_df = df_clean[df_clean['Date_DT'].dt.date < split_date].copy()
                    test_df = df_clean[df_clean['Date_DT'].dt.date >= split_date].copy()
                    
                    if train_df.empty or test_df.empty:
                        st.error("Error: Not enough data on one side of the split date.")
                    else:
                        test_model = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, max_depth=3, l2_regularization=15.0, min_samples_leaf=250, random_state=42)
                        test_model.fit(train_df[feats].fillna(0), (train_df['Fin Pos'] == 1).astype(int))
                        
                        train_df['Blind_Prob'] = test_model.predict_proba(train_df[feats].fillna(0))[:, 1]
                        test_df['Blind_Prob'] = test_model.predict_proba(test_df[feats].fillna(0))[:, 1]
                        
                        train_df['Blind_True_Price'] = np.where(train_df['Blind_Prob'] > 0.001, 1.0 / train_df['Blind_Prob'], 1000.0)
                        test_df['Blind_True_Price'] = np.where(test_df['Blind_Prob'] > 0.001, 1.0 / test_df['Blind_Prob'], 1000.0)
                        
                        for d in [train_df, test_df]:
                            d['Market_Price'] = np.where(d['BSP'] > 0, d['BSP'], d['7:30AM Price'])
                            d['Blind_Edge'] = ((d['Market_Price'] / d['Blind_True_Price']) - 1) * 100
                            d['Is_Win'] = np.where(d['Win P/L <2%'] > 0, 1, 0)
                        
                        def calculate_value_kpis(data_subset, target_edge):
                            bets = data_subset[(data_subset['Blind_Edge'] >= target_edge) & (data_subset['Market_Price'] <= 20.0)]
                            count = len(bets)
                            if count == 0: return 0, 0, 0, 0
                            wins = bets['Is_Win'].sum()
                            profit = bets['Win P/L <2%'].sum()
                            roi = (profit / count) * 100
                            sr = (wins / count) * 100
                            return count, profit, roi, sr
                            
                        tr_count, tr_profit, tr_roi, tr_sr = calculate_value_kpis(train_df, test_edge)
                        te_count, te_profit, te_roi, te_sr = calculate_value_kpis(test_df, test_edge)
                        
                        st.success(f"✅ Blind Test Complete! Testing all bets with > {test_edge}% True Value Edge (Max Price 20.0).")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"""
                            <div style='background-color: #e8f4f8; padding: 20px; border-radius: 10px; border-left: 5px solid #1976d2;'>
                                <h4>📚 Training Set (The Past)</h4>
                                <p style='font-size: 14px; color: #555;'>Data <b>before</b> {split_date.strftime('%d %b %Y')}</p>
                                <h2>ROI: <span style='color: {"#2e7d32" if tr_roi >= 0 else "#d32f2f"};'>{tr_roi:.2f}%</span></h2>
                                <p><b>Total Bets:</b> {tr_count}<br>
                                <b>Win S/R:</b> {tr_sr:.1f}%<br>
                                <b>Total P/L:</b> £{tr_profit:.2f}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        with col2:
                            st.markdown(f"""
                            <div style='background-color: #f8e8e8; padding: 20px; border-radius: 10px; border-left: 5px solid #d32f2f;'>
                                <h4>🎯 Blind Test Set (The Future)</h4>
                                <p style='font-size: 14px; color: #555;'>Data <b>after</b> {split_date.strftime('%d %b %Y')}</p>
                                <h2>ROI: <span style='color: {"#2e7d32" if te_roi >= 0 else "#d32f2f"};'>{te_roi:.2f}%</span></h2>
                                <p><b>Total Bets:</b> {te_count}<br>
                                <b>Win S/R:</b> {te_sr:.1f}%<br>
                                <b>Total P/L:</b> £{te_profit:.2f}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        st.markdown("<br>", unsafe_allow_html=True)
                        if te_roi > 0 and tr_roi > 0:
                            st.info("🏆 **Verdict: Phenomenal.** The Value calculation generated profit in the past AND survived the blind test into the future. Your edge is real.")
                        elif tr_roi > 0 and te_roi <= 0:
                            st.warning("⚠️ **Verdict: Failed.** The Value edge was profitable in the past but collapsed in the blind future. The threshold might be too low, or the edge dried up.")
                        else:
                            st.error("🚨 **Verdict: Unprofitable.** This Value threshold did not generate a profit in either the past or the future.")

    if st.session_state.get("is_admin"):
        st.sidebar.markdown("---")
        st.sidebar.markdown("### ⚙️ Admin Tools")
        
        if os.path.exists("K2_login_log.csv"):
            with open("K2_login_log.csv", "rb") as f:
                st.sidebar.download_button(
                    label="📥 Download Login Logs", 
                    data=f.read(), 
                    file_name=f"K2_Login_Logs_{datetime.now().strftime('%d%m%y')}.csv", 
                    mime="text/csv", 
                    use_container_width=True
                )
        else:
            st.sidebar.info("ℹ️ No login logs generated yet.")

        if os.path.exists("K2_performance_log.csv"):
            with open("K2_performance_log.csv", "rb") as f:
                st.sidebar.download_button(
                    label="📊 Download Server RAM Logs", 
                    data=f.read(), 
                    file_name=f"K2_RAM_Logs_{datetime.now().strftime('%d%m%y')}.csv", 
                    mime="text/csv", 
                    use_container_width=True
                )

        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🗄️ Prediction Vault")
        st.sidebar.markdown("<span style='font-size:12px;'>Generate a master snapshot of all historical AI probabilities to lock in backtests against model drift.</span>", unsafe_allow_html=True)
        
        if df_all_prepped is not None and not df_all_prepped.empty:
            vault_export_cols = ['Date_Key', 'Time', 'Course', 'Horse', 'ML_Prob', 'Shadow_Prob']
            existing_v_cols = [c for c in vault_export_cols if c in df_all_prepped.columns]
            vault_export_df = df_all_prepped[existing_v_cols].copy()
            
            for col in ['ML_Prob', 'Shadow_Prob']:
                if col in vault_export_df.columns:
                    vault_export_df[col] = vault_export_df[col].round(6)
            
            vault_csv = vault_export_df.to_csv(index=False).encode('utf-8')
            st.sidebar.download_button(
                label="📥 Download Master Vault CSV", 
                data=vault_csv, 
                file_name="K2_Prediction_Vault.csv", 
                mime="text/csv", 
                use_container_width=True
            )
