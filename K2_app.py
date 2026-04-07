# --- Tab 4: Mini SYSTEM BUILDER ---
    with tab4:
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

        if df_all is not None and not df_all.empty:
            b_df = prep_system_builder_data(df_all, model, feats, shadow_model, shadow_feats)

            # --- Extract available options from the dataset ---
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
            rank_opts = ["Any", "Rank 1", "Top 2", "Top 3", "Not Top 3"]
            
            defaults = {
                'ui_months': all_months, 'ui_courses': [], 'ui_race_types': rt_opts, 'ui_hcap_types': hcap_opts,
                'ui_price_min': 0.0, 'ui_price_max': 1000.0, 'ui_prob_gap': -100.0,
                'ui_rnrs': rnr_opts, 'ui_classes': class_opts, 'ui_cm': cm_opts,
                'ui_ai_rank_filter': "Any", 'ui_sex': sex_opts, 'ui_value_filter': "Off",
                'ui_irish_f': "Any", 'ui_age_range': (1, 20),
                'ui_comb_f': "Any", 'ui_comp_f': "Any", 'ui_speed_f': "Any", 'ui_race_f': "Any", 'ui_primary_f': "Any",
                'ui_msai_f': "Any", 'ui_prb_f': "Any", 'ui_tprb_f': "Any", 'ui_jprb_f': "Any", 'ui_form_f': "Any", 'ui_pure_f': "Any"
            }
            
            if st.session_state.get('force_reset', False):
                for k, v in defaults.items(): st.session_state[k] = v
                st.session_state['force_reset'] = False
            else:
                for k, v in defaults.items():
                    if k not in st.session_state: st.session_state[k] = v

            # --- ROBUST INJECTION FUNCTION ---
            def load_sys_to_ui(sys_data):
                st.session_state['ui_months'] = sys_data.get('months', all_months)
                st.session_state['ui_courses'] = [c for c in sys_data.get('courses', []) if c in course_opts]
                st.session_state['ui_race_types'] = [r for r in sys_data.get('race_types', []) if r in rt_opts]
                st.session_state['ui_hcap_types'] = [h for h in sys_data.get('hcap_types', []) if h in hcap_opts]
                st.session_state['ui_price_min'] = float(sys_data.get('price_min', 0.0))
                st.session_state['ui_price_max'] = float(sys_data.get('price_max', 1000.0))
                st.session_state['ui_prob_gap'] = float(sys_data.get('min_prob_gap', -1.0)) * 100
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

            # --- Explicit index retrieval helpers for selectboxes ---
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
                    with p_col1: price_min = st.number_input("Min Price", 0.0, 1000.0, value=float(st.session_state.get('ui_price_min', 0.0)), step=0.5)
                    with p_col2: price_max = st.number_input("Max Price", 0.0, 1000.0, value=float(st.session_state.get('ui_price_max', 1000.0)), step=0.5)
                    min_prob_gap = st.number_input("Minimum Prob Gap (%)", -100.0, 50.0, value=float(st.session_state.get('ui_prob_gap', -100.0)), step=0.5) / 100
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
                
                submit_button = st.form_submit_button(label="🚀 Process Data")

            if st.session_state.get("is_admin"):
                st.markdown("---")
                st.markdown("### ⚙️ Admin: Generate System Code (For GitHub)")
                c_name, c_btn = st.columns([3, 1])
                with c_name: new_sys_name = st.text_input("System Name:", placeholder="e.g., A/W MSAI Value")
                with c_btn:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("Generate JSON Code", use_container_width=True):
                        if new_sys_name:
                            sys_data = {
                                "race_types": selected_race_types, "hcap_types": selected_hcap, "price_min": price_min, "price_max": price_max, "min_prob_gap": min_prob_gap, "rnrs": selected_rnrs, "classes": selected_classes, "cm": selected_cm, "sex": selected_sex, "courses": selected_courses, "ai_rank_filter": ai_rank_filter, "value_filter": value_filter, "irish": irish_f, "age_min": age_min, "age_max": age_max, "months": selected_months,
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
                                        with st.expander(f"🔍 {s_key}"): 
                                            cc1, cc2 = st.columns([4, 1])
                                            with cc1: st.json(s_data)
                                            with cc2: 
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
                                        with st.expander(f"🔍 {s_key}"):
                                            cc1, cc2 = st.columns([4, 1])
                                            with cc1: st.json(s_data)
                                            with cc2: 
                                                if st.button("Load 📥", key=f"load_sec_{s_key}", use_container_width=True):
                                                    load_sys_to_ui(s_data)
                                                    st.rerun()
                                else: st.write("No admin systems currently active.")
                            except Exception as e: st.error(f"Error reading admin file: {e}")
                st.markdown("---")

            if submit_button:
                st.success("✅ System recalculated instantly!")

                mask = (b_df['Race Type'].isin(selected_race_types) & b_df['H/Cap'].isin(selected_hcap) & (b_df['7:30AM Price'] >= price_min) & (b_df['7:30AM Price'] <= price_max) & (b_df['Prob Gap'] >= min_prob_gap))
                
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
                    breakdown = df_filtered.groupby(['Race Type', 'H/Cap', 'Price Bracket'], observed=False).agg(
                        Bets=('Horse', 'count'), Wins=('Is_Win', 'sum'), Win_Profit=('Win P/L <2%', 'sum'), Places=('Is_Place', 'sum'), Place_Profit=('Place P/L <2%', 'sum')
                    ).reset_index()
                    
                    breakdown = breakdown[breakdown['Bets'] > 0]
                    breakdown['Strike Rate (%)'] = (breakdown['Wins'] / breakdown['Bets'] * 100).fillna(0)
                    breakdown['Place SR (%)'] = (breakdown['Places'] / breakdown['Bets'] * 100).fillna(0)
                    breakdown['Win ROI (%)'] = (breakdown['Win_Profit'] / breakdown['Bets'] * 100).fillna(0)
                    breakdown['Total P/L'] = breakdown['Win_Profit'] + breakdown['Place_Profit']
                    breakdown = breakdown.sort_values(by=['Race Type', 'H/Cap', 'Price Bracket'])

                    total_bets_for_roi = breakdown['Bets'].sum()
                    total_pl_for_roi = breakdown['Total P/L'].sum()
                    total_roi_perc = (total_pl_for_roi / total_bets_for_roi * 100) if total_bets_for_roi > 0 else 0.0

                    kpis = [
                        total_bets_for_roi, breakdown['Wins'].sum(), breakdown['Places'].sum(), 
                        breakdown['Win_Profit'].sum(), breakdown['Place_Profit'].sum(),
                        total_roi_perc
                    ]

                    qual_html_out, csv_data_out, timestamp_out = "", None, ""
                    val_bsp_warning = value_filter in ["Value vs BSP", "AI Value vs BSP", "My Value vs BSP", "NOT AI Value vs BSP", "NOT My Value vs BSP"]

                    if df_today is not None and not df_today.empty:
                        t_df = prep_system_builder_data(df_today, model, feats, shadow_model, shadow_feats, is_live_today=True)
                        
                        t_mask = (t_df['Race Type'].isin(selected_race_types) & t_df['H/Cap'].isin(selected_hcap) & (t_df['7:30AM Price'] >= price_min) & (t_df['7:30AM Price'] <= price_max) & (t_df['Prob Gap'] >= min_prob_gap))
                        
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
                            t_filtered = t_filtered.sort_values(by=['Time', 'Course'])
                            dl_cols = [c for c in ['Date', 'Time', 'Course', 'Horse', '7:30AM Price', 'ML_Prob', 'Rank', 'Pure Rank'] if c in t_filtered.columns]
                            csv_data_out = t_filtered[dl_cols].to_csv(index=False).encode('utf-8')
                            timestamp_out = datetime.now().strftime('%d%m%y_%H%M%S')
                            
                            qual_html_out = '<div style="overflow-x: auto; width: 100%;"><table class="builder-table" style="min-width: 700px;"><thead><tr style="background-color: #2e7d32; color: white;"><th class="center-text">Date</th><th class="center-text">Time</th><th class="left-align">Course</th><th class="left-align">Horse</th><th class="center-text">7:30AM Price</th><th class="center-text">Pure Rank</th></tr></thead><tbody>'
                            for _, q_row in t_filtered.iterrows(): qual_html_out += f"<tr><td class='center-text'>{q_row['Date']}</td><td class='center-text'>{q_row['Time']}</td><td class='left-align'>{q_row['Course']}</td><td class='left-align'><b>{q_row['Horse']}</b></td><td class='center-text'>{q_row['7:30AM Price']:.2f}</td><td class='center-text'><b>{int(q_row.get('Pure Rank', 0))}</b></td></tr>"
                            qual_html_out += "</tbody></table></div>"

                    html_table_out = '<style>.builder-table { border-collapse: collapse; width: 100%; font-size: 14px; font-family: sans-serif; } .builder-table th, .builder-table td { border: 1px solid #ccc; padding: 4px; text-align: center; } .builder-table tr:hover { background-color: #0000FF !important; color: white !important; } .left-align { text-align: left !important; padding-left: 8px !important; }</style>'
                    html_table_out += '<div style="overflow-x: auto; width: 100%;"><table class="builder-table" style="min-width: 900px;"><thead><tr style="background-color: #f0f2f6; color: black;"><th class="left-align">Race Type</th><th class="left-align">H/Cap</th><th class="left-align">Price Bracket</th><th>Bets</th><th>Wins</th><th>Win P/L</th><th>Win SR</th><th>Places</th><th>Plc P/L</th><th>Plc SR</th><th>Total P/L</th></tr></thead><tbody>'
                    for _, row in breakdown.iterrows(): 
                        t_col = "#2e7d32" if row['Total P/L'] >= 0 else "#d32f2f"
                        html_table_out += f"<tr><td class='left-align'>{row['Race Type']}</td><td class='left-align'>{row['H/Cap']}</td><td class='left-align'>{row['Price Bracket']}</td><td>{row['Bets']}</td><td>{row['Wins']}</td><td><b>£{row['Win_Profit']:.2f}</b></td><td>{row['Strike Rate (%)']:.2f}%</td><td>{row['Places']}</td><td><b>£{row['Place_Profit']:.2f}</b></td><td>{row['Place SR (%)']:.2f}%</td><td style='color:{t_col};'><b>£{row['Total P/L']:.2f}</b></td></tr>"
                    html_table_out += "</tbody></table></div>"

                    st.session_state['tab4_results'] = {'kpis': kpis, 'breakdown_html': html_table_out, 'qual_html': qual_html_out, 'csv': csv_data_out, 'timestamp': timestamp_out, 'val_warn': val_bsp_warning}
                else: st.session_state['tab4_results'] = "empty"

            if 'tab4_results' in st.session_state:
                if st.session_state['tab4_results'] == "empty": st.warning("No bets found matching these exact criteria.")
                else:
                    res = st.session_state['tab4_results']
                    kpis = res['kpis']
                    st.markdown("### System Preview Performance")
                    kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
                    kpi1.metric("Total Bets", int(kpis[0]))
                    kpi2.metric("Wins", int(kpis[1]))
                    kpi3.metric("Places", int(kpis[2]))
                    kpi4.metric("Win P/L", f"£{kpis[3]:.2f}")
                    kpi5.metric("Place P/L", f"£{kpis[4]:.2f}")
                    kpi6.metric("Total ROI", f"{kpis[5]:.2f}%")

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
