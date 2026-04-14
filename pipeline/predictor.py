import streamlit as st
import pandas as pd


def show_predictor(model, feature_names):
    st.markdown("""
        <div style='
            background: linear-gradient(135deg, rgba(212,255,0,0.08), rgba(0,212,180,0.08));
            border: 1px solid rgba(212,255,0,0.2);
            border-radius: 18px;
            padding: 28px 32px;
            margin-bottom: 24px;
        '>
            <h2 style='color:#d4ff00; font-family:Outfit,sans-serif;
                       font-size:1.6rem; font-weight:800;
                       margin:0 0 6px 0; letter-spacing:-0.5px;'>
                🔮 Live Loan Predictor
            </h2>
            <p style='color:#94a3b8; font-family:Outfit,sans-serif;
                      font-size:0.9rem; margin:0;'>
                Fill in the applicant details below and get an instant prediction
            </p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("<p style='color:#94a3b8;font-size:0.78rem;font-weight:600;text-transform:uppercase;letter-spacing:1.2px;font-family:Outfit,sans-serif;margin-bottom:4px;'>CIBIL Score</p>", unsafe_allow_html=True)
        cibil = st.number_input("", min_value=300, max_value=900, value=750, step=1, key="cibil", label_visibility="collapsed")

        st.markdown("<p style='color:#94a3b8;font-size:0.78rem;font-weight:600;text-transform:uppercase;letter-spacing:1.2px;font-family:Outfit,sans-serif;margin-bottom:4px;margin-top:12px;'>Annual Income</p>", unsafe_allow_html=True)
        income = st.number_input("", min_value=0, max_value=100000000, value=800000, step=50000, key="income", label_visibility="collapsed")

        st.markdown("<p style='color:#94a3b8;font-size:0.78rem;font-weight:600;text-transform:uppercase;letter-spacing:1.2px;font-family:Outfit,sans-serif;margin-bottom:4px;margin-top:12px;'>Loan Amount</p>", unsafe_allow_html=True)
        loan_amt = st.number_input("", min_value=0, max_value=500000000, value=2000000, step=100000, key="loan_amt", label_visibility="collapsed")

    with col2:
        st.markdown("<p style='color:#94a3b8;font-size:0.78rem;font-weight:600;text-transform:uppercase;letter-spacing:1.2px;font-family:Outfit,sans-serif;margin-bottom:4px;'>Loan Term (months)</p>", unsafe_allow_html=True)
        loan_term = st.number_input("", min_value=1, max_value=360, value=12, step=1, key="loan_term", label_visibility="collapsed")

        st.markdown("<p style='color:#94a3b8;font-size:0.78rem;font-weight:600;text-transform:uppercase;letter-spacing:1.2px;font-family:Outfit,sans-serif;margin-bottom:4px;margin-top:12px;'>No. of Dependents</p>", unsafe_allow_html=True)
        dependents = st.number_input("", min_value=0, max_value=10, value=2, step=1, key="dependents", label_visibility="collapsed")

        st.markdown("<p style='color:#94a3b8;font-size:0.78rem;font-weight:600;text-transform:uppercase;letter-spacing:1.2px;font-family:Outfit,sans-serif;margin-bottom:4px;margin-top:12px;'>Residential Assets</p>", unsafe_allow_html=True)
        res_assets = st.number_input("", min_value=0, max_value=500000000, value=2000000, step=100000, key="res_assets", label_visibility="collapsed")

    with col3:
        st.markdown("<p style='color:#94a3b8;font-size:0.78rem;font-weight:600;text-transform:uppercase;letter-spacing:1.2px;font-family:Outfit,sans-serif;margin-bottom:4px;'>Commercial Assets</p>", unsafe_allow_html=True)
        com_assets = st.number_input("", min_value=0, max_value=500000000, value=1000000, step=100000, key="com_assets", label_visibility="collapsed")

        st.markdown("<p style='color:#94a3b8;font-size:0.78rem;font-weight:600;text-transform:uppercase;letter-spacing:1.2px;font-family:Outfit,sans-serif;margin-bottom:4px;margin-top:12px;'>Luxury Assets</p>", unsafe_allow_html=True)
        lux_assets = st.number_input("", min_value=0, max_value=500000000, value=5000000, step=100000, key="lux_assets", label_visibility="collapsed")

        st.markdown("<p style='color:#94a3b8;font-size:0.78rem;font-weight:600;text-transform:uppercase;letter-spacing:1.2px;font-family:Outfit,sans-serif;margin-bottom:4px;margin-top:12px;'>Bank Assets</p>", unsafe_allow_html=True)
        bank_assets = st.number_input("", min_value=0, max_value=500000000, value=3000000, step=100000, key="bank_assets", label_visibility="collapsed")

    st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)

    col4, col5, col6 = st.columns([1, 1, 1])
    with col4:
        st.markdown("<p style='color:#94a3b8;font-size:0.78rem;font-weight:600;text-transform:uppercase;letter-spacing:1.2px;font-family:Outfit,sans-serif;margin-bottom:4px;'>Education</p>", unsafe_allow_html=True)
        education = st.selectbox("", ["Graduate", "Not Graduate"], key="edu", label_visibility="collapsed")
    with col5:
        st.markdown("<p style='color:#94a3b8;font-size:0.78rem;font-weight:600;text-transform:uppercase;letter-spacing:1.2px;font-family:Outfit,sans-serif;margin-bottom:4px;'>Self Employed</p>", unsafe_allow_html=True)
        self_emp = st.selectbox("", ["No", "Yes"], key="self_emp", label_visibility="collapsed")
    with col6:
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🔮 Predict Now", key="predict_btn")

    if predict_btn:
        input_dict = {
            'no_of_dependents':         dependents,
            'income_annum':             income,
            'loan_amount':              loan_amt,
            'loan_term':                loan_term,
            'cibil_score':              cibil,
            'residential_assets_value': res_assets,
            'commercial_assets_value':  com_assets,
            'luxury_assets_value':      lux_assets,
            'bank_asset_value':         bank_assets,
            'education':    1 if education == "Not Graduate" else 0,
            'self_employed': 1 if self_emp == "Yes" else 0,
        }

        input_df = pd.DataFrame([input_dict])

        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[feature_names]


        prediction = model.predict(input_df)[0]
        confidence = model.predict_proba(input_df)[0][prediction] * 100

        if prediction == 0:
            st.markdown(f"""
                <div style='
                    background: linear-gradient(135deg, rgba(16,185,129,0.12), rgba(212,255,0,0.06));
                    border: 1.5px solid rgba(16,185,129,0.5);
                    border-radius: 20px;
                    padding: 36px 40px;
                    text-align: center;
                    margin-top: 24px;
                '>
                    <div style='font-size:3.5rem;margin-bottom:12px;'>✅</div>
                    <div style='font-family:Outfit,sans-serif;font-size:2.4rem;
                                font-weight:900;color:#10b981;letter-spacing:-1px;
                                margin-bottom:8px;'>LOAN APPROVED</div>
                    <div style='font-family:Outfit,sans-serif;font-size:1rem;
                                color:#94a3b8;margin-bottom:20px;'>
                        The model predicts this applicant qualifies for the loan
                    </div>
                    <div style='display:inline-block;background:rgba(16,185,129,0.15);
                                border:1px solid rgba(16,185,129,0.4);border-radius:50px;
                                padding:8px 24px;font-family:DM Mono,monospace;
                                font-size:1.1rem;font-weight:600;color:#10b981;'>
                        Confidence: {confidence:.1f}%
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div style='
                    background: linear-gradient(135deg, rgba(239,68,68,0.12), rgba(255,107,53,0.06));
                    border: 1.5px solid rgba(239,68,68,0.5);
                    border-radius: 20px;
                    padding: 36px 40px;
                    text-align: center;
                    margin-top: 24px;
                '>
                    <div style='font-size:3.5rem;margin-bottom:12px;'>❌</div>
                    <div style='font-family:Outfit,sans-serif;font-size:2.4rem;
                                font-weight:900;color:#ef4444;letter-spacing:-1px;
                                margin-bottom:8px;'>LOAN REJECTED</div>
                    <div style='font-family:Outfit,sans-serif;font-size:1rem;
                                color:#94a3b8;margin-bottom:20px;'>
                        The model predicts this applicant does not qualify
                    </div>
                    <div style='display:inline-block;background:rgba(239,68,68,0.15);
                                border:1px solid rgba(239,68,68,0.4);border-radius:50px;
                                padding:8px 24px;font-family:DM Mono,monospace;
                                font-size:1.1rem;font-weight:600;color:#ef4444;'>
                        Confidence: {confidence:.1f}%
                    </div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
        st.markdown("<h3 style='color:#94a3b8;font-family:Outfit,sans-serif;font-size:0.78rem;text-transform:uppercase;letter-spacing:2px;margin-bottom:16px;'>Key Factors</h3>", unsafe_allow_html=True)

        f1, f2, f3, f4 = st.columns(4)
        cibil_color  = "#10b981" if cibil >= 700 else "#ef4444"
        income_color = "#10b981" if income >= 500000 else "#ef4444"
        ratio        = (loan_amt / income * 100) if income > 0 else 0
        ratio_color  = "#10b981" if ratio < 300 else "#ef4444"
        asset_total  = res_assets + com_assets + lux_assets + bank_assets
        asset_color  = "#10b981" if asset_total >= loan_amt else "#ef4444"

        for col_obj, label, value, color in [
            (f1, "CIBIL Score",   str(cibil),            cibil_color),
            (f2, "Annual Income", f"₹{income:,.0f}",     income_color),
            (f3, "Loan/Income %", f"{ratio:.0f}%",       ratio_color),
            (f4, "Total Assets",  f"₹{asset_total:,.0f}", asset_color),
        ]:
            col_obj.markdown(f"""
                <div style='background:#16213e;border:1px solid rgba(255,255,255,0.07);
                            border-radius:14px;padding:18px 16px;text-align:center;
                            border-top:2px solid {color};'>
                    <div style='color:#94a3b8;font-size:0.7rem;font-weight:600;
                                text-transform:uppercase;letter-spacing:1.2px;
                                font-family:Outfit,sans-serif;margin-bottom:8px;'>
                        {label}
                    </div>
                    <div style='color:{color};font-size:1.3rem;font-weight:800;
                                font-family:Outfit,sans-serif;'>
                        {value}
                    </div>
                </div>
            """, unsafe_allow_html=True)