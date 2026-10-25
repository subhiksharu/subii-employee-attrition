import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from turnover_model import IQRCapper, train  # noqa: F401 (IQRCapper needed by pipeline)

st.set_page_config(page_title="subii | Early Risk Prediction",
                   page_icon="🧭", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown("""
<style>
.block-container{padding-top:1.5rem;max-width:1200px}
.hero{background:linear-gradient(120deg,#4F46E5,#7C3AED 60%,#06B6D4);color:#fff;
 padding:28px 32px;border-radius:18px;margin-bottom:18px}
.hero h1{margin:0;font-size:2rem;color:#fff}.hero p{margin:6px 0 0;opacity:.92}
.card{background:#fff;border:1px solid #E2E8F0;border-radius:14px;padding:16px 18px;
 box-shadow:0 1px 4px rgba(15,23,42,.06);text-align:center}
.card .v{font-size:1.7rem;font-weight:700;color:#4F46E5}.card .l{color:#64748B;font-size:.85rem}
.badge{display:inline-block;padding:6px 16px;border-radius:999px;color:#fff;font-weight:700}
.chip{display:inline-block;background:#FEE2E2;color:#991B1B;padding:4px 12px;
 border-radius:999px;margin:3px 4px 3px 0;font-size:.85rem}
.chip.ok{background:#DCFCE7;color:#166534}
#MainMenu,footer,header,[data-testid="stToolbar"],[data-testid="stDecoration"],
[data-testid="stStatusWidget"],[data-testid="stAppDeployButton"],.stAppDeployButton,
[data-testid="stSidebar"],[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"],[class*="viewerBadge"],[class*="_profileContainer"],
a[href*="github.com"]{display:none!important;visibility:hidden!important}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Training the stacking model (first load only)…")
def get_bundle():
    return train()


B = get_bundle()
model, D = B["model"], B["defaults"]

RISK = {"LOW": ("#16A34A", "Continue regular engagement and retention activities."),
        "MEDIUM": ("#F59E0B", "Monitor engagement, workload, satisfaction and career development."),
        "HIGH": ("#DC2626", "Review workload and career progression. Consider immediate retention action.")}


def risk_level(p):
    return "LOW" if p < 0.35 else "MEDIUM" if p < 0.66 else "HIGH"


def risk_factors(r):
    f = []
    if r["OverTime"] == "Yes": f.append("Works overtime")
    for col, name in [("JobSatisfaction", "Job satisfaction"),
                      ("EnvironmentSatisfaction", "Environment satisfaction"),
                      ("JobInvolvement", "Job involvement"),
                      ("WorkLifeBalance", "Work-life balance")]:
        if r[col] <= 2: f.append(f"{name} low ({r[col]}/4)")
    if r["YearsAtCompany"] <= 2: f.append(f"Short tenure ({r['YearsAtCompany']} yrs)")
    if r["YearsSinceLastPromotion"] >= 4: f.append(f"No promotion for {r['YearsSinceLastPromotion']} yrs")
    if r["MonthlyIncome"] < B["income_p25"]: f.append(f"Income in bottom quartile (₹{r['MonthlyIncome']:,.0f})")
    if r["BusinessTravel"] == "Travel_Frequently": f.append("Frequent business travel")
    if r["StockOptionLevel"] == 0: f.append("No stock options")
    return f


def gauge(p, color):
    fig = go.Figure(go.Indicator(mode="gauge+number", value=p * 100,
        number={"suffix": "%", "font": {"size": 44}},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": color},
               "steps": [{"range": [0, 35], "color": "#DCFCE7"},
                         {"range": [35, 66], "color": "#FEF3C7"},
                         {"range": [66, 100], "color": "#FEE2E2"}]}))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=20, b=0), paper_bgcolor="rgba(0,0,0,0)")
    return fig


st.markdown("""<div class="hero"><h1>🧭 subii – Employee Turnover & Early Risk Prediction</h1>
<p>Stacking ensemble (Random Forest · Decision Tree · KNN · Logistic Regression) that flags
employees at elevated attrition risk so HR can act early.</p></div>""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predict", "📂 Bulk Analysis", "📊 Model Performance", "🔍 Data Insights"])

# ---------------- Predict ----------------
with tab1:
    sat = {1: "1 – Low", 2: "2 – Medium", 3: "3 – High", 4: "4 – Very High"}
    with st.form("emp"):
        st.subheader("Employee profile")
        a, b, c = st.columns(3)
        with a:
            st.markdown("**Personal**")
            Age = st.slider("Age", 18, 60, int(D["Age"]))
            Gender = st.selectbox("Gender", ["Male", "Female"])
            MaritalStatus = st.selectbox("Marital status", ["Single", "Married", "Divorced"], index=1)
            Education = st.slider("Education (1–5)", 1, 5, int(D["Education"]))
            EducationField = st.selectbox("Education field", ["Life Sciences", "Medical", "Marketing",
                                          "Technical Degree", "Human Resources", "Other"])
            DistanceFromHome = st.slider("Distance from home (km)", 1, 30, int(D["DistanceFromHome"]))
        with b:
            st.markdown("**Job**")
            Department = st.selectbox("Department", ["Research & Development", "Sales", "Human Resources"])
            JobRole = st.selectbox("Job role", ["Sales Executive", "Research Scientist", "Laboratory Technician",
                                   "Manufacturing Director", "Healthcare Representative", "Manager",
                                   "Sales Representative", "Research Director", "Human Resources"])
            JobLevel = st.slider("Job level", 1, 5, int(D["JobLevel"]))
            OverTime = st.radio("Overtime", ["No", "Yes"], horizontal=True)
            BusinessTravel = st.selectbox("Business travel", ["Travel_Rarely", "Non-Travel", "Travel_Frequently"])
            PerformanceRating = st.select_slider("Performance rating", [3, 4], value=int(D["PerformanceRating"]))
        with c:
            st.markdown("**Compensation**")
            MonthlyIncome = st.number_input("Monthly income (₹)", 1000, 200000, int(D["MonthlyIncome"]), 500)
            PercentSalaryHike = st.slider("Last salary hike (%)", 11, 25, int(D["PercentSalaryHike"]))
            StockOptionLevel = st.slider("Stock option level", 0, 3, int(D["StockOptionLevel"]))
            TrainingTimesLastYear = st.slider("Trainings last year", 0, 6, int(D["TrainingTimesLastYear"]))
            NumCompaniesWorked = st.slider("Companies worked before", 0, 9, int(D["NumCompaniesWorked"]))
        st.markdown("---")
        d, e = st.columns(2)
        with d:
            st.markdown("**Satisfaction**")
            JobSatisfaction = st.select_slider("Job satisfaction", [1, 2, 3, 4], 3, format_func=sat.get)
            EnvironmentSatisfaction = st.select_slider("Environment satisfaction", [1, 2, 3, 4], 3, format_func=sat.get)
            RelationshipSatisfaction = st.select_slider("Relationship satisfaction", [1, 2, 3, 4], 3, format_func=sat.get)
            JobInvolvement = st.select_slider("Job involvement", [1, 2, 3, 4], 3, format_func=sat.get)
            WorkLifeBalance = st.select_slider("Work-life balance", [1, 2, 3, 4], 3,
                                               format_func={1: "1 – Bad", 2: "2 – Good", 3: "3 – Better", 4: "4 – Best"}.get)
        with e:
            st.markdown("**Tenure**")
            TotalWorkingYears = st.slider("Total working years", 0, 40, int(D["TotalWorkingYears"]))
            YearsAtCompany = st.slider("Years at company", 0, 40, int(D["YearsAtCompany"]))
            YearsInCurrentRole = st.slider("Years in current role", 0, 20, int(D["YearsInCurrentRole"]))
            YearsSinceLastPromotion = st.slider("Years since last promotion", 0, 15, int(D["YearsSinceLastPromotion"]))
            YearsWithCurrManager = st.slider("Years with current manager", 0, 20, int(D["YearsWithCurrManager"]))
        go_btn = st.form_submit_button("Assess attrition risk", type="primary", width="stretch")

    if go_btn:
        row = {**D, **{k: v for k, v in locals().items() if k in B["columns"]}}
        row.update(DailyRate=D["DailyRate"], HourlyRate=D["HourlyRate"], MonthlyRate=D["MonthlyRate"])
        X = pd.DataFrame([row])[B["columns"]]
        p = float(model.predict_proba(X)[0, 1])
        lvl = risk_level(p)
        color, action = RISK[lvl]
        l, r = st.columns([1, 1.2])
        with l:
            st.plotly_chart(gauge(p, color), width="stretch")
            st.markdown(f"<div style='text-align:center'><span class='badge' style='background:{color}'>"
                        f"{lvl} RISK · {'MAY LEAVE' if p >= .5 else 'LIKELY TO STAY'}</span></div>",
                        unsafe_allow_html=True)
        with r:
            st.markdown("#### Key risk factors")
            fs = risk_factors(row)
            st.markdown("".join(f"<span class='chip'>{x}</span>" for x in fs) if fs else
                        "<span class='chip ok'>No major risk factors identified</span>",
                        unsafe_allow_html=True)
            st.markdown("#### Recommended HR action")
            if lvl == "LOW":
                st.success(action)
            elif lvl == "MEDIUM":
                st.warning(action)
            else:
                st.error(action)
            rep = pd.DataFrame([{**row, "AttritionProbability": round(p, 4), "RiskLevel": lvl}])
            st.download_button("⬇️ Download assessment (CSV)", rep.to_csv(index=False),
                               "employee_risk_assessment.csv", "text/csv")

# ---------------- Bulk ----------------
with tab2:
    st.subheader("Score a whole workforce")
    st.caption("Upload a CSV with the same columns as the training data (Attrition column optional).")
    up = st.file_uploader("CSV file", type="csv")
    if up:
        try:
            raw = pd.read_csv(up)
            X = raw.reindex(columns=B["columns"])
            for c in B["columns"]:
                X[c] = X[c].fillna(D[c])
            raw["AttritionProbability"] = model.predict_proba(X)[:, 1].round(4)
            raw["RiskLevel"] = raw["AttritionProbability"].map(risk_level)
            counts = raw["RiskLevel"].value_counts().reindex(["LOW", "MEDIUM", "HIGH"], fill_value=0)
            cols = st.columns(3)
            for col, (k, v) in zip(cols, counts.items()):
                col.markdown(f"<div class='card'><div class='v' style='color:{RISK[k][0]}'>{v}</div>"
                             f"<div class='l'>{k} risk</div></div>", unsafe_allow_html=True)
            st.dataframe(raw.sort_values("AttritionProbability", ascending=False), width="stretch", height=380)
            st.download_button("⬇️ Download scored file", raw.to_csv(index=False), "scored_employees.csv", "text/csv")
        except Exception as ex:
            st.error(f"Could not score this file: {ex}")
    else:
        st.download_button("Download sample template", B["df"].drop(columns="Attrition").head(5).to_csv(index=False),
                           "template.csv", "text/csv")

# ---------------- Performance ----------------
with tab3:
    m = B["metrics"]
    cols = st.columns(5)
    for col, (k, v) in zip(cols, m.items()):
        col.markdown(f"<div class='card'><div class='v'>{v:.2f}</div><div class='l'>{k.replace('_', '-')}</div></div>",
                     unsafe_allow_html=True)
    st.write("")
    l, r = st.columns(2)
    comp = pd.Series(B["comparison"]).sort_values()
    best = comp.idxmax()
    fig = px.bar(comp, orientation="h", text=comp.round(2), labels={"value": "F1 score", "index": ""},
                 title="Model comparison (F1 on held-out test set)", color_discrete_sequence=["#6366F1"])
    fig.update_layout(showlegend=False, height=380)
    l.plotly_chart(fig, width="stretch")
    l.caption(f"Top F1 on the test set: **{best}**. The stacking ensemble combines all four models.")
    cm = B["cm"]
    fig = px.imshow(cm, text_auto=True, x=["Stay", "Leave"], y=["Stay", "Leave"], color_continuous_scale="Blues",
                    labels=dict(x="Predicted", y="Actual"), title="Confusion matrix – Stacking")
    fig.update_layout(height=380, coloraxis_showscale=False)
    r.plotly_chart(fig, width="stretch")

# ---------------- Insights ----------------
with tab4:
    l, r = st.columns(2)
    imp = B["importance"].sort_values("Importance")
    fig = px.bar(imp, x="Importance", y="Feature", orientation="h", title="Top 15 drivers of attrition",
                 color="Importance", color_continuous_scale="Viridis")
    fig.update_layout(height=480, coloraxis_showscale=False)
    l.plotly_chart(fig, width="stretch")
    ac = B["df"]["Attrition"].map({"No": "Stay", "Yes": "Leave"}).value_counts()
    fig = px.pie(values=ac.values, names=ac.index, hole=.55, title="Attrition distribution",
                 color_discrete_sequence=["#6366F1", "#F43F5E"])
    fig.update_layout(height=480)
    r.plotly_chart(fig, width="stretch")
    ot = B["df"].groupby("OverTime")["Attrition"].apply(lambda s: (s == "Yes").mean() * 100).reset_index()
    st.plotly_chart(px.bar(ot, x="OverTime", y="Attrition", text=ot["Attrition"].round(1),
                           title="Attrition rate (%) by overtime", labels={"Attrition": "Attrition %"},
                           color_discrete_sequence=["#7C3AED"]).update_layout(height=320), width="stretch")
