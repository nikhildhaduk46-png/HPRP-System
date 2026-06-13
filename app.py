"""
Streamlit Web App
Run:  streamlit run app\app.py

Extra dependencies:
    pip install reportlab
"""
import streamlit as st
import sys, os, math, io
from datetime import datetime
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.predict import predict_patient

st.set_page_config(page_title="Patient Risk Prediction", page_icon="🏥", layout="wide")
st.title("🏥 Hospital Patient Risk Prediction System")
st.markdown("Fill in the patient details and click **Predict** to get the risk level.")
st.divider()

# ── Session state for history ──────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []   # list of dicts


# ══════════════════════════════════════════════════════════════════════════
# HELPER: Risk Meter (pure SVG — no extra library)
# ══════════════════════════════════════════════════════════════════════════
def show_risk_meter(probability: float):
    pct = round(probability * 100, 1)

    if pct < 25:
        color, label, bg = "#2ecc71", "Low Risk",      "#eafaf1"
    elif pct < 50:
        color, label, bg = "#f39c12", "Moderate Risk", "#fef9e7"
    elif pct < 75:
        color, label, bg = "#e67e22", "High Risk",     "#fef5e7"
    else:
        color, label, bg = "#e74c3c", "Critical Risk", "#fdedec"

    cx, cy, r      = 150, 140, 100
    start_deg      = 210
    sweep_deg      = 240

    def polar(deg, radius=r):
        rad = math.radians(deg)
        return cx + radius * math.cos(rad), cy + radius * math.sin(rad)

    def arc(a1, a2, col, width=16):
        x1, y1 = polar(a1)
        x2, y2 = polar(a2)
        lg = 1 if (a2 - a1) > 180 else 0
        return (f'<path d="M{x1:.2f},{y1:.2f} A{r},{r} 0 {lg},1 {x2:.2f},{y2:.2f}" '
                f'fill="none" stroke="{col}" stroke-width="{width}" stroke-linecap="round"/>')

    bg_arc    = arc(start_deg, start_deg + sweep_deg, "#dfe6e9", 16)
    zone_arcs = (
        arc(start_deg +   0, start_deg +  60, "#2ecc71") +
        arc(start_deg +  60, start_deg + 120, "#f39c12") +
        arc(start_deg + 120, start_deg + 180, "#e67e22") +
        arc(start_deg + 180, start_deg + 240, "#e74c3c")
    )

    needle_deg = start_deg + (pct / 100) * sweep_deg
    nx, ny     = polar(needle_deg, r * 0.72) # type: ignore
    needle_svg = (
        f'<line x1="{cx}" y1="{cy}" x2="{nx:.2f}" y2="{ny:.2f}" '
        f'stroke="#2c3e50" stroke-width="3.5" stroke-linecap="round"/>'
        f'<circle cx="{cx}" cy="{cy}" r="8" fill="#2c3e50"/>'
        f'<circle cx="{cx}" cy="{cy}" r="4" fill="#ecf0f1"/>'
    )

    ticks = ""
    for val in [0, 25, 50, 75, 100]:
        deg = start_deg + (val / 100) * sweep_deg
        tx, ty = polar(deg, r + 22)
        ticks += (f'<text x="{tx:.1f}" y="{ty:.1f}" text-anchor="middle" '
                  f'font-size="11" fill="#95a5a6" font-family="sans-serif">{val}</text>')

    svg = (f'<svg width="300" height="185" viewBox="0 0 300 185" '
           f'style="display:block;margin:auto;overflow:visible;">'
           f'{bg_arc}{zone_arcs}{needle_svg}{ticks}</svg>')

    html = f"""
    <div style="background:{bg};border-radius:16px;padding:14px 10px 10px;
                border:1.5px solid {color}55;text-align:center;">
      {svg}
      <div style="font-size:34px;font-weight:800;color:{color};
                  margin-top:2px;letter-spacing:-1px;">{pct}%</div>
      <div style="font-size:14px;font-weight:700;color:{color};
                  letter-spacing:1.5px;margin-bottom:10px;">{label}</div>
      <div style="display:flex;border-radius:6px;overflow:hidden;height:8px;margin:0 12px 5px;">
        <div style="flex:1;background:#2ecc71;"></div>
        <div style="flex:1;background:#f39c12;"></div>
        <div style="flex:1;background:#e67e22;"></div>
        <div style="flex:1;background:#e74c3c;"></div>
      </div>
      <div style="display:flex;justify-content:space-between;
                  font-size:10px;color:#95a5a6;margin:0 12px;">
        <span>Low</span><span>Moderate</span><span>High</span><span>Critical</span>
      </div>
    </div>"""
    st.markdown(html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# HELPER: PDF Report Generator (uses reportlab)
# ══════════════════════════════════════════════════════════════════════════
def generate_pdf_report(patient_name, city, prob, prediction, patient_data, model_choice):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                        Table, TableStyle, HRFlowable)
        from reportlab.lib.enums import TA_CENTER, TA_LEFT

        buf    = io.BytesIO()
        doc    = SimpleDocTemplate(buf, pagesize=A4,
                                   leftMargin=2*cm, rightMargin=2*cm,
                                   topMargin=2*cm, bottomMargin=2*cm)
        styles = getSampleStyleSheet()
        story  = []

        pct = round(prob * 100, 1)
        if pct < 25:   risk_label, risk_color = "LOW RISK",      colors.HexColor("#27ae60")
        elif pct < 50: risk_label, risk_color = "MODERATE RISK", colors.HexColor("#f39c12")
        elif pct < 75: risk_label, risk_color = "HIGH RISK",     colors.HexColor("#e67e22")
        else:          risk_label, risk_color = "CRITICAL RISK", colors.HexColor("#e74c3c")

        # ── Title block ────────────────────────────────────────────────────
        title_style = ParagraphStyle("title", fontSize=20, fontName="Helvetica-Bold",
                                     alignment=TA_CENTER, spaceAfter=4,
                                     textColor=colors.HexColor("#2c3e50"))
        sub_style   = ParagraphStyle("sub",   fontSize=11, fontName="Helvetica",
                                     alignment=TA_CENTER, spaceAfter=2,
                                     textColor=colors.HexColor("#7f8c8d"))

        story.append(Paragraph("🏥 Hospital Patient Risk Prediction", title_style))
        story.append(Paragraph("Clinical Risk Assessment Report", sub_style))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%d %B %Y, %I:%M %p')}", sub_style))
        story.append(HRFlowable(width="100%", thickness=1.5,
                                color=colors.HexColor("#2c3e50"), spaceAfter=14))

        # ── Patient info ───────────────────────────────────────────────────
        heading = ParagraphStyle("h2", fontSize=13, fontName="Helvetica-Bold",
                                 textColor=colors.HexColor("#2c3e50"), spaceAfter=6)
        story.append(Paragraph("Patient Information", heading))

        info_data = [
            ["Patient Name", patient_name or "—", "City",  city or "—"],
            ["Model Used",   model_choice,         "Date",  datetime.now().strftime("%d/%m/%Y")],
        ]
        info_tbl = Table(info_data, colWidths=[3.8*cm, 6*cm, 3.8*cm, 6*cm])
        info_tbl.setStyle(TableStyle([
            ("BACKGROUND",  (0,0), (0,-1), colors.HexColor("#eaf0fb")),
            ("BACKGROUND",  (2,0), (2,-1), colors.HexColor("#eaf0fb")),
            ("FONTNAME",    (0,0), (-1,-1), "Helvetica"),
            ("FONTNAME",    (0,0), (0,-1),  "Helvetica-Bold"),
            ("FONTNAME",    (2,0), (2,-1),  "Helvetica-Bold"),
            ("FONTSIZE",    (0,0), (-1,-1), 10),
            ("GRID",        (0,0), (-1,-1), 0.5, colors.HexColor("#bdc3c7")),
            ("ROWBACKGROUNDS", (0,0), (-1,-1), [colors.white, colors.HexColor("#f8f9fa")]),
            ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
            ("PADDING",     (0,0), (-1,-1), 6),
        ]))
        story.append(info_tbl)
        story.append(Spacer(1, 14))

        # ── Risk result box ────────────────────────────────────────────────
        story.append(Paragraph("Risk Assessment Result", heading))
        result_data = [
            ["Risk Level", risk_label],
            ["Probability", f"{prob:.2%}"],
            ["Prediction", "HIGH RISK" if prediction == 1 else "LOW RISK"],
        ]
        result_tbl = Table(result_data, colWidths=[5*cm, 14.6*cm])
        result_tbl.setStyle(TableStyle([
            ("BACKGROUND",  (0,0), (0,-1), colors.HexColor("#2c3e50")),
            ("TEXTCOLOR",   (0,0), (0,-1), colors.white),
            ("BACKGROUND",  (1,0), (1,0),  colors.HexColor("#f0fff4") if pct < 25 else
                                           colors.HexColor("#fffbf0") if pct < 50 else
                                           colors.HexColor("#fff5f0") if pct < 75 else
                                           colors.HexColor("#fff0f0")),
            ("TEXTCOLOR",   (1,0), (1,0),  risk_color),
            ("FONTNAME",    (0,0), (-1,-1), "Helvetica-Bold"),
            ("FONTSIZE",    (0,0), (-1,-1), 11),
            ("FONTSIZE",    (1,0), (1,0),  15),
            ("GRID",        (0,0), (-1,-1), 0.5, colors.HexColor("#bdc3c7")),
            ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
            ("PADDING",     (0,0), (-1,-1), 8),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f8f9fa")]),
        ]))
        story.append(result_tbl)
        story.append(Spacer(1, 14))

        # ── Clinical data tables ───────────────────────────────────────────
        sections = {
            "Demographics & Vitals": [
                ("Age",           patient_data.get("Age"),            "years"),
                ("Gender",        "Male" if patient_data.get("Gender")==0 else "Female", ""),
                ("BMI",           patient_data.get("BMI"),            "kg/m2"),
                ("Systolic BP",   patient_data.get("Systolic_BP"),    "mmHg"),
                ("Diastolic BP",  patient_data.get("Diastolic_BP"),   "mmHg"),
                ("Heart Rate",    patient_data.get("Heart_Rate_bpm"), "bpm"),
                ("Temperature",   patient_data.get("Temperature_C"),  "C"),
            ],
            "Blood Values": [
                ("Blood Glucose", patient_data.get("Blood_Glucose_mgdL"), "mg/dL",),
                ("SpO2",          patient_data.get("SpO2_%"),          "%"),
                ("WBC",           patient_data.get("WBC_x10e3"),       "x10^3"),
                ("RBC",           patient_data.get("RBC_x10e6"),       "x10^6"),
                ("Hemoglobin",    patient_data.get("Hemoglobin_gdL"),  "g/dL"),
                ("Hematocrit",    patient_data.get("Hematocrit_%"),    "%"),
                ("Platelets",     patient_data.get("Platelets_x10e3"),"x10^3"),
            ],
            "Lab Results & Clinical Info": [
                ("Sodium",        patient_data.get("Sodium_mEqL"),    "mEq/L"),
                ("Potassium",     patient_data.get("Potassium_mEqL"), "mEq/L"),
                ("Creatinine",    patient_data.get("Creatinine_mgdL"),"mg/dL"),
                ("BUN",           patient_data.get("BUN_mgdL"),       "mg/dL"),
                ("ALT",           patient_data.get("ALT_UL"),         "U/L"),
                ("AST",           patient_data.get("AST_UL"),         "U/L"),
                ("Length of Stay",patient_data.get("Length_of_Stay_days"), "days"),
                ("Prev. Admissions", patient_data.get("Previous_Admissions"), ""),
                ("Diagnosis Code",patient_data.get("Diagnosis_Code"), ""),
            ],
        }

        for sec_title, rows in sections.items():
            story.append(Paragraph(sec_title, heading))
            tbl_data = [["Parameter", "Value", "Unit"]]
            for param, val, unit in rows:
                tbl_data.append([param, str(val) if val is not None else "—", unit])
            tbl = Table(tbl_data, colWidths=[7*cm, 6*cm, 6.6*cm])
            tbl.setStyle(TableStyle([
                ("BACKGROUND",   (0,0), (-1,0),  colors.HexColor("#2c3e50")),
                ("TEXTCOLOR",    (0,0), (-1,0),  colors.white),
                ("FONTNAME",     (0,0), (-1,0),  "Helvetica-Bold"),
                ("FONTNAME",     (0,1), (-1,-1), "Helvetica"),
                ("FONTSIZE",     (0,0), (-1,-1), 10),
                ("GRID",         (0,0), (-1,-1), 0.5, colors.HexColor("#bdc3c7")),
                ("ROWBACKGROUNDS",(0,1),(-1,-1), [colors.white, colors.HexColor("#f8f9fa")]),
                ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
                ("PADDING",      (0,0), (-1,-1), 6),
            ]))
            story.append(tbl)
            story.append(Spacer(1, 12))

        # ── Disclaimer ────────────────────────────────────────────────────
        story.append(HRFlowable(width="100%", thickness=0.5,
                                color=colors.HexColor("#bdc3c7"), spaceAfter=6))
        disc = ParagraphStyle("disc", fontSize=8, fontName="Helvetica",
                              textColor=colors.HexColor("#95a5a6"), alignment=TA_CENTER)
        story.append(Paragraph(
            "This report is generated by an AI model for clinical decision support only. "
            "It is not a substitute for professional medical advice, diagnosis, or treatment.", disc))

        doc.build(story)
        buf.seek(0)
        return buf

    except ImportError:
        return None


# ══════════════════════════════════════════════════════════════════════════
# HELPER: Risk History Chart (pure HTML/SVG sparkline)
# ══════════════════════════════════════════════════════════════════════════
def show_history_chart(history):
    if len(history) < 1:
        st.info("No predictions yet. Make a prediction to see history here.")
        return

    # Summary stats
    scores = [h["pct"] for h in history]
    high_count = sum(1 for h in history if h["pct"] >= 50)
    avg_score  = round(sum(scores) / len(scores), 1)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Predictions", len(history))
    c2.metric("High / Critical",   high_count)
    c3.metric("Avg Risk Score",    f"{avg_score}%")

    st.markdown("---")

    # ── Sparkline chart ────────────────────────────────────────────────────
    W, H  = 640, 160
    pad_l, pad_r, pad_t, pad_b = 40, 20, 20, 30
    n     = len(history)
    x_gap = (W - pad_l - pad_r) / max(n - 1, 1)

    def sx(i):   return pad_l + i * x_gap
    def sy(pct): return pad_t + (H - pad_t - pad_b) * (1 - pct / 100)

    def dot_color(p):
        if p < 25:  return "#2ecc71"
        if p < 50:  return "#f39c12"
        if p < 75:  return "#e67e22"
        return "#e74c3c"

    # Polyline points
    pts = " ".join(f"{sx(i):.1f},{sy(h['pct']):.1f}" for i, h in enumerate(history))

    # Zone bands
    bands = [
        (0,   25,  "#2ecc71"),
        (25,  50,  "#f39c12"),
        (50,  75,  "#e67e22"),
        (75,  100, "#e74c3c"),
    ]
    band_rects = ""
    for lo, hi, col in bands:
        y_top = sy(hi)
        y_bot = sy(lo)
        band_rects += (f'<rect x="{pad_l}" y="{y_top:.1f}" '
                       f'width="{W-pad_l-pad_r}" height="{y_bot-y_top:.1f}" '
                       f'fill="{col}" opacity="0.07"/>')

    # Y-axis labels
    y_labels = ""
    for val in [0, 25, 50, 75, 100]:
        y = sy(val)
        y_labels += (f'<line x1="{pad_l}" y1="{y:.1f}" x2="{W-pad_r}" y2="{y:.1f}" '
                     f'stroke="#dfe6e9" stroke-width="1"/>'
                     f'<text x="{pad_l-6}" y="{y+4:.1f}" text-anchor="end" '
                     f'font-size="10" fill="#95a5a6" font-family="sans-serif">{val}</text>')

    # Dots + tooltips (title tag)
    dots = ""
    for i, h in enumerate(history):
        col = dot_color(h["pct"])
        tip = f"{h['name']} | {h['pct']}% | {h['label']} | {h['time']}"
        dots += (f'<circle cx="{sx(i):.1f}" cy="{sy(h["pct"]):.1f}" r="5" '
                 f'fill="{col}" stroke="white" stroke-width="1.5">'
                 f'<title>{tip}</title></circle>')
        # X label (every entry if <=8, else every 2nd)
        if n <= 8 or i % 2 == 0:
            short = h["name"].split()[0][:8]
            dots += (f'<text x="{sx(i):.1f}" y="{H-4}" text-anchor="middle" '
                     f'font-size="9" fill="#95a5a6" font-family="sans-serif">{short}</text>')

    svg = f"""
    <svg width="100%" viewBox="0 0 {W} {H}"
         style="display:block;border-radius:12px;background:#fafafa;
                border:1px solid #eee;padding:4px;overflow:visible;">
      {band_rects}
      {y_labels}
      <polyline points="{pts}" fill="none" stroke="#7f8c8d"
                stroke-width="2" stroke-linejoin="round" stroke-dasharray="4 2" opacity="0.5"/>
      {dots}
    </svg>"""

    st.markdown(svg, unsafe_allow_html=True)

    # Legend
    st.markdown("""
    <div style="display:flex;gap:18px;margin-top:6px;font-size:11px;color:#95a5a6;">
      <span><span style="color:#2ecc71">●</span> Low (&lt;25%)</span>
      <span><span style="color:#f39c12">●</span> Moderate (25–50%)</span>
      <span><span style="color:#e67e22">●</span> High (50–75%)</span>
      <span><span style="color:#e74c3c">●</span> Critical (&gt;75%)</span>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── History table ──────────────────────────────────────────────────────
    st.markdown("**Recent Predictions**")
    for h in reversed(history):
        col = {"Low Risk":"🟢","Moderate Risk":"🟡","High Risk":"🟠","Critical Risk":"🔴"}.get(h["label"],"⚪")
        st.markdown(
            f"{col} &nbsp; **{h['name']}** — {h['label']} &nbsp;|&nbsp; "
            f"Score: **{h['pct']}%** &nbsp;|&nbsp; {h['time']}",
            unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# PAGE: Input form
# ══════════════════════════════════════════════════════════════════════════

patient_name = st.text_input("Patient Name", placeholder="e.g. John Doe")
city         = st.text_input("City",         placeholder="e.g. Surat")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Demographics")
    age    = st.number_input("Age",          18, 100, 55)
    gender = st.selectbox("Gender", ["Male", "Female"])
    gender = 0 if gender == "Male" else 1
    bmi    = st.number_input("BMI",          15.0, 55.0, 27.5, step=0.1)

    st.subheader("🫀 Vitals")
    systolic  = st.number_input("Systolic BP",    80,  200, 130)
    diastolic = st.number_input("Diastolic BP",   50,  120, 85)
    hr        = st.number_input("Heart Rate",     40,  150, 80)
    temp      = st.number_input("Temperature °C", 35.0, 42.0, 37.0, step=0.1)

with col2:
    st.subheader("🩸 Blood Values")
    glucose = st.number_input("Blood Glucose (mg/dL)", 50,  400, 110)
    spo2    = st.number_input("SpO2 (%)",              80.0,100.0, 97.0, step=0.1)
    wbc     = st.number_input("WBC (×10³)",            1.0, 30.0, 8.5,  step=0.1)
    rbc     = st.number_input("RBC (×10⁶)",            2.0, 7.0,  4.5,  step=0.01)
    hgb     = st.number_input("Hemoglobin (g/dL)",     5.0, 20.0, 13.5, step=0.1)
    hct     = st.number_input("Hematocrit (%)",        15.0, 60.0, 40.0, step=0.1)
    plt_    = st.number_input("Platelets (×10³)",      50,  500,  220)

with col3:
    st.subheader("🔬 Lab Results")
    sodium     = st.number_input("Sodium (mEq/L)",     125, 155, 138)
    potassium  = st.number_input("Potassium (mEq/L)",  2.5, 6.5, 4.0, step=0.1)
    creatinine = st.number_input("Creatinine (mg/dL)", 0.4, 10.0, 1.0, step=0.1)
    bun        = st.number_input("BUN (mg/dL)",        5,   80,  18)
    alt        = st.number_input("ALT (U/L)",          5,  200,  30)
    ast        = st.number_input("AST (U/L)",          5,  200,  30)

    st.subheader("📋 Clinical Info")
    los          = st.number_input("Length of Stay (days)", 1, 60, 5)
    prev_adm     = st.number_input("Previous Admissions",   0, 20,  1)
    diag_code    = st.number_input("Diagnosis Code (0-14)", 0, 14,  0)
    model_choice = st.selectbox("Select Model", [
        "RandomForest", "GradientBoosting",
        "DecisionTree", "LogisticRegression"])

st.divider()

# ══════════════════════════════════════════════════════════════════════════
# PREDICT BUTTON
# ══════════════════════════════════════════════════════════════════════════

if st.button("🔍 Predict Patient Risk", use_container_width=True):

    pulse_pressure = systolic - diastolic
    bmi_cat   = 0 if bmi < 18.5 else (1 if bmi < 25 else (2 if bmi < 30 else 3))
    age_group = 0 if age < 30  else (1 if age < 50 else (2 if age < 65 else 3))
    glc_risk  = 1 if glucose > 126    else 0
    kid_risk  = 1 if creatinine > 1.5 else 0
    low_oxy   = 1 if spo2 < 94        else 0
    high_bun  = 1 if bun > 25         else 0

    patient = {
        'Age': age, 'Gender': gender, 'BMI': bmi,
        'Heart_Rate_bpm': hr, 'Temperature_C': temp,
        'Blood_Glucose_mgdL': glucose, 'SpO2_%': spo2,
        'WBC_x10e3': wbc, 'RBC_x10e6': rbc,
        'Hemoglobin_gdL': hgb, 'Hematocrit_%': hct,
        'Platelets_x10e3': plt_, 'Sodium_mEqL': sodium,
        'Potassium_mEqL': potassium, 'Creatinine_mgdL': creatinine,
        'BUN_mgdL': bun, 'ALT_UL': alt, 'AST_UL': ast,
        'Length_of_Stay_days': los, 'Previous_Admissions': prev_adm,
        'Diagnosis_Code': diag_code,
        'Systolic_BP': systolic, 'Diastolic_BP': diastolic,
        'Pulse_Pressure': pulse_pressure, 'BMI_Category': bmi_cat,
        'Age_Group': age_group, 'Glucose_Risk': glc_risk,
        'Kidney_Risk': kid_risk, 'LowOxygen': low_oxy, 'HighBUN': high_bun
    }

    try:
        result     = predict_patient(patient, model_name=model_choice)
        prob       = result['probability']
        prediction = result['prediction']
        pct        = round(prob * 100, 1)

        if pct < 25:   risk_label = "Low Risk"
        elif pct < 50: risk_label = "Moderate Risk"
        elif pct < 75: risk_label = "High Risk"
        else:          risk_label = "Critical Risk"

        # Save to session history
        st.session_state.history.append({
            "name":  patient_name or "Unknown",
            "city":  city or "—",
            "pct":   pct,
            "label": risk_label,
            "prob":  prob,
            "prediction": prediction,
            "model": model_choice,
            "time":  datetime.now().strftime("%d %b %H:%M"),
            "patient_data": patient,
        })

        st.divider()

        # ── Result layout ──────────────────────────────────────────────────
        r_col1, r_col2 = st.columns([1, 1.5])

        with r_col1:
            st.markdown("#### 📊 Risk Meter")
            show_risk_meter(prob)

            # PDF download button
            st.markdown("<br>", unsafe_allow_html=True)
            pdf_buf = generate_pdf_report(
                patient_name, city, prob, prediction, patient, model_choice)
            if pdf_buf:
                fname = f"risk_report_{(patient_name or 'patient').replace(' ','_')}.pdf"
                st.download_button(
                    label="📄 Download PDF Report",
                    data=pdf_buf,
                    file_name=fname,
                    mime="application/pdf",
                    use_container_width=True,
                )
            else:
                st.caption("💡 Install `reportlab` to enable PDF download: "
                           "`pip install reportlab`")

        with r_col2:
            st.markdown("#### 🩺 Prediction Result")
            if prediction == 1:
                st.error(f"### 🔴 HIGH RISK\nProbability: **{prob:.2%}**")
                st.warning("⚠️ Immediate medical attention recommended.")
            else:
                st.success(f"### 🟢 LOW RISK\nProbability of High Risk: **{prob:.2%}**")
                st.info("✅ Patient appears stable. Continue routine monitoring.")

            st.markdown("---")
            st.markdown("#### 📋 Key Vitals Summary")

            k1, k2, k3 = st.columns(3)
            k1.metric("BP (mmHg)",  f"{systolic}/{diastolic}",
                      delta="High" if systolic > 140 else "Normal",
                      delta_color="inverse" if systolic > 140 else "normal")
            k2.metric("SpO2",       f"{spo2:.1f}%",
                      delta="Low"  if spo2 < 94    else "Normal",
                      delta_color="inverse" if spo2 < 94 else "normal")
            k3.metric("Glucose",    f"{glucose} mg/dL",
                      delta="High" if glucose > 126 else "Normal",
                      delta_color="inverse" if glucose > 126 else "normal")

            k4, k5, k6 = st.columns(3)
            k4.metric("Heart Rate", f"{hr} bpm",
                      delta="High" if hr > 100 else ("Low" if hr < 60 else "Normal"),
                      delta_color="inverse" if hr > 100 or hr < 60 else "normal")
            k5.metric("Creatinine", f"{creatinine} mg/dL",
                      delta="High" if creatinine > 1.5 else "Normal",
                      delta_color="inverse" if creatinine > 1.5 else "normal")
            k6.metric("Hemoglobin", f"{hgb} g/dL",
                      delta="Low"  if hgb < 11 else "Normal",
                      delta_color="inverse" if hgb < 11 else "normal")

    except FileNotFoundError:
        st.error("❌ Models not found. Please run  `python main.py`  first.")

# ══════════════════════════════════════════════════════════════════════════
# RISK HISTORY SECTION (always visible at bottom)
# ══════════════════════════════════════════════════════════════════════════

st.divider()
st.markdown("## 📈 Risk History Dashboard")
show_history_chart(st.session_state.history)