"""
=============================================================================
app.py
Wire Manufacturing AI — Promise Date Dashboard (Streamlit UI)
=============================================================================
HOW TO RUN:
    pip install streamlit
    streamlit run app.py

FEATURES:
    • Manual order entry form (sidebar)
    • CSV upload for batch / single-row orders
    • Step-by-step transparent ML pipeline display
    • P10 / P50 / P90 probability timeline
    • OTIF risk gauge
    • SHAP-proxy top-3 feature drivers
    • Guardrail warnings

REQUIRES:
    01_train_models.py must have been run first (models/ folder must exist)
=============================================================================
"""

import os
import sys
import json
import csv
import io
from datetime import datetime, date

import streamlit as st
import numpy as np
import pandas as pd

# ── Make sure engine.py is importable from same folder ────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Wire Manufacturing AI — Promise Date Engine",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ────────────────────────────────── */
body { font-family: 'Segoe UI', sans-serif; }

/* ── Layer Cards ───────────────────────────── */
.layer-card {
    background: #1e2533;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
    border-left: 5px solid #4f8ef7;
}
.layer-card-green  { border-left-color: #22c55e; }
.layer-card-purple { border-left-color: #a855f7; }
.layer-card-orange { border-left-color: #f97316; }
.layer-card-blue   { border-left-color: #3b82f6; }
.layer-card-red    { border-left-color: #ef4444; }

.layer-title {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 6px;
}
.layer-heading {
    font-size: 1.15rem;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 14px;
}
.metric-row { display: flex; gap: 24px; flex-wrap: wrap; margin-top: 4px; }
.metric-box {
    background: #0f172a;
    border-radius: 8px;
    padding: 10px 18px;
    min-width: 160px;
}
.metric-label { font-size: 0.72rem; color: #64748b; text-transform: uppercase; letter-spacing: 1px; }
.metric-value { font-size: 1.4rem; font-weight: 700; color: #f8fafc; margin-top: 2px; }
.metric-sub   { font-size: 0.78rem; color: #94a3b8; margin-top: 2px; }
.badge-green  { background: #16a34a; color: #dcfce7; border-radius: 6px; padding: 2px 10px; font-size: 0.82rem; font-weight: 700; }
.badge-yellow { background: #ca8a04; color: #fef9c3; border-radius: 6px; padding: 2px 10px; font-size: 0.82rem; font-weight: 700; }
.badge-red    { background: #dc2626; color: #fee2e2; border-radius: 6px; padding: 2px 10px; font-size: 0.82rem; font-weight: 700; }
.badge-blue   { background: #1d4ed8; color: #dbeafe; border-radius: 6px; padding: 2px 10px; font-size: 0.82rem; font-weight: 700; }

/* ── Final Promise Panel ───────────────────── */
.promise-panel {
    background: linear-gradient(135deg, #0f172a 60%, #1e2533);
    border: 2px solid #3b82f6;
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 20px;
    text-align: center;
}
.promise-label { font-size: 0.8rem; color: #94a3b8; letter-spacing: 2px; text-transform: uppercase; }
.promise-date  { font-size: 2.8rem; font-weight: 800; color: #60a5fa; margin: 6px 0; }
.promise-band  { font-size: 0.95rem; color: #64748b; }

/* ── Timeline ──────────────────────────────── */
.timeline-bar {
    background: #1e2533;
    border-radius: 10px;
    padding: 18px 20px;
    margin-bottom: 16px;
}
.tl-label { font-size: 0.72rem; color: #64748b; text-transform: uppercase; letter-spacing: 1px; }
.tl-row { display: flex; align-items: center; gap: 12px; margin-top: 4px; }
.tl-tag { border-radius: 6px; padding: 4px 12px; font-size: 0.85rem; font-weight: 700; }

/* ── SHAP Bars ─────────────────────────────── */
.shap-row { display: flex; align-items: center; gap: 10px; margin: 6px 0; }
.shap-label { width: 200px; font-size: 0.82rem; color: #94a3b8; }
.shap-bar-pos { height: 14px; background: #ef4444; border-radius: 4px; }
.shap-bar-neg { height: 14px; background: #22c55e; border-radius: 4px; }
.shap-val { font-size: 0.82rem; font-weight: 700; color: #f1f5f9; }
.shap-desc { font-size: 0.75rem; color: #64748b; }

/* ── Warning box ───────────────────────────── */
.warn-box {
    background: #431407;
    border-left: 4px solid #f97316;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
    color: #fed7aa;
    font-size: 0.87rem;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# LOAD ENGINE  (cached so it only loads once)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading ML models…")
def load_engine():
    from engine import PromiseDateEngine
    return PromiseDateEngine()


# ══════════════════════════════════════════════════════════════════════════════
# HELPER RENDERING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def render_layer1(report):
    adj_pct = report.run_rate_adjustment_pct
    pct_sign = "+" if adj_pct >= 0 else ""
    st.markdown(f"""
    <div class="layer-card layer-card-green">
      <div class="layer-title">LAYER 1</div>
      <div class="layer-heading">🔧 Run Rate Refiner <small style="font-weight:400;font-size:0.8rem;color:#64748b">(Ridge Regression)</small></div>
      <div class="metric-row">
        <div class="metric-box">
          <div class="metric-label">Nominal Mill Rate</div>
          <div class="metric-value">{report.nominal_run_rate_kghr:.0f} <span style="font-size:1rem;color:#64748b">kg/hr</span></div>
          <div class="metric-sub">From mill spec table</div>
        </div>
        <div class="metric-box">
          <div class="metric-label">ML-Adjusted Rate</div>
          <div class="metric-value">{report.adjusted_run_rate_kghr:.0f} <span style="font-size:1rem;color:#64748b">kg/hr</span></div>
          <div class="metric-sub">After diameter, item-type & seasonal adjustments</div>
        </div>
        <div class="metric-box">
          <div class="metric-label">Adjustment</div>
          <div class="metric-value" style="color:{'#22c55e' if adj_pct >= 0 else '#ef4444'}">{pct_sign}{adj_pct:.1f}%</div>
          <div class="metric-sub">vs. nominal rate</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_layer2(report):
    q_pct = report.queue_contention_pct
    pct_sign = "+" if q_pct >= 0 else ""
    st.markdown(f"""
    <div class="layer-card layer-card-purple">
      <div class="layer-title">LAYER 2</div>
      <div class="layer-heading">📦 Multi-Order Interaction Adjuster <small style="font-weight:400;font-size:0.8rem;color:#64748b">(Random Forest)</small></div>
      <div class="metric-row">
        <div class="metric-box">
          <div class="metric-label">Base Lead Time</div>
          <div class="metric-value">{report.base_lead_time_days:.1f} <span style="font-size:1rem;color:#64748b">days</span></div>
          <div class="metric-sub">qty ÷ (adj.rate × 8 hrs)</div>
        </div>
        <div class="metric-box">
          <div class="metric-label">Queue-Adjusted</div>
          <div class="metric-value">{report.queue_adjusted_days:.1f} <span style="font-size:1rem;color:#64748b">days</span></div>
          <div class="metric-sub">After multi-order contention</div>
        </div>
        <div class="metric-box">
          <div class="metric-label">Queue Contention</div>
          <div class="metric-value" style="color:{'#ef4444' if q_pct > 0 else '#22c55e'}">{pct_sign}{q_pct:.1f}%</div>
          <div class="metric-sub">vs. base lead time</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_layer3(report):
    st.markdown(f"""
    <div class="layer-card layer-card-orange">
      <div class="layer-title">LAYER 3</div>
      <div class="layer-heading">⚙️ Setup Time Cascading Calculator <small style="font-weight:400;font-size:0.8rem;color:#64748b">(Gradient Boosting)</small></div>
      <div class="metric-row">
        <div class="metric-box">
          <div class="metric-label">Random Sequence Setup</div>
          <div class="metric-value">{report.random_setup_min:.0f} <span style="font-size:1rem;color:#64748b">min</span></div>
          <div class="metric-sub">Heuristic rule-based estimate</div>
        </div>
        <div class="metric-box">
          <div class="metric-label">ML-Optimized Setup</div>
          <div class="metric-value">{report.ml_optimized_setup_min:.0f} <span style="font-size:1rem;color:#64748b">min</span></div>
          <div class="metric-sub">After smart sequencing</div>
        </div>
        <div class="metric-box">
          <div class="metric-label">Savings</div>
          <div class="metric-value" style="color:#22c55e">{report.setup_savings_pct:.1f}%</div>
          <div class="metric-sub">Setup time reduction</div>
        </div>
        <div class="metric-box">
          <div class="metric-label">Day Adjustment</div>
          <div class="metric-value" style="color:{'#22c55e' if report.setup_adjustment_days <= 0 else '#ef4444'}">{report.setup_adjustment_days:+.2f} <span style="font-size:1rem;color:#64748b">days</span></div>
          <div class="metric-sub">Applied to promise date</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_layer4(report):
    exceed_pct = report.prob_exceed_42days * 100
    guard_badge = (
        '<span class="badge-red">⚠ EXCEEDS 42-DAY LIMIT</span>'
        if report.guardrail_flag else
        '<span class="badge-green">✓ PASS</span>'
    )
    st.markdown(f"""
    <div class="layer-card layer-card-blue">
      <div class="layer-title">LAYER 4</div>
      <div class="layer-heading">📅 Probabilistic Promise Date Predictor <small style="font-weight:400;font-size:0.8rem;color:#64748b">(Random Forest — 500 trees)</small></div>
      <div class="metric-row">
        <div class="metric-box">
          <div class="metric-label">P10 — Optimistic</div>
          <div class="metric-value">{report.p10_days:.0f} <span style="font-size:1rem;color:#64748b">days</span></div>
          <div class="metric-sub">{report.promise_date_p10}</div>
        </div>
        <div class="metric-box" style="border: 1px solid #3b82f6;">
          <div class="metric-label">P50 — Commit Date ★</div>
          <div class="metric-value" style="color:#60a5fa">{report.p50_days:.0f} <span style="font-size:1rem;color:#64748b">days</span></div>
          <div class="metric-sub" style="color:#93c5fd;font-weight:700">{report.promise_date_p50}</div>
        </div>
        <div class="metric-box">
          <div class="metric-label">P90 — Conservative</div>
          <div class="metric-value">{report.p90_days:.0f} <span style="font-size:1rem;color:#64748b">days</span></div>
          <div class="metric-sub">{report.promise_date_p90}</div>
        </div>
        <div class="metric-box">
          <div class="metric-label">Prob. Exceed 42 days</div>
          <div class="metric-value" style="color:{'#ef4444' if exceed_pct > 30 else '#f97316' if exceed_pct > 10 else '#22c55e'}">{exceed_pct:.1f}%</div>
          <div class="metric-sub">6-Week Guardrail: {guard_badge}</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_layer5(report):
    prob_pct = report.otif_risk_probability * 100
    label    = report.otif_risk_label
    badge    = (
        '<span class="badge-green">LOW RISK</span>'   if label == "LOW"    else
        '<span class="badge-yellow">MEDIUM RISK</span>' if label == "MEDIUM" else
        '<span class="badge-red">HIGH RISK</span>'
    )
    # Build a simple text gauge bar
    filled = int(prob_pct / 5)  # 0-20 blocks
    bar_color = "#22c55e" if label == "LOW" else "#ca8a04" if label == "MEDIUM" else "#dc2626"
    bar_html  = f"<span style='color:{bar_color}'>{'█' * filled}{'░' * (20 - filled)}</span>"

    st.markdown(f"""
    <div class="layer-card layer-card-red">
      <div class="layer-title">LAYER 5</div>
      <div class="layer-heading">🎯 OTIF Risk Scorer <small style="font-weight:400;font-size:0.8rem;color:#64748b">(Random Forest Classifier)</small></div>
      <div class="metric-row">
        <div class="metric-box">
          <div class="metric-label">Risk Probability</div>
          <div class="metric-value">{prob_pct:.1f}%</div>
          <div class="metric-sub">of missing On-Time delivery</div>
        </div>
        <div class="metric-box" style="min-width:320px">
          <div class="metric-label">Risk Level</div>
          <div class="metric-value">{badge}</div>
          <div class="metric-sub" style="font-family:monospace;margin-top:8px">{bar_html} {prob_pct:.0f}%</div>
        </div>
      </div>
      <div style="margin-top:12px;font-size:0.78rem;color:#475569;">
        Threshold: &lt;35% = LOW &nbsp;|&nbsp; 35–65% = MEDIUM &nbsp;|&nbsp; &gt;65% = HIGH
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_shap(report):
    if not report.shap_top3:
        return
    max_impact = max(abs(v) for _, v, _ in report.shap_top3) or 1

    rows_html = ""
    for feat, val, desc in report.shap_top3:
        bar_width = int(abs(val) / max_impact * 180)
        bar_class = "shap-bar-pos" if val > 0 else "shap-bar-neg"
        sign      = "+" if val > 0 else ""
        color     = "#ef4444" if val > 0 else "#22c55e"
        label     = feat.replace("_", " ").title()
        rows_html += f"""
        <div class="shap-row">
          <div class="shap-label">{label}</div>
          <div class="{bar_class}" style="width:{bar_width}px"></div>
          <div class="shap-val" style="color:{color}">{sign}{val:.2f} days</div>
        </div>
        <div style="margin-left:210px;margin-top:-4px;margin-bottom:8px" class="shap-desc">{desc}</div>
        """

    st.markdown(f"""
    <div style="background:#1e2533;border-radius:12px;padding:20px 24px;border-left:5px solid #a855f7">
      <div class="layer-title">EXPLAINABILITY</div>
      <div class="layer-heading">🔍 Top 3 Feature Drivers (SHAP-proxy)</div>
      <div style="margin-bottom:6px;font-size:0.78rem;color:#64748b">
        Red = adding days (delay) &nbsp;|&nbsp; Green = reducing days (accelerating)
      </div>
      {rows_html}
    </div>
    """, unsafe_allow_html=True)


def render_warnings(report):
    if not report.guardrail_warnings:
        return
    for w in report.guardrail_warnings:
        st.markdown(f'<div class="warn-box">⚠ {w}</div>', unsafe_allow_html=True)


def render_promise_panel(report):
    label = report.otif_risk_label
    badge_color = "#16a34a" if label == "LOW" else "#ca8a04" if label == "MEDIUM" else "#dc2626"
    st.markdown(f"""
    <div class="promise-panel">
      <div class="promise-label">★ Final Promise Date (P50 Commit)</div>
      <div class="promise-date">{report.promise_date_p50}</div>
      <div class="promise-band">
        Confidence band: &nbsp;
        <b>{report.promise_date_p10}</b> &nbsp;→&nbsp; <b>{report.promise_date_p90}</b>
        &nbsp;&nbsp;|&nbsp;&nbsp;
        <span style="background:{badge_color};color:#fff;border-radius:6px;padding:3px 12px;font-weight:700">
          {label} OTIF RISK ({report.otif_risk_probability*100:.0f}%)
        </span>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_timeline_chart(report):
    """Draw a matplotlib-free visual timeline using Streamlit columns."""
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.markdown(f"""
        <div class="timeline-bar" style="border-left:4px solid #22c55e">
          <div class="tl-label">P10 — Optimistic (10th percentile)</div>
          <div class="tl-row">
            <div class="tl-tag" style="background:#166534;color:#dcfce7">🟢 {report.p10_days:.0f} days</div>
            <div style="font-size:0.9rem;color:#86efac;font-weight:600">{report.promise_date_p10}</div>
          </div>
          <div style="font-size:0.75rem;color:#4b7c5e;margin-top:6px">Best-case if everything runs smoothly</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="timeline-bar" style="border-left:4px solid #3b82f6;border: 2px solid #3b82f6;">
          <div class="tl-label">P50 ★ COMMIT DATE (Median)</div>
          <div class="tl-row">
            <div class="tl-tag" style="background:#1e3a8a;color:#dbeafe">🔵 {report.p50_days:.0f} days</div>
            <div style="font-size:0.9rem;color:#93c5fd;font-weight:700">{report.promise_date_p50}</div>
          </div>
          <div style="font-size:0.75rem;color:#3b5998;margin-top:6px">This is the date to promise to the customer</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="timeline-bar" style="border-left:4px solid #f97316">
          <div class="tl-label">P90 — Conservative (90th percentile)</div>
          <div class="tl-row">
            <div class="tl-tag" style="background:#7c2d12;color:#fed7aa">🟠 {report.p90_days:.0f} days</div>
            <div style="font-size:0.9rem;color:#fdba74;font-weight:600">{report.promise_date_p90}</div>
          </div>
          <div style="font-size:0.75rem;color:#7c4a1a;margin-top:6px">Worst-case buffer for risk planning</div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — ORDER INPUT FORM
# ══════════════════════════════════════════════════════════════════════════════

ITEM_TYPES    = ["WHF","PHF","WSA","WCH","WTE","WDG","WGA","WSP","WDU"]
DIAMETERS     = [1.2, 1.6, 2.0, 2.4, 2.8, 3.2]
PACKAGE_CODES = ["KH","DJ","MC","L1","L2","KH1","SP1","SP2","PP"]
MILLS         = ["Mill_A","Mill_B","Mill_C","Mill_D","Mill_E"]

ITEM_TYPE_NAMES = {
    "WHF": "WHF — Wire Hard Fine",
    "PHF": "PHF — Plain Hard Fine",
    "WSA": "WSA — Wire Soft Annealed",
    "WCH": "WCH — Wire Chain",
    "WTE": "WTE — Wire Tempered",
    "WDG": "WDG — Wire Drawn Galvanized",
    "WGA": "WGA — Wire Galvanized Annealed",
    "WSP": "WSP — Wire Spring",
    "WDU": "WDU — Wire Drawn Uncoated",
}


def build_sidebar():
    st.sidebar.markdown("## 🏭 Sales Order Entry")
    st.sidebar.markdown("---")

    # ── Upload mode toggle ─────────────────────────────────────────────────────
    mode = st.sidebar.radio("Input Mode", ["📝 Manual Entry", "📤 CSV Upload"], horizontal=True)

    order = {}
    csv_orders = []
    uploaded_file = None

    if mode == "📤 CSV Upload":
        st.sidebar.markdown("**Upload a CSV file** with one or more orders.")
        uploaded_file = st.sidebar.file_uploader(
            "Upload CSV", type=["csv"],
            help="Each row = one order. Column names must match the field names shown in manual form."
        )
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.sidebar.success(f"✓ {len(df)} order(s) loaded")
                csv_orders = df.to_dict(orient="records")
            except Exception as e:
                st.sidebar.error(f"CSV parse error: {e}")
        return mode, order, csv_orders

    # ── MANUAL ENTRY FORM ─────────────────────────────────────────────────────
    with st.sidebar.form("order_form"):
        st.markdown("#### Order Identity")
        order["order_id"] = st.text_input(
            "Order ID", value=f"SO-{datetime.today().strftime('%Y%m%d')}-001"
        )
        order["order_date"] = st.date_input("Order Date", value=date.today()).strftime("%Y-%m-%d")

        st.markdown("#### Product Details")
        it_display = st.selectbox(
            "Item Type",
            options=list(ITEM_TYPE_NAMES.keys()),
            format_func=lambda x: ITEM_TYPE_NAMES[x]
        )
        order["item_type"] = it_display

        order["wire_diameter_mm"] = st.selectbox(
            "Wire Diameter (mm)", options=DIAMETERS, index=2
        )
        order["qty_ordered_kg"] = st.number_input(
            "Quantity Ordered (kg)", min_value=100.0, max_value=100000.0,
            value=10000.0, step=500.0
        )
        order["package_code"] = st.selectbox("Package Code", options=PACKAGE_CODES)
        order["mill_assigned"] = st.selectbox("Mill Assigned", options=MILLS)

        st.markdown("#### SO Structure")
        order["num_lines_per_SO"] = st.number_input(
            "Number of Lines in SO", min_value=1, max_value=20, value=3
        )
        order["total_qty_per_SO"] = st.number_input(
            "Total Qty across all SO Lines (kg)",
            min_value=100.0, max_value=1_000_000.0,
            value=float(int(order["qty_ordered_kg"]) * 2), step=1000.0
        )
        order["unique_diameters_per_SO"] = st.number_input(
            "Unique Wire Diameters in SO", min_value=1, max_value=6, value=2
        )

        st.markdown("#### Queue State (Current Week)")
        order["concurrent_orders_this_week"] = st.number_input(
            "Active Orders This Week", min_value=0, max_value=50, value=10
        )
        order["concurrent_same_mill_orders"] = st.number_input(
            "Orders on Same Mill", min_value=0, max_value=20, value=3
        )
        order["week_total_load_kg"] = st.number_input(
            "Total Backlog This Week (kg)",
            min_value=0.0, max_value=2_000_000.0, value=120000.0, step=5000.0
        )
        order["diameter_group_load_kg"] = st.number_input(
            "Same-Diameter Group Backlog (kg)",
            min_value=0.0, max_value=2_000_000.0, value=200000.0, step=5000.0
        )
        order["previous_order_diameter_on_mill"] = st.selectbox(
            "Previous Order Diameter on Mill (mm)",
            options=DIAMETERS,
            index=DIAMETERS.index(order["wire_diameter_mm"])
        )

        submitted = st.form_submit_button("🚀 Calculate Promise Date", use_container_width=True)

    # Derive month and day-of-week
    try:
        dt = datetime.strptime(order["order_date"], "%Y-%m-%d")
        order["order_month"]       = dt.month
        order["order_day_of_week"] = dt.weekday()
    except Exception:
        order["order_month"]       = datetime.today().month
        order["order_day_of_week"] = 0

    if not submitted:
        return mode, None, []

    return mode, order, []


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PAGE
# ══════════════════════════════════════════════════════════════════════════════

def render_header():
    st.markdown("""
    <div style="background:linear-gradient(90deg,#0f172a,#1e2533);padding:24px 32px;border-radius:12px;margin-bottom:24px;border-bottom:3px solid #3b82f6">
      <h1 style="margin:0;font-size:1.8rem;color:#f8fafc">
        🏭 Wire Manufacturing AI — Promise Date Engine
      </h1>
      <p style="margin:6px 0 0;color:#64748b;font-size:0.9rem">
        5-Layer Machine Learning Pipeline &nbsp;|&nbsp;
        Run Rate Refiner → Multi-Order Adjuster → Setup Optimizer → Probabilistic Predictor → OTIF Scorer
      </p>
    </div>
    """, unsafe_allow_html=True)


def render_pipeline_diagram():
    """Static pipeline diagram using colored boxes."""
    cols = st.columns(5)
    layers = [
        ("L1","#22c55e","Run Rate\nRefiner","Ridge\nRegression"),
        ("L2","#a855f7","Multi-Order\nAdjuster","Random\nForest"),
        ("L3","#f97316","Setup Time\nCalculator","Gradient\nBoosting"),
        ("L4","#3b82f6","Promise Date\nPredictor","RF Quantile\n500 trees"),
        ("L5","#ef4444","OTIF Risk\nScorer","RF Classifier"),
    ]
    for i, (col, (tag, color, name, model)) in enumerate(zip(cols, layers)):
        with col:
            st.markdown(f"""
            <div style="background:#1e2533;border:2px solid {color};border-radius:10px;padding:14px 10px;text-align:center">
              <div style="background:{color};color:#fff;border-radius:6px;padding:2px 8px;font-size:0.7rem;font-weight:700;margin-bottom:8px;display:inline-block">{tag}</div>
              <div style="color:#f1f5f9;font-weight:700;font-size:0.85rem;line-height:1.3">{name}</div>
              <div style="color:#64748b;font-size:0.72rem;margin-top:4px">{model}</div>
            </div>
            {"<div style='text-align:center;font-size:1.2rem;color:#334155;margin-top:10px'>→</div>" if i < 4 else ""}
            """, unsafe_allow_html=True)


def render_report(report):
    """Full pipeline transparent display."""
    # Final promise panel (shown first for visibility)
    render_promise_panel(report)

    # Timeline
    st.markdown("### 📊 Probability Timeline")
    render_timeline_chart(report)

    # Layer-by-layer breakdown
    st.markdown("### 🔬 Pipeline Transparency — All 5 Layers")

    render_layer1(report)
    render_layer2(report)
    render_layer3(report)
    render_layer4(report)
    render_layer5(report)

    # SHAP explainability
    st.markdown("### 🔍 Explainability")
    render_shap(report)

    # Guardrail warnings
    if report.guardrail_warnings:
        st.markdown("### ⚠ Guardrail Warnings")
        render_warnings(report)

    # JSON export
    with st.expander("📋 View Raw Report (JSON)"):
        report_dict = {
            "order_id":               report.order_id,
            "order_date":             report.order_date,
            "layer1": {
                "nominal_run_rate_kghr":   report.nominal_run_rate_kghr,
                "adjusted_run_rate_kghr":  report.adjusted_run_rate_kghr,
                "adjustment_pct":          report.run_rate_adjustment_pct,
            },
            "layer2": {
                "base_lead_time_days":    report.base_lead_time_days,
                "queue_adjusted_days":    report.queue_adjusted_days,
                "queue_contention_pct":   report.queue_contention_pct,
            },
            "layer3": {
                "random_setup_min":        report.random_setup_min,
                "ml_optimized_setup_min":  report.ml_optimized_setup_min,
                "setup_savings_pct":       report.setup_savings_pct,
                "setup_adjustment_days":   report.setup_adjustment_days,
            },
            "layer4": {
                "p10_days":          report.p10_days,
                "p50_days":          report.p50_days,
                "p90_days":          report.p90_days,
                "promise_date_p10":  report.promise_date_p10,
                "promise_date_p50":  report.promise_date_p50,
                "promise_date_p90":  report.promise_date_p90,
                "prob_exceed_42d":   report.prob_exceed_42days,
                "guardrail_flag":    report.guardrail_flag,
            },
            "layer5": {
                "otif_risk_probability": report.otif_risk_probability,
                "otif_risk_label":       report.otif_risk_label,
            },
            "shap_top3": [
                {"feature": f, "impact_days": round(v, 3), "description": d}
                for f, v, d in report.shap_top3
            ],
            "guardrail_warnings": report.guardrail_warnings,
        }
        st.json(report_dict)

    # Download report button
    report_txt = f"""WIRE MANUFACTURING AI — PROMISE DATE REPORT
Order ID   : {report.order_id}
Order Date : {report.order_date}

LAYER 1 — RUN RATE REFINER
  Nominal Run Rate    : {report.nominal_run_rate_kghr:.1f} kg/hr
  Adjusted Run Rate   : {report.adjusted_run_rate_kghr:.1f} kg/hr  ({report.run_rate_adjustment_pct:+.1f}%)

LAYER 2 — MULTI-ORDER INTERACTION ADJUSTER
  Base Lead Time      : {report.base_lead_time_days:.1f} days
  Queue-Adjusted      : {report.queue_adjusted_days:.1f} days  ({report.queue_contention_pct:+.1f}% queue effect)

LAYER 3 — SETUP TIME CASCADING CALCULATOR
  Random Sequence     : {report.random_setup_min:.0f} min
  ML-Optimized Setup  : {report.ml_optimized_setup_min:.0f} min  ({report.setup_savings_pct:.1f}% savings)
  Day Adjustment      : {report.setup_adjustment_days:+.2f} days

LAYER 4 — PROBABILISTIC PROMISE DATE PREDICTOR
  P10 (Optimistic)    : {report.p10_days:.0f} days  → {report.promise_date_p10}
  P50 (Commit) ★      : {report.p50_days:.0f} days  → {report.promise_date_p50}
  P90 (Conservative)  : {report.p90_days:.0f} days  → {report.promise_date_p90}
  Prob. exceed 42 days: {report.prob_exceed_42days*100:.1f}%
  6-Week Guardrail    : {'EXCEEDS LIMIT ⚠' if report.guardrail_flag else 'PASS ✓'}

LAYER 5 — OTIF RISK SCORER
  Risk Probability    : {report.otif_risk_probability*100:.1f}%
  Risk Label          : {report.otif_risk_label}

TOP 3 FEATURE DRIVERS
""" + "\n".join(
        f"  {i+1}. {f.replace('_',' ').title():<30} {v:+.2f} days — {d}"
        for i, (f, v, d) in enumerate(report.shap_top3)
    ) + f"""

GUARDRAIL WARNINGS
""" + ("\n".join(f"  ⚠ {w}" for w in report.guardrail_warnings) or "  None") + f"""

★ FINAL PROMISE DATE : {report.promise_date_p50}
  Confidence Band    : {report.promise_date_p10}  →  {report.promise_date_p90}
  OTIF Risk          : {report.otif_risk_label} ({report.otif_risk_probability*100:.0f}%)
"""
    st.download_button(
        "⬇ Download Report (.txt)",
        data=report_txt,
        file_name=f"promise_report_{report.order_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
    )


def render_csv_results(reports):
    """Summary table for batch CSV results."""
    rows = []
    for r in reports:
        rows.append({
            "Order ID":         r.order_id,
            "Promise Date (P50)": r.promise_date_p50,
            "P10 Date":         r.promise_date_p10,
            "P90 Date":         r.promise_date_p90,
            "P50 Days":         r.p50_days,
            "OTIF Risk":        r.otif_risk_label,
            "Risk %":           f"{r.otif_risk_probability*100:.0f}%",
            "Guardrail":        "⚠ EXCEED" if r.guardrail_flag else "✓ PASS",
        })
    df = pd.DataFrame(rows)

    def color_risk(val):
        if val == "LOW":    return "background-color:#16a34a;color:#fff"
        if val == "MEDIUM": return "background-color:#ca8a04;color:#fff"
        if val == "HIGH":   return "background-color:#dc2626;color:#fff"
        return ""

    styled = df.style.applymap(color_risk, subset=["OTIF Risk"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Offer detailed view for each order
    st.markdown("#### Drill into an Order")
    selected_id = st.selectbox("Select Order ID for detailed view", options=[r.order_id for r in reports])
    if selected_id:
        sel_report = next(r for r in reports if r.order_id == selected_id)
        render_report(sel_report)


# ══════════════════════════════════════════════════════════════════════════════
# LANDING PAGE (shown when no order submitted yet)
# ══════════════════════════════════════════════════════════════════════════════

def render_landing():
    st.markdown("""
    <div style="text-align:center;padding:40px 20px">
      <div style="font-size:3rem;margin-bottom:16px">🏭</div>
      <h2 style="color:#f1f5f9;margin-bottom:8px">Wire Manufacturing AI Promise Engine</h2>
      <p style="color:#64748b;max-width:600px;margin:0 auto 24px">
        Enter a sales order in the sidebar (or upload a CSV) to run it through the
        full 5-layer ML pipeline and receive a transparent, explainable promise date.
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### How It Works")
    render_pipeline_diagram()

    st.markdown("""
    <div style="display:flex;gap:16px;margin-top:24px;flex-wrap:wrap">
      <div style="background:#1e2533;border-radius:10px;padding:16px;flex:1;min-width:200px;border-left:4px solid #22c55e">
        <div style="font-weight:700;color:#86efac;margin-bottom:6px">① Run Rate Refiner</div>
        <div style="font-size:0.82rem;color:#94a3b8">Adjusts mill throughput rate for wire diameter, item type, and seasonal factors using Ridge Regression</div>
      </div>
      <div style="background:#1e2533;border-radius:10px;padding:16px;flex:1;min-width:200px;border-left:4px solid #a855f7">
        <div style="font-weight:700;color:#d8b4fe;margin-bottom:6px">② Multi-Order Adjuster</div>
        <div style="font-size:0.82rem;color:#94a3b8">Accounts for queue contention, concurrent orders, and mill loading using Random Forest</div>
      </div>
      <div style="background:#1e2533;border-radius:10px;padding:16px;flex:1;min-width:200px;border-left:4px solid #f97316">
        <div style="font-weight:700;color:#fdba74;margin-bottom:6px">③ Setup Time Calculator</div>
        <div style="font-size:0.82rem;color:#94a3b8">Optimizes setup sequence for diameter transitions using Gradient Boosting — saves vs. random order</div>
      </div>
      <div style="background:#1e2533;border-radius:10px;padding:16px;flex:1;min-width:200px;border-left:4px solid #3b82f6">
        <div style="font-weight:700;color:#93c5fd;margin-bottom:6px">④ Promise Date Predictor</div>
        <div style="font-size:0.82rem;color:#94a3b8">500-tree Random Forest quantile model → P10 / P50 / P90 delivery dates with confidence intervals</div>
      </div>
      <div style="background:#1e2533;border-radius:10px;padding:16px;flex:1;min-width:200px;border-left:4px solid #ef4444">
        <div style="font-weight:700;color:#fca5a5;margin-bottom:6px">⑤ OTIF Risk Scorer</div>
        <div style="font-size:0.82rem;color:#94a3b8">Classifies delivery risk (LOW / MEDIUM / HIGH) using balanced Random Forest Classifier</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#0f172a;border-radius:10px;padding:16px 20px;margin-top:20px;border-left:4px solid #475569">
      <div style="font-weight:700;color:#f1f5f9;margin-bottom:8px">📋 Sample CSV Format</div>
      <code style="font-size:0.78rem;color:#94a3b8">
        order_id, item_type, wire_diameter_mm, qty_ordered_kg, package_code, mill_assigned, order_date,
        order_month, order_day_of_week, num_lines_per_SO, total_qty_per_SO, unique_diameters_per_SO,
        diameter_group_load_kg, concurrent_orders_this_week, concurrent_same_mill_orders,
        week_total_load_kg, previous_order_diameter_on_mill
      </code>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    render_header()

    # Load engine
    try:
        engine = load_engine()
    except Exception as e:
        st.error(f"""
        ❌ **Model files not found!**

        Run the training script first:
        ```
        python 01_train_models.py
        ```
        Then restart this app. Error: {e}
        """)
        st.stop()

    # Sidebar
    mode, order, csv_orders = build_sidebar()

    # ── CSV Upload Mode ────────────────────────────────────────────────────────
    if mode == "📤 CSV Upload":
        if csv_orders:
            st.markdown(f"### 📤 Batch Prediction — {len(csv_orders)} order(s)")
            reports = []
            prog = st.progress(0, text="Running predictions…")
            for i, row in enumerate(csv_orders):
                # Cast numeric fields
                for num_col in ["wire_diameter_mm","qty_ordered_kg","num_lines_per_SO",
                                 "total_qty_per_SO","unique_diameters_per_SO",
                                 "diameter_group_load_kg","concurrent_orders_this_week",
                                 "concurrent_same_mill_orders","week_total_load_kg",
                                 "previous_order_diameter_on_mill","order_month",
                                 "order_day_of_week"]:
                    if num_col in row and row[num_col] != "":
                        try:
                            row[num_col] = float(row[num_col])
                        except (ValueError, TypeError):
                            pass
                # Derive month/day if missing
                if "order_month" not in row or row.get("order_month") == "":
                    try:
                        dt = datetime.strptime(str(row.get("order_date","")), "%Y-%m-%d")
                        row["order_month"]       = dt.month
                        row["order_day_of_week"] = dt.weekday()
                    except Exception:
                        row["order_month"]       = datetime.today().month
                        row["order_day_of_week"] = 0
                try:
                    reports.append(engine.predict(row))
                except Exception as e:
                    st.warning(f"Order `{row.get('order_id','?')}` failed: {e}")
                prog.progress((i + 1) / len(csv_orders), text=f"Processed {i+1}/{len(csv_orders)}")

            prog.empty()
            render_csv_results(reports)
        else:
            render_landing()
        return

    # ── Manual Entry Mode ──────────────────────────────────────────────────────
    if order is None:
        render_landing()
        return

    # Run prediction
    with st.spinner("🔄 Running through 5-layer ML pipeline…"):
        try:
            report = engine.predict(order)
        except ValueError as e:
            st.error(f"❌ Validation Error: {e}")
            return
        except Exception as e:
            st.error(f"❌ Prediction Error: {e}")
            return

    render_report(report)


if __name__ == "__main__":
    main()
