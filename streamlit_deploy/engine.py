"""
=============================================================================
02_engine.py
Wire Manufacturing AI — Promise Date Engine (Core Pipeline)
=============================================================================
PURPOSE:
    The PromiseDateEngine class orchestrates all 5 ML models in sequence.
    Given a single sales order dict, it returns a full PromiseDateReport.

USAGE:
    from engine import PromiseDateEngine

    engine = PromiseDateEngine()          # loads all saved models
    report = engine.predict({
        "order_id":            "SO-2026-0451",
        "item_type":           "WHF",
        "wire_diameter_mm":    2.0,
        "qty_ordered_kg":      15000,
        "package_code":        "KH",
        "mill_assigned":       "Mill_B",
        "order_date":          "2026-04-14",
        "order_month":         4,
        "order_day_of_week":   0,
        "num_lines_per_SO":    3,
        "total_qty_per_SO":    45000,
        "unique_diameters_per_SO": 2,
        "diameter_group_load_kg": 200000,
        "concurrent_orders_this_week": 10,
        "concurrent_same_mill_orders": 3,
        "week_total_load_kg":  150000,
        "previous_order_diameter_on_mill": 1.6,
    })
    print(report)
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import warnings
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# ─── Paths ───────────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# ─── Production table lookup (mill specs) ────────────────────────────────────
MILL_SPECS = {
    "Mill_A": {"mill_run_rate_kghr": 150, "mill_setup_time_min": 240,
               "draw_run_rate_kghr": 130, "pack_run_rate_kghr": 110},
    "Mill_B": {"mill_run_rate_kghr": 95,  "mill_setup_time_min": 60,
               "draw_run_rate_kghr": 85,  "pack_run_rate_kghr": 76},
    "Mill_C": {"mill_run_rate_kghr": 120, "mill_setup_time_min": 120,
               "draw_run_rate_kghr": 100, "pack_run_rate_kghr": 90},
    "Mill_D": {"mill_run_rate_kghr": 80,  "mill_setup_time_min": 90,
               "draw_run_rate_kghr": 70,  "pack_run_rate_kghr": 62},
    "Mill_E": {"mill_run_rate_kghr": 110, "mill_setup_time_min": 180,
               "draw_run_rate_kghr": 95,  "pack_run_rate_kghr": 85},
}

# ─── Guardrail constants ──────────────────────────────────────────────────────
GUARDRAIL_MAX_DAYS       = 42      # 6-week hard cap
GUARDRAIL_MIN_RUN_RATE   = 50.0    # kg/hr floor
GUARDRAIL_MAX_RUN_RATE   = 180.0   # kg/hr ceiling
GUARDRAIL_MIN_LEAD_DAYS  = 1.0
HOURS_PER_DAY            = 8.0
OTIF_RISK_THRESHOLD      = 0.35    # risk probability → HIGH if > 0.65

VALID_ITEM_TYPES   = ["WHF","PHF","WSA","WCH","WTE","WDG","WGA","WSP","WDU"]
VALID_DIAMETERS    = [1.2, 1.6, 2.0, 2.4, 2.8, 3.2]
VALID_PACKAGE_CODES= ["KH","DJ","MC","L1","L2","KH1","SP1","SP2","PP"]
VALID_MILLS        = list(MILL_SPECS.keys())


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASS: PromiseDateReport
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PromiseDateReport:
    """Structured output from PromiseDateEngine.predict()"""

    order_id:             str   = ""
    order_date:           str   = ""

    # Layer 1 outputs
    nominal_run_rate_kghr:   float = 0.0
    adjusted_run_rate_kghr:  float = 0.0
    run_rate_adjustment_pct: float = 0.0

    # Layer 2 outputs
    base_lead_time_days:      float = 0.0
    queue_adjusted_days:      float = 0.0
    queue_contention_pct:     float = 0.0

    # Layer 3 outputs
    random_setup_min:         float = 0.0
    ml_optimized_setup_min:   float = 0.0
    setup_savings_pct:        float = 0.0
    setup_adjustment_days:    float = 0.0

    # Layer 4 outputs
    p10_days:             float = 0.0
    p50_days:             float = 0.0
    p90_days:             float = 0.0
    promise_date_p10:     str   = ""
    promise_date_p50:     str   = ""
    promise_date_p90:     str   = ""
    prob_exceed_42days:   float = 0.0
    guardrail_flag:       bool  = False

    # Layer 5 outputs
    otif_risk_probability: float = 0.0
    otif_risk_label:       str   = ""

    # Explainability
    shap_top3: list = field(default_factory=list)

    # Guardrail log
    guardrail_warnings: list = field(default_factory=list)

    def __str__(self):
        bar   = "=" * 68
        dbar  = "-" * 68
        GREEN = "\033[92m"; YELLOW = "\033[93m"
        RED   = "\033[91m"; RESET  = "\033[0m"; BOLD = "\033[1m"

        risk_color = (GREEN if self.otif_risk_label == "LOW"
                      else YELLOW if self.otif_risk_label == "MEDIUM"
                      else RED)
        guard_str  = f"{RED}⚠ EXCEEDS 42-DAY LIMIT{RESET}" if self.guardrail_flag else f"{GREEN}PASS{RESET}"

        lines = [
            f"\n{BOLD}{bar}{RESET}",
            f"{BOLD}  PROMISE DATE REPORT — Order: {self.order_id}{RESET}",
            f"{BOLD}{bar}{RESET}",
            f"  Order Date  : {self.order_date}",
            f"",
            f"  {BOLD}LAYER 1 — RUN RATE REFINER{RESET}",
            f"  {'Nominal Run Rate':<30} {self.nominal_run_rate_kghr:>8.1f} kg/hr",
            f"  {'Adjusted Run Rate':<30} {self.adjusted_run_rate_kghr:>8.1f} kg/hr"
            f"  ({self.run_rate_adjustment_pct:+.1f}%)",
            f"",
            f"  {BOLD}LAYER 2 — MULTI-ORDER ADJUSTER{RESET}",
            f"  {'Base Lead Time':<30} {self.base_lead_time_days:>8.1f} days",
            f"  {'Queue-Adjusted Lead Time':<30} {self.queue_adjusted_days:>8.1f} days"
            f"  ({self.queue_contention_pct:+.1f}% queue effect)",
            f"",
            f"  {BOLD}LAYER 3 — SETUP TIME OPTIMIZER{RESET}",
            f"  {'Random Sequence Setup':<30} {self.random_setup_min:>8.0f} min",
            f"  {'ML-Optimized Setup':<30} {self.ml_optimized_setup_min:>8.0f} min"
            f"  ({self.setup_savings_pct:.1f}% savings)",
            f"  {'Setup Time Adjustment':<30} {self.setup_adjustment_days:>+8.2f} days",
            f"",
            f"  {BOLD}LAYER 4 — PROBABILISTIC PROMISE DATE{RESET}",
            f"  {'P10  (Optimistic)':<30} {self.p10_days:>5.0f} days  →  {self.promise_date_p10}",
            f"  {'P50  (Commit Date) ★':<30} {self.p50_days:>5.0f} days  →  {BOLD}{self.promise_date_p50}{RESET}",
            f"  {'P90  (Conservative)':<30} {self.p90_days:>5.0f} days  →  {self.promise_date_p90}",
            f"  {'Prob. exceed 42 days':<30} {self.prob_exceed_42days*100:>7.1f}%",
            f"  {'6-Week Guardrail':<30} {guard_str}",
            f"",
            f"  {BOLD}LAYER 5 — OTIF RISK SCORE{RESET}",
            f"  {'Risk Probability':<30} {self.otif_risk_probability*100:>7.1f}%",
            f"  {'Risk Label':<30} {risk_color}{BOLD}{self.otif_risk_label}{RESET}",
            f"",
        ]
        if self.shap_top3:
            lines.append(f"  {BOLD}TOP 3 DRIVERS (SHAP){RESET}")
            for i, (feat, val, desc) in enumerate(self.shap_top3, 1):
                sign = "+" if val >= 0 else ""
                lines.append(f"  {i}. {feat:<28} {sign}{val:+.2f} days  — {desc}")
            lines.append("")

        if self.guardrail_warnings:
            lines.append(f"  {YELLOW}GUARDRAIL WARNINGS:{RESET}")
            for w in self.guardrail_warnings:
                lines.append(f"  ⚠  {w}")
            lines.append("")

        lines += [
            dbar,
            f"  {BOLD}★ FINAL PROMISE DATE : {self.promise_date_p50}{RESET}",
            f"  {'Confidence Band':<30} {self.promise_date_p10}  →  {self.promise_date_p90}",
            f"  {'OTIF Risk':<30} {risk_color}{self.otif_risk_label} ({self.otif_risk_probability*100:.0f}%){RESET}",
            f"{bar}\n",
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CLASS: PromiseDateEngine
# ─────────────────────────────────────────────────────────────────────────────
class PromiseDateEngine:
    """
    5-layer ML pipeline: Sales Order → Promise Date Report.

    Layers
    ------
    1. Run Rate Refiner        (Ridge Regression)
    2. Multi-Order Adjuster    (Random Forest Regressor)
    3. Setup Time Calculator   (Gradient Boosting Regressor)
    4. Promise Date Predictor  (Random Forest + Quantile Trees)
    5. OTIF Risk Scorer        (Random Forest Classifier)
    """

    def __init__(self, models_dir: str = MODELS_DIR):
        print("  Loading models...", end=" ", flush=True)
        self.m1       = joblib.load(os.path.join(models_dir, "layer1_run_rate.pkl"))
        self.m2       = joblib.load(os.path.join(models_dir, "layer2_queue_adjuster.pkl"))
        self.m3       = joblib.load(os.path.join(models_dir, "layer3_setup_time.pkl"))
        self.m4       = joblib.load(os.path.join(models_dir, "layer4_promise_date.pkl"))
        self.m5       = joblib.load(os.path.join(models_dir, "layer5_otif_risk.pkl"))
        self.encoders = joblib.load(os.path.join(models_dir, "encoders.pkl"))
        self.features = joblib.load(os.path.join(models_dir, "feature_columns.pkl"))
        print("done ✓")

    # ── helpers ──────────────────────────────────────────────────────────────

    def _encode(self, value: str, col: str) -> int:
        le = self.encoders.get(col)
        if le is None:
            return 0
        return int(le.transform([value])[0]) if value in le.classes_ else -1

    @staticmethod
    def _add_business_days(start_date: datetime, days: int) -> datetime:
        """Add `days` calendar days, skipping weekends."""
        current = start_date
        added   = 0
        while added < int(days):
            current += timedelta(days=1)
            if current.weekday() < 5:   # Mon–Fri only
                added += 1
        return current

    @staticmethod
    def _guardrail_clamp(value, lo, hi, name, warnings_list):
        if value < lo:
            warnings_list.append(f"{name} clamped from {value:.2f} to floor {lo:.2f}")
            return lo
        if value > hi:
            warnings_list.append(f"{name} clamped from {value:.2f} to ceiling {hi:.2f}")
            return hi
        return value

    @staticmethod
    def _risk_label(prob: float) -> str:
        if prob < 0.35:
            return "LOW"
        elif prob < 0.65:
            return "MEDIUM"
        return "HIGH"

    @staticmethod
    def _predict_quantiles(model, x_row: np.ndarray, quantiles=(0.10, 0.50, 0.90)):
        """Use individual trees of a RandomForest to produce quantile predictions."""
        tree_preds = np.array([t.predict(x_row.reshape(1, -1))[0]
                                for t in model.estimators_])
        return {q: float(np.percentile(tree_preds, q * 100)) for q in quantiles}

    # ── input validation ─────────────────────────────────────────────────────

    def _validate_order(self, order: dict) -> list:
        errors = []
        if order.get("item_type") not in VALID_ITEM_TYPES:
            errors.append(f"item_type '{order.get('item_type')}' not in {VALID_ITEM_TYPES}")
        if order.get("wire_diameter_mm") not in VALID_DIAMETERS:
            errors.append(f"wire_diameter_mm must be one of {VALID_DIAMETERS}")
        if not (0 < order.get("qty_ordered_kg", 0) <= 100_000):
            errors.append("qty_ordered_kg must be > 0 and <= 100,000")
        if order.get("mill_assigned") not in VALID_MILLS:
            errors.append(f"mill_assigned must be one of {VALID_MILLS}")
        return errors

    # ── Layer 1: Run Rate Refiner ─────────────────────────────────────────────

    def _layer1_run_rate(self, order: dict, warnings_list: list) -> float:
        mill   = order["mill_assigned"]
        specs  = MILL_SPECS.get(mill, MILL_SPECS["Mill_B"])
        nominal = float(specs["mill_run_rate_kghr"])

        x = np.array([[
            order["wire_diameter_mm"],
            self._encode(order["item_type"], "item_type"),
            specs["mill_run_rate_kghr"],
            specs["mill_setup_time_min"],
            order.get("diameter_group_load_kg", 200_000),
            order.get("order_month", datetime.now().month),
            order.get("order_day_of_week", 0),
        ]])
        adj = float(self.m1.predict(x)[0])
        adj = self._guardrail_clamp(
            adj, GUARDRAIL_MIN_RUN_RATE, GUARDRAIL_MAX_RUN_RATE,
            "adjusted_run_rate", warnings_list
        )
        return nominal, adj

    # ── Layer 2: Multi-Order Interaction ─────────────────────────────────────

    def _layer2_queue_adjuster(self, order: dict,
                                adjusted_rate: float) -> tuple:
        qty  = order["qty_ordered_kg"]
        base = qty / max(adjusted_rate * HOURS_PER_DAY, 1.0)

        x = np.array([[
            order["wire_diameter_mm"],
            self._encode(order["item_type"], "item_type"),
            qty,
            order.get("concurrent_orders_this_week", 10),
            order.get("concurrent_same_mill_orders", 3),
            order.get("week_total_load_kg", 120_000),
            order.get("diameter_group_load_kg", 200_000),
            base,
            order.get("order_month", datetime.now().month),
        ]])
        adjusted = float(self.m2.predict(x)[0])
        adjusted = max(adjusted, GUARDRAIL_MIN_LEAD_DAYS)
        return base, adjusted

    # ── Layer 3: Setup Time Calculator ───────────────────────────────────────

    def _layer3_setup_time(self, order: dict) -> tuple:
        curr_diam = order["wire_diameter_mm"]
        prev_diam = order.get("previous_order_diameter_on_mill", curr_diam)
        num_ord   = order.get("concurrent_same_mill_orders", 3) + 1
        u_diams   = order.get("unique_diameters_per_SO", 2)
        qty       = order["qty_ordered_kg"] * num_ord
        max_jump  = abs(curr_diam - prev_diam)
        avg_jump  = max_jump
        n_trans   = num_ord - 1
        is_sorted = 1 if curr_diam >= prev_diam else 0

        # Estimate random sequence setup (rule-based heuristic)
        setup_per_trans = 60 + max_jump * 80    # minutes
        random_setup    = setup_per_trans * max(n_trans, 1)

        x = np.array([[num_ord, u_diams, qty, max_jump, avg_jump, n_trans, is_sorted]])
        ml_setup = float(self.m3.predict(x)[0])
        ml_setup = max(ml_setup, 10.0)          # floor: 10 min minimum

        savings_pct  = (random_setup - ml_setup) / max(random_setup, 1) * 100
        setup_adjust = -(random_setup - ml_setup) / 60.0 / HOURS_PER_DAY
        return random_setup, ml_setup, savings_pct, setup_adjust

    # ── Layer 4: Probabilistic Promise Date ──────────────────────────────────

    def _layer4_promise_date(self, order: dict,
                              queue_days: float,
                              setup_adj:  float,
                              warnings_list: list) -> dict:
        mill  = order["mill_assigned"]
        specs = MILL_SPECS.get(mill, MILL_SPECS["Mill_B"])

        x_row = np.array([
            order["wire_diameter_mm"],
            self._encode(order["item_type"], "item_type"),
            order["qty_ordered_kg"],
            self._encode(order.get("package_code", "KH"), "package_code"),
            order.get("num_lines_per_SO", 1),
            order.get("total_qty_per_SO", order["qty_ordered_kg"]),
            order.get("unique_diameters_per_SO", 1),
            self._encode(mill, "mill_assigned"),
            specs["mill_run_rate_kghr"],
            specs["mill_setup_time_min"],
            specs["draw_run_rate_kghr"],
            specs["pack_run_rate_kghr"],
            order.get("diameter_group_load_kg", 200_000),
            order.get("order_month", datetime.now().month),
            order.get("order_day_of_week", 0),
        ])

        q = self._predict_quantiles(self.m4, x_row)
        # Derive a base estimate from qty and nominal rate as safety floor
        mill_rate = MILL_SPECS.get(order.get("mill_assigned","Mill_B"),
                                    MILL_SPECS["Mill_B"])["mill_run_rate_kghr"]
        base_floor = max(order["qty_ordered_kg"] / (mill_rate * HOURS_PER_DAY), 1.0)

        p10 = max(q[0.10] + setup_adj, 1.0)
        p50 = max(q[0.50] + setup_adj, base_floor)   # never lower than base
        p90 = max(q[0.90] + setup_adj, p50)

        # Monotonicity guardrail
        p10 = min(p10, p50)
        p90 = max(p90, p50)

        # Probability of exceeding 42 days
        tree_preds = np.array([t.predict(x_row.reshape(1, -1))[0] for t in self.m4.estimators_])
        prob_exceed = float(np.mean(tree_preds + setup_adj > GUARDRAIL_MAX_DAYS))

        # Convert to calendar dates
        try:
            order_dt = datetime.strptime(order["order_date"], "%Y-%m-%d")
        except Exception:
            order_dt = datetime.today()

        date_p10 = self._add_business_days(order_dt, int(p10)).strftime("%B %d, %Y")
        date_p50 = self._add_business_days(order_dt, int(p50)).strftime("%B %d, %Y")
        date_p90 = self._add_business_days(order_dt, int(p90)).strftime("%B %d, %Y")

        guardrail_flag = p50 > GUARDRAIL_MAX_DAYS
        if guardrail_flag:
            warnings_list.append(
                f"P50 ({p50:.0f} days) EXCEEDS 6-WEEK GUARDRAIL ({GUARDRAIL_MAX_DAYS} days). "
                f"Escalate to planning manager."
            )
        if (p90 - p10) > 30:
            warnings_list.append(
                f"Wide confidence band ({p10:.0f}d–{p90:.0f}d = {p90-p10:.0f} days span). "
                f"Consider manual review."
            )

        return dict(p10=p10, p50=p50, p90=p90,
                    date_p10=date_p10, date_p50=date_p50, date_p90=date_p90,
                    prob_exceed=prob_exceed, flag=guardrail_flag)

    # ── Layer 5: OTIF Risk Scorer ─────────────────────────────────────────────

    def _layer5_otif_risk(self, order: dict, p50_days: float) -> tuple:
        mill  = order["mill_assigned"]
        specs = MILL_SPECS.get(mill, MILL_SPECS["Mill_B"])

        x = np.array([[
            order["wire_diameter_mm"],
            self._encode(order["item_type"], "item_type"),
            order["qty_ordered_kg"],
            self._encode(order.get("package_code", "KH"), "package_code"),
            order.get("num_lines_per_SO", 1),
            order.get("total_qty_per_SO", order["qty_ordered_kg"]),
            order.get("unique_diameters_per_SO", 1),
            self._encode(mill, "mill_assigned"),
            specs["mill_run_rate_kghr"],
            specs["mill_setup_time_min"],
            order.get("diameter_group_load_kg", 200_000),
            order.get("order_month", datetime.now().month),
            order.get("order_day_of_week", 0),
            order.get("concurrent_orders_this_week", 10),
        ]])
        prob  = float(self.m5.predict_proba(x)[0][1])
        prob  = float(np.clip(prob, 0.0, 1.0))
        label = self._risk_label(prob)
        return prob, label

    # ── SHAP-style top-3 feature contributions ────────────────────────────────

    def _explain_top3(self, order: dict, p50_days: float) -> list:
        """
        Lightweight feature attribution using RF feature importances
        scaled by each feature's deviation from training mean.
        Returns list of (feature_name, impact_days, description) tuples.
        """
        mill     = order["mill_assigned"]
        specs    = MILL_SPECS.get(mill, MILL_SPECS["Mill_B"])
        imp      = self.m4.feature_importances_
        feats    = self.features["L4"]

        # Approximate per-feature contribution as importance × normalised value
        # (this is a lightweight proxy; use shap.TreeExplainer for exact SHAP)
        ref_values = dict(
            wire_diameter=2.0, item_type=0, qty_ordered=10000,
            package_code=0, num_lines_per_SO=3, total_qty_per_SO=30000,
            unique_diameters_per_SO=2, mill_assigned=1,
            mill_run_rate_kghr=100, mill_setup_time_min=90,
            draw_run_rate_kghr=90,  pack_run_rate_kghr=80,
            diameter_group_load_kg=150000, order_month=6, order_day_of_week=2
        )
        feat_vals = dict(
            wire_diameter=order["wire_diameter_mm"],
            item_type=self._encode(order["item_type"], "item_type"),
            qty_ordered=order["qty_ordered_kg"],
            package_code=self._encode(order.get("package_code","KH"),"package_code"),
            num_lines_per_SO=order.get("num_lines_per_SO",1),
            total_qty_per_SO=order.get("total_qty_per_SO", order["qty_ordered_kg"]),
            unique_diameters_per_SO=order.get("unique_diameters_per_SO",1),
            mill_assigned=self._encode(mill, "mill_assigned"),
            mill_run_rate_kghr=specs["mill_run_rate_kghr"],
            mill_setup_time_min=specs["mill_setup_time_min"],
            draw_run_rate_kghr=specs["draw_run_rate_kghr"],
            pack_run_rate_kghr=specs["pack_run_rate_kghr"],
            diameter_group_load_kg=order.get("diameter_group_load_kg",200000),
            order_month=order.get("order_month", datetime.now().month),
            order_day_of_week=order.get("order_day_of_week",0),
        )
        DESCRIPTIONS = {
            "qty_ordered":           "Order size directly drives production hours",
            "diameter_group_load_kg":"Same-diameter queue congestion",
            "mill_run_rate_kghr":    "Mill throughput rate",
            "item_type":             "Wire type affects run speed",
            "unique_diameters_per_SO":"More diameters = more setup transitions",
            "num_lines_per_SO":      "Complex multi-line orders take longer",
            "total_qty_per_SO":      "Total SO volume creates scheduling pressure",
            "wire_diameter":         "Wire gauge affects draw speed",
            "order_month":           "Seasonal demand pattern",
            "mill_setup_time_min":   "Mill setup overhead",
            "mill_assigned":         "Mill capacity profile",
            "package_code":          "Packaging format speed",
            "draw_run_rate_kghr":    "Drawing station throughput",
            "pack_run_rate_kghr":    "Packaging station throughput",
            "order_day_of_week":     "Day-of-week scheduling effects",
        }
        # Feature scale denominators for normalisation (approximate std dev of each feature)
        feat_scale = dict(
            wire_diameter=0.8, item_type=3.0, qty_ordered=12000,
            package_code=3.0, num_lines_per_SO=4.0, total_qty_per_SO=80000,
            unique_diameters_per_SO=1.5, mill_assigned=1.5,
            mill_run_rate_kghr=25.0, mill_setup_time_min=80.0,
            draw_run_rate_kghr=20.0, pack_run_rate_kghr=18.0,
            diameter_group_load_kg=200000, order_month=3.5, order_day_of_week=2.0
        )
        scores = []
        for i, f in enumerate(feats):
            ref   = ref_values.get(f, 0)
            val   = feat_vals.get(f, 0)
            scale = feat_scale.get(f, max(abs(ref), 1))
            delta_norm = (val - ref) / scale          # normalised deviation
            impact = imp[i] * delta_norm * p50_days   # in days
            desc   = DESCRIPTIONS.get(f, f)
            scores.append((f, impact, desc))

        scores.sort(key=lambda x: abs(x[1]), reverse=True)
        return scores[:3]

    # ── Main predict method ───────────────────────────────────────────────────

    def predict(self, order: dict) -> PromiseDateReport:
        """
        Run the full 5-layer pipeline on a single sales order dict.

        Parameters
        ----------
        order : dict
            Must contain at minimum:
            order_id, item_type, wire_diameter_mm, qty_ordered_kg,
            mill_assigned, order_date

        Returns
        -------
        PromiseDateReport
        """
        report   = PromiseDateReport()
        warnings = []

        # ── Validate input ────────────────────────────────────────────────────
        errors = self._validate_order(order)
        if errors:
            raise ValueError("Order validation failed:\n" + "\n".join(f"  • {e}" for e in errors))

        report.order_id   = order.get("order_id", "UNKNOWN")
        report.order_date = order.get("order_date", datetime.today().strftime("%Y-%m-%d"))

        # ── Layer 1 ───────────────────────────────────────────────────────────
        nominal, adjusted = self._layer1_run_rate(order, warnings)
        report.nominal_run_rate_kghr   = round(nominal, 1)
        report.adjusted_run_rate_kghr  = round(adjusted, 1)
        report.run_rate_adjustment_pct = round((adjusted - nominal) / nominal * 100, 1)

        # ── Layer 2 ───────────────────────────────────────────────────────────
        base_days, queue_days = self._layer2_queue_adjuster(order, adjusted)
        report.base_lead_time_days   = round(base_days, 1)
        report.queue_adjusted_days   = round(queue_days, 1)
        report.queue_contention_pct  = round((queue_days - base_days) / max(base_days,1) * 100, 1)

        # ── Layer 3 ───────────────────────────────────────────────────────────
        r_setup, ml_setup, sav_pct, setup_adj = self._layer3_setup_time(order)
        report.random_setup_min       = round(r_setup,  1)
        report.ml_optimized_setup_min = round(ml_setup, 1)
        report.setup_savings_pct      = round(sav_pct,  1)
        report.setup_adjustment_days  = round(setup_adj, 2)

        # ── Layer 4 ───────────────────────────────────────────────────────────
        l4 = self._layer4_promise_date(order, queue_days, setup_adj, warnings)
        report.p10_days           = round(l4["p10"], 1)
        report.p50_days           = round(l4["p50"], 1)
        report.p90_days           = round(l4["p90"], 1)
        report.promise_date_p10   = l4["date_p10"]
        report.promise_date_p50   = l4["date_p50"]
        report.promise_date_p90   = l4["date_p90"]
        report.prob_exceed_42days = round(l4["prob_exceed"], 3)
        report.guardrail_flag     = l4["flag"]

        # ── Layer 5 ───────────────────────────────────────────────────────────
        risk_prob, risk_label = self._layer5_otif_risk(order, l4["p50"])
        report.otif_risk_probability = round(risk_prob, 3)
        report.otif_risk_label       = risk_label

        # ── Explainability ────────────────────────────────────────────────────
        report.shap_top3           = self._explain_top3(order, l4["p50"])
        report.guardrail_warnings  = warnings

        return report
