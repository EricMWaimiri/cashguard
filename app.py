import os
import json
import requests
import joblib
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv(override=True)
CLIENT_ID = os.getenv("PAYPAL_CLIENT_ID")
SECRET = os.getenv("PAYPAL_SECRET")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
BASE = "https://api-m.sandbox.paypal.com"
GEMINI_MODELS = ["gemini-3.1-flash-lite-preview", "gemini-flash-lite-latest", "gemini-3.5-flash"]
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

LABELS = {
    "amount": "a high order amount",
    "account_age_days": "a very new customer account",
    "ship_bill_mismatch": "billing and shipping addresses that do not match",
    "hour_of_day": "an unusual time of day",
    "orders_last_hour": "several orders placed within the last hour",
}

# ---------- Load the model and data ----------
if not os.path.exists("fraud_model.joblib") or not os.path.exists("orders.csv"):
    st.error("Model files not found. Run `python gen_data.py` and then `python train.py` first.")
    st.stop()
fraud_model, fraud_cols = joblib.load("fraud_model.joblib")
typical = pd.read_csv("orders.csv")[fraud_cols].mean()


def load_held():
    with open("held_orders.json") as f:
        return json.load(f)


def save_held(held):
    with open("held_orders.json", "w") as f:
        json.dump(held, f, indent=2)


# ---------- Fraud scoring ----------
def score_order(order):
    row = pd.DataFrame([order])[fraud_cols]
    risk = fraud_model.predict_proba(row)[0][1]
    push = fraud_model.coef_[0] * (row.iloc[0] - typical)
    factors = push.sort_values(ascending=False)
    top = [(name, round(float(v), 2)) for name, v in factors.items() if v > 0.1][:3]
    if risk < 0.30:
        action = "CAPTURE"
    elif risk < 0.70:
        action = "REVIEW"
    else:
        action = "VOID"
    return risk, action, top


# ---------- Plain-English explanation ----------
def backup_explanation(risk, action, reasons):
    if not reasons:
        return f"Risk is {risk:.0%}. No unusual signals compared with typical orders."
    return f"Risk is {risk:.0%}. Main factors: " + ", ".join(reasons) + "."


def explain(risk, action, top):
    """Returns (text, came_from_ai). Tries each model in turn, then uses backup text."""
    reasons = [LABELS[name] for name, value in top]

    if st.session_state.get("ai_down"):
        return backup_explanation(risk, action, reasons), False

    prompt = (
        "You help a small online merchant review an order. "
        f"The fraud model gave a risk of {risk:.0%} and recommends: {action}. "
        f"The only factors that raised the risk were: {', '.join(reasons) if reasons else 'none'}. "
        "Write 2 short sentences for the merchant. Use ONLY the factors listed. "
        "Do not invent any other reasons. Do not use technical terms."
    )

    for model in GEMINI_MODELS:
        try:
            r = requests.post(
                f"{GEMINI_BASE}/{model}:generateContent",
                headers={"x-goog-api-key": GEMINI_KEY, "Content-Type": "application/json"},
                json={"contents": [{"parts": [{"text": prompt}]}]},
                timeout=12,
            )
            r.raise_for_status()
            text = r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
            st.session_state["ai_model_used"] = model
            return text, True
        except Exception as e:
            st.session_state["last_ai_error"] = f"{model}: {str(e)[:120]}"

    st.session_state["ai_down"] = True
    return backup_explanation(risk, action, reasons), False


# ---------- PayPal ----------
def get_token():
    r = requests.post(
        f"{BASE}/v1/oauth2/token",
        auth=(CLIENT_ID, SECRET),
        data={"grant_type": "client_credentials"},
    )
    r.raise_for_status()
    return r.json()["access_token"]


def paypal_action(auth_id, action):
    headers = {"Authorization": f"Bearer {get_token()}", "Content-Type": "application/json"}
    if action == "capture":
        r = requests.post(f"{BASE}/v2/payments/authorizations/{auth_id}/capture",
                          headers=headers, json={})
    else:
        r = requests.post(f"{BASE}/v2/payments/authorizations/{auth_id}/void",
                          headers=headers)
    body = r.json() if r.text else {}
    return r.status_code, body


# ---------- Page ----------
st.set_page_config(page_title="CashGuard", page_icon="🛡️", layout="wide")
st.title("CashGuard")
st.caption("Know which payments to trust. Models are trained on synthetic data. "
           "Payments run through the real PayPal sandbox.")

if st.sidebar.button("Retry AI explanations"):
    st.session_state.pop("ai_down", None)
    st.session_state.pop("last_ai_error", None)
    st.rerun()

tab_orders = st.container()
st.subheader("Incoming orders")

with tab_orders:
    if not os.path.exists("held_orders.json"):
        st.warning("No held orders yet. Run `python setup_orders.py` first, then refresh this page.")
        st.stop()
    held = load_held()
    for ref, info in held.items():
        feats = info["features"]
        risk, action, top = score_order(feats)

        with st.container(border=True):
            c1, c2, c3 = st.columns([1, 1, 3])
            c1.subheader(ref)
            c1.write(f"**${feats['amount']:.2f}**")
            c2.metric("Fraud risk", f"{risk:.0%}")

            if action == "CAPTURE":
                c2.success("Suggested: CAPTURE")
            elif action == "REVIEW":
                c2.warning("Suggested: REVIEW")
            else:
                c2.error("Suggested: VOID")

            if info.get("explanation"):
                text = info["explanation"]
            else:
                text, from_ai = explain(risk, action, top)
                if from_ai:
                    info["explanation"] = text
                    save_held(held)
            c3.write("**Why:** " + text)
            if top:
                c3.caption("Top risk factors: " + ", ".join(LABELS[n] for n, v in top))

            if info.get("status"):
                st.info(f"Already {info['status']}. PayPal response: {info.get('paypal_response')}")
            else:
                if action == "REVIEW":
                    st.caption("Held for review: the payment is authorized but not captured until you approve or void it.")
                b1, b2, _ = st.columns([1, 1, 3])
                if b1.button("Approve capture", key=f"cap_{ref}"):
                    code, body = paypal_action(info["auth_id"], "capture")
                    if code in (200, 201):
                        info["status"] = "captured"
                        info["paypal_response"] = body.get("status", "OK")
                        save_held(held)
                        st.rerun()
                    else:
                        st.error(f"PayPal error {code}: {body}")
                if b2.button("Void order", key=f"void_{ref}"):
                    code, body = paypal_action(info["auth_id"], "void")
                    if code in (200, 204):
                        info["status"] = "voided"
                        info["paypal_response"] = f"HTTP {code} (authorization voided)"
                        save_held(held)
                        st.rerun()
                    else:
                        st.error(f"PayPal error {code}: {body}")

    if st.session_state.get("ai_down"):
        st.caption("AI is busy right now, so backup text is shown. "
                   "Use 'Retry AI explanations' in the sidebar. Last note: "
                   + st.session_state.get("last_ai_error", ""))