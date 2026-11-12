import os
import json
import requests
from dotenv import load_dotenv

load_dotenv(override=True)
CLIENT_ID = os.getenv("PAYPAL_CLIENT_ID")
SECRET = os.getenv("PAYPAL_SECRET")
BASE = "https://api-m.sandbox.paypal.com"

# Three demo orders: one low risk, one borderline, one high risk.
DEMO_ORDERS = {
    "CG-2001": {"amount": 62,  "account_age_days": 500, "ship_bill_mismatch": 0, "hour_of_day": 11, "orders_last_hour": 0},
    "CG-2002": {"amount": 240, "account_age_days": 60,  "ship_bill_mismatch": 1, "hour_of_day": 16, "orders_last_hour": 0},
    "CG-2003": {"amount": 520, "account_age_days": 1,   "ship_bill_mismatch": 1, "hour_of_day": 3,  "orders_last_hour": 4},
}


def get_token():
    r = requests.post(
        f"{BASE}/v1/oauth2/token",
        auth=(CLIENT_ID, SECRET),
        data={"grant_type": "client_credentials"},
    )
    r.raise_for_status()
    return r.json()["access_token"]


def main():
    if not CLIENT_ID or not SECRET:
        raise SystemExit("Missing PAYPAL_CLIENT_ID or PAYPAL_SECRET. Copy .env.example to .env and fill it in.")

    headers = {"Authorization": f"Bearer {get_token()}", "Content-Type": "application/json"}

    # 1. Create the orders (intent AUTHORIZE: money is held, not taken)
    pending = {}
    for ref, feats in DEMO_ORDERS.items():
        body = {
            "intent": "AUTHORIZE",
            "purchase_units": [{
                "reference_id": ref,
                "amount": {"currency_code": "USD", "value": f"{feats['amount']:.2f}"},
            }],
            "payment_source": {"paypal": {"experience_context": {
                "return_url": "https://example.com/return",
                "cancel_url": "https://example.com/cancel",
                "user_action": "PAY_NOW",
            }}},
        }
        r = requests.post(f"{BASE}/v2/checkout/orders", headers=headers, json=body)
        r.raise_for_status()
        order = r.json()
        link = next(l["href"] for l in order["links"] if l["rel"] in ("payer-action", "approve"))
        pending[ref] = {"order_id": order["id"], "features": feats}
        print(f"\n{ref}  ${feats['amount']}\n{link}")

    # 2. The buyer must approve each order in the browser
    print("\nOpen each link above in a private window, log in with your sandbox PERSONAL (buyer)")
    print("account, and approve until you reach an example.com page.")

    # 3. Authorize the approved orders and save them
    held = {}
    while len(held) < len(pending):
        input("\nPress Enter after approving the orders... ")
        headers["Authorization"] = f"Bearer {get_token()}"
        for ref, info in pending.items():
            if ref in held:
                continue
            oid = info["order_id"]
            status = requests.get(f"{BASE}/v2/checkout/orders/{oid}", headers=headers).json().get("status")
            if status != "APPROVED":
                print(f"{ref}: not approved yet (status {status})")
                continue
            r = requests.post(f"{BASE}/v2/checkout/orders/{oid}/authorize", headers=headers)
            if r.status_code == 201:
                auth_id = r.json()["purchase_units"][0]["payments"]["authorizations"][0]["id"]
                held[ref] = {"order_id": oid, "auth_id": auth_id, "features": info["features"]}
                print(f"{ref}: authorized")
            else:
                print(f"{ref}: failed ({r.status_code}) {r.text[:150]}")

    with open("held_orders.json", "w") as f:
        json.dump(held, f, indent=2)
    print("\nSaved held_orders.json. Now run: streamlit run app.py")


if __name__ == "__main__":
    main()