import hashlib
import json
import time
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Federated Healthcare", layout="wide")


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


HOSPITALS = ["City General", "Riverside Medical", "Northside Clinic", "St. Mary's", "Lakeview Health"]
USERS = {
    "admin": {"pw": sha("admin123"), "role": "Admin"},
    "server": {"pw": sha("server123"), "role": "Central Server"},
}
for i, n in enumerate(HOSPITALS, 1):
    USERS[f"hospital{i}"] = {"pw": sha(f"hosp{i}"), "role": "Hospital", "hospital": n}

DELTA = 1e-5


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class Federation:
    def __init__(self):
        d = load_breast_cancer()
        X = StandardScaler().fit_transform(d.data)
        X = np.hstack([X, np.ones((len(X), 1))])
        Xtr, self.Xte, ytr, self.yte = train_test_split(X, d.target, test_size=0.25, random_state=0, stratify=d.target)
        rng = np.random.default_rng(1)
        order = np.argsort(ytr + rng.normal(0, 0.9, len(ytr)))
        parts = np.array_split(order, len(HOSPITALS))
        self.data = {n: (Xtr[p], ytr[p]) for n, p in zip(HOSPITALS, parts)}
        self.approved = {n: True for n in HOSPITALS}
        self.policy = dict(dp=True, secure=True, clip=1.0, sigma=1.0, lr=0.2, epochs=3, budget=40.0)
        self.rng = np.random.default_rng(7)
        self.reset()

    def reset(self):
        self.w = np.zeros(self.Xte.shape[1])
        self.round = 0
        self.rho = 0.0
        self.pending = {}
        self.history = []
        self.last_view = {}
        self.contrib = {n: 0 for n in HOSPITALS}
        self.log = []
        self.add_log("system", "Federation initialised")

    def add_log(self, actor, action):
        self.log.append({"time": time.strftime("%H:%M:%S"), "actor": actor, "action": action})

    def accuracy(self, w, X, y):
        return float(((sigmoid(X @ w) > 0.5) == y).mean())

    def epsilon(self, rho=None):
        rho = self.rho if rho is None else rho
        if rho == 0:
            return 0.0
        return rho + 2 * np.sqrt(rho * np.log(1 / DELTA))

    def round_cost(self):
        return 1 / (2 * self.policy["sigma"] ** 2)

    def budget_exhausted(self):
        return self.policy["dp"] and self.epsilon(self.rho + self.round_cost()) > self.policy["budget"]

    def train_local(self, name):
        p = self.policy
        X, y = self.data[name]
        w = self.w.copy()
        for _ in range(p["epochs"]):
            w = w - p["lr"] * X.T @ (sigmoid(X @ w) - y) / len(y)
        update = w - self.w
        if p["dp"]:
            update = update * min(1.0, p["clip"] / (np.linalg.norm(update) + 1e-12))
        self.pending[name] = update
        self.add_log(name, f"Submitted update for round {self.round + 1}")

    def mask(self, updates):
        masked = [u.copy() for u in updates]
        for i in range(len(masked)):
            for j in range(i + 1, len(masked)):
                m = self.rng.normal(0, 5, len(masked[0]))
                masked[i] += m
                masked[j] -= m
        return masked

    def aggregate(self):
        p = self.policy
        names = [n for n in self.pending if self.approved[n]]
        updates = [self.pending[n] for n in names]
        sent = self.mask(updates) if p["secure"] else updates
        self.last_view = {n: round(float(np.linalg.norm(s)), 2) for n, s in zip(names, sent)}
        total = np.sum(sent, axis=0)
        if p["dp"]:
            total = total + self.rng.normal(0, p["sigma"] * p["clip"], total.shape)
            self.rho += self.round_cost()
        self.w = self.w + total / len(names)
        self.round += 1
        for n in names:
            self.contrib[n] += 1
        acc = self.accuracy(self.w, self.Xte, self.yte)
        self.history.append({"round": self.round, "accuracy": acc, "hospitals": len(names), "epsilon": self.epsilon() if p["dp"] else 0.0})
        self.pending = {}
        self.add_log("server", f"Aggregated round {self.round} from {len(names)} hospitals, accuracy {acc:.1%}")


@st.cache_resource
def get_fed():
    return Federation()


fed = get_fed()


def login_page():
    st.title("Federated Healthcare Platform")
    st.caption("Sign in as an admin, the central server, or a hospital.")
    with st.form("login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Sign in", type="primary"):
            rec = USERS.get(username)
            if rec and rec["pw"] == sha(password):
                st.session_state.user = username
                fed.add_log(username, "Signed in")
                st.rerun()
            else:
                st.error("Invalid username or password")
    with st.expander("Demo accounts"):
        rows = [("admin", "admin123", "Admin"), ("server", "server123", "Central Server")]
        rows += [(f"hospital{i}", f"hosp{i}", n) for i, n in enumerate(HOSPITALS, 1)]
        st.table(pd.DataFrame(rows, columns=["Username", "Password", "Profile"]))


def admin_page():
    st.title("Admin Console")
    tab_policy, tab_hosp, tab_log = st.tabs(["Privacy policy", "Hospitals", "Audit log"])

    with tab_policy:
        p = fed.policy
        c1, c2 = st.columns(2)
        dp = c1.toggle("Differential privacy", p["dp"])
        secure = c1.toggle("Secure aggregation", p["secure"])
        clip = c1.slider("Clip norm", 0.1, 5.0, float(p["clip"]))
        sigma = c2.slider("Noise multiplier", 0.3, 5.0, float(p["sigma"]))
        budget = c2.slider("Privacy budget (epsilon)", 5.0, 200.0, float(p["budget"]))
        epochs = c1.slider("Local epochs", 1, 10, int(p["epochs"]))
        lr = c2.slider("Learning rate", 0.01, 1.0, float(p["lr"]))
        if st.button("Save policy", type="primary"):
            fed.policy = dict(dp=dp, secure=secure, clip=clip, sigma=sigma, lr=lr, epochs=epochs, budget=budget)
            fed.add_log("admin", f"Policy updated: dp={dp}, secure={secure}, clip={clip}, sigma={sigma}, budget={budget}")
            st.success("Policy saved")
        if st.button("Reset federation"):
            fed.reset()
            fed.add_log("admin", "Federation reset")
            st.rerun()

    with tab_hosp:
        for n in HOSPITALS:
            c1, c2, c3 = st.columns([3, 1, 2])
            c1.write(f"**{n}**")
            c2.write(f"{len(fed.data[n][1])} patients")
            state = c3.toggle("Approved", fed.approved[n], key=f"ap_{n}")
            if state != fed.approved[n]:
                fed.approved[n] = state
                fed.pending.pop(n, None)
                fed.add_log("admin", f"{'Approved' if state else 'Revoked'} {n}")
                st.rerun()

    with tab_log:
        st.dataframe(pd.DataFrame(fed.log[::-1]), hide_index=True, width="stretch")


def server_page():
    st.title("Central Server")
    p = fed.policy
    acc = fed.history[-1]["accuracy"] if fed.history else 0.0
    active = [n for n in HOSPITALS if fed.approved[n]]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Round", fed.round)
    c2.metric("Global accuracy", f"{acc:.1%}")
    c3.metric("Updates received", f"{len(fed.pending)} / {len(active)}")
    c4.metric("Privacy spent", f"{fed.epsilon():.1f} / {p['budget']:.0f}" if p["dp"] else "DP off")

    status = pd.DataFrame(
        {
            "Hospital": HOSPITALS,
            "Approved": [fed.approved[n] for n in HOSPITALS],
            "Update received": [n in fed.pending for n in HOSPITALS],
            "Rounds contributed": [fed.contrib[n] for n in HOSPITALS],
        }
    )
    st.dataframe(status, hide_index=True, width="stretch")

    exhausted = fed.budget_exhausted()
    if exhausted:
        st.warning("Privacy budget exhausted. Training is blocked until the admin raises the budget or resets.")
    ready = len([n for n in fed.pending if fed.approved[n]]) >= 2
    if st.button("Aggregate round", type="primary", disabled=not ready or exhausted):
        fed.aggregate()
        st.rerun()
    if not ready:
        st.caption("At least 2 approved hospitals must submit updates.")

    if fed.history:
        hist = pd.DataFrame(fed.history).set_index("round")
        left, right = st.columns(2)
        left.subheader("Global accuracy")
        left.line_chart(hist["accuracy"])
        right.subheader("Privacy budget spent")
        right.line_chart(hist["epsilon"])

    if fed.last_view:
        st.subheader("What the server saw last round")
        label = "Masked update norm" if p["secure"] else "Raw update norm"
        st.dataframe(pd.DataFrame({"Hospital": list(fed.last_view), label: list(fed.last_view.values())}), hide_index=True)

    st.download_button("Download global model", json.dumps(fed.w.tolist()), "global_model.json")


def hospital_page(name):
    X, y = fed.data[name]
    st.title(name)
    approved = fed.approved[name]
    if approved:
        st.success("Approved to participate. Patient records never leave this hospital.")
    else:
        st.error("Participation revoked by admin.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patients", len(y))
    c2.metric("Malignant cases", f"{(1 - y.mean()):.0%}")
    c3.metric("Global model on my data", f"{fed.accuracy(fed.w, X, y):.1%}")
    c4.metric("Rounds contributed", fed.contrib[name])

    st.subheader(f"Round {fed.round + 1}")
    submitted = name in fed.pending
    exhausted = fed.budget_exhausted()
    if submitted:
        st.info("Your update is submitted. Waiting for the central server to aggregate.")
    if exhausted:
        st.warning("Privacy budget exhausted.")
    if st.button("Train locally and submit update", type="primary", disabled=not approved or submitted or exhausted):
        fed.train_local(name)
        st.rerun()
    p = fed.policy
    st.caption(f"Policy: DP {'on' if p['dp'] else 'off'}, secure aggregation {'on' if p['secure'] else 'off'}, clip {p['clip']}, noise {p['sigma']}")

    st.subheader("Risk check with the global model")
    i = st.number_input("Patient index", 0, len(y) - 1, 0)
    benign = float(sigmoid(X[i] @ fed.w))
    st.progress(benign, text=f"Predicted probability benign: {benign:.0%}")
    st.write(f"Recorded diagnosis: **{'Benign' if y[i] == 1 else 'Malignant'}**")


if "user" not in st.session_state:
    login_page()
    st.stop()

user = USERS[st.session_state.user]
with st.sidebar:
    st.write(f"Signed in as **{st.session_state.user}**")
    st.write(user["role"])
    if st.button("Refresh"):
        st.rerun()
    if st.button("Sign out"):
        fed.add_log(st.session_state.user, "Signed out")
        del st.session_state.user
        st.rerun()

if user["role"] == "Admin":
    admin_page()
elif user["role"] == "Central Server":
    server_page()
else:
    hospital_page(user["hospital"])
