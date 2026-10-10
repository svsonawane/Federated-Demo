import ast
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Federated Healthcare", layout="wide")

FED_VERSION = 2
DELTA = 1e-5
DATA_PATH = Path(__file__).parent / "disease_data.csv"
PATIENTS_PER_DISEASE = 60
SYMPTOM_KEEP_PROB = 0.75
KEY_COLS = ["patient_id", "diagnosis", "n_symptoms", "symptoms"]
HOSPITAL_NAMES = ["City General", "Riverside Medical", "Northside Clinic", "St. Mary's", "Lakeview Health"]
DEFAULT_POLICY = dict(dp=True, secure=True, clip=1.0, sigma=0.5, lr=8.0, epochs=3, budget=120.0)
PRESETS = {
    "No privacy": dict(dp=False, secure=False, clip=1.0, sigma=0.5, budget=120.0),
    "Balanced": dict(dp=True, secure=True, clip=1.0, sigma=0.5, budget=120.0),
    "Strong privacy": dict(dp=True, secure=True, clip=1.0, sigma=1.0, budget=40.0),
}


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def max_rounds(sigma, budget):
    L = np.log(1 / DELTA)
    rho = (np.sqrt(L + budget) - np.sqrt(L)) ** 2
    return int(rho * 2 * sigma ** 2)


def clean(text):
    return text.replace("\xa0", " ").strip()


def load_profiles(path):
    raw = pd.read_csv(path, index_col=0)
    raw["Disease"] = raw["Disease"].map(clean)
    raw["Symptom"] = raw["Symptom"].map(
        lambda s: list(dict.fromkeys(clean(x) for x in ast.literal_eval(s)))
    )
    return raw.drop_duplicates("Disease").reset_index(drop=True)


class Federation:
    def __init__(self):
        profiles = load_profiles(DATA_PATH)
        self.classes = profiles["Disease"].tolist()
        self.feature_names = sorted({s for syms in profiles["Symptom"] for s in syms})
        K, F = len(self.classes), len(self.feature_names)
        sym_idx = {s: i for i, s in enumerate(self.feature_names)}
        freq = np.zeros(F)
        for syms in profiles["Symptom"]:
            for s in syms:
                freq[sym_idx[s]] += 1
        freq /= freq.sum()

        rng = np.random.default_rng(0)
        rows, labels = [], []
        for c, syms in enumerate(profiles["Symptom"]):
            for _ in range(PATIENTS_PER_DISEASE):
                keep = [s for s in syms if rng.random() < SYMPTOM_KEEP_PROB]
                if len(keep) < min(2, len(syms)):
                    keep = list(rng.choice(syms, size=min(2, len(syms)), replace=False))
                noise = [self.feature_names[j] for j in rng.choice(F, size=rng.integers(0, 3), replace=False, p=freq)]
                rows.append(sorted(set(keep) | set(noise)))
                labels.append(c)
        perm = rng.permutation(len(rows))
        rows, y_all = [rows[i] for i in perm], np.array(labels)[perm]

        X_all = np.zeros((len(rows), F + 1))
        X_all[:, -1] = 1.0
        for i, syms in enumerate(rows):
            X_all[i, [sym_idx[s] for s in syms]] = 1.0
        raw = pd.DataFrame(
            {
                "patient_id": [f"P-{i:04d}" for i in range(len(rows))],
                "diagnosis": [self.classes[c] for c in y_all],
                "n_symptoms": [len(s) for s in rows],
                "symptoms": [", ".join(s) for s in rows],
            }
        )

        train_idx, test_idx = train_test_split(np.arange(len(raw)), test_size=0.25, random_state=0, stratify=y_all)
        self.Xte, self.yte = X_all[test_idx], y_all[test_idx]
        rank = rng.permutation(K)
        key = rank[y_all[train_idx]] + rng.normal(0, K * 0.5, len(train_idx))
        order = train_idx[np.argsort(key)]
        shards = [
            dict(X=X_all[p], y=y_all[p], df=raw.iloc[p].reset_index(drop=True))
            for p in np.array_split(order, 8)
        ]
        self.users, self.data, self.approved, self.log = {}, {}, {}, []
        self.policy = dict(DEFAULT_POLICY)
        self.rng = np.random.default_rng(7)
        self.add_user("admin", "admin123", "Admin", demo=True)
        self.add_user("server", "server123", "Central Server", demo=True)
        for i, name in enumerate(HOSPITAL_NAMES, 1):
            self.data[name] = shards[i - 1]
            self.approved[name] = True
            self.add_user(f"hospital{i}", f"hosp{i}", "Hospital", hospital=name, demo=True)
        self.free = shards[len(HOSPITAL_NAMES):]
        self.add_log("system", "Federation initialised")
        self.reset()

    def add_user(self, username, password, role, hospital=None, demo=False):
        self.users[username] = dict(pw=sha(f"{username}:{password}"), role=role, hospital=hospital, demo=demo)

    def add_log(self, actor, action):
        self.log.append({"time": time.strftime("%H:%M:%S"), "actor": actor, "action": action})

    def reset(self):
        self.w = np.zeros((self.Xte.shape[1], len(self.classes)))
        self.round = 0
        self.rho = 0.0
        self.pending = {}
        self.history = []
        self.last_view = {}
        self.last_train = {}
        self.contrib = {n: 0 for n in self.data}

    def sign_up(self, username, password, hospital):
        username, hospital = username.strip().lower(), hospital.strip()
        if not username or not hospital:
            return "Username and hospital name are required"
        if len(password) < 6:
            return "Password must be at least 6 characters"
        if username in self.users:
            return "Username already taken"
        if hospital in self.data:
            return "Hospital name already registered"
        if not self.free:
            return "No patient data left for new hospitals in this demo"
        self.data[hospital] = self.free.pop(0)
        self.approved[hospital] = False
        self.contrib[hospital] = 0
        self.add_user(username, password, "Hospital", hospital=hospital)
        self.add_log(username, f"Signed up {hospital}, awaiting approval")
        return None

    def predict_proba(self, w, X):
        return softmax(X @ w)

    def accuracy(self, w, X, y):
        return float((np.argmax(X @ w, axis=1) == y).mean())

    def top_k_accuracy(self, k=3):
        top = np.argsort(-(self.Xte @ self.w), axis=1)[:, :k]
        return float((top == self.yte[:, None]).any(axis=1).mean())

    def per_disease(self):
        pred = np.argmax(self.Xte @ self.w, axis=1)
        rows = []
        for c, name in enumerate(self.classes):
            m = self.yte == c
            rows.append({"Disease": name, "Test patients": int(m.sum()), "Correct": float((pred[m] == c).mean())})
        return pd.DataFrame(rows).sort_values("Correct")

    def epsilon(self, rho=None):
        rho = self.rho if rho is None else rho
        return 0.0 if rho == 0 else rho + 2 * np.sqrt(rho * np.log(1 / DELTA))

    def round_cost(self):
        return 1 / (2 * self.policy["sigma"] ** 2)

    def budget_exhausted(self):
        return self.policy["dp"] and self.epsilon(self.rho + self.round_cost()) > self.policy["budget"]

    def train_local(self, name):
        p = self.policy
        X, y = self.data[name]["X"], self.data[name]["y"]
        Y = np.eye(len(self.classes))[y]
        w = self.w.copy()
        for _ in range(p["epochs"]):
            w = w - p["lr"] * X.T @ (self.predict_proba(w, X) - Y) / len(y)
        update = w - self.w
        raw_norm = float(np.linalg.norm(update))
        if p["dp"]:
            update = update * min(1.0, p["clip"] / (raw_norm + 1e-12))
        self.pending[name] = update
        self.last_train[name] = dict(
            round=self.round + 1,
            local_acc=self.accuracy(w, X, y),
            global_acc=self.accuracy(self.w, X, y),
            raw_norm=raw_norm,
            sent_norm=float(np.linalg.norm(update)),
            update=update,
        )
        self.add_log(name, f"Submitted update for round {self.round + 1}")

    def mask(self, updates):
        masked = [u.copy() for u in updates]
        for i in range(len(masked)):
            for j in range(i + 1, len(masked)):
                m = self.rng.normal(0, 5, masked[0].shape)
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
            self.contrib[n] = self.contrib.get(n, 0) + 1
        acc = self.accuracy(self.w, self.Xte, self.yte)
        self.history.append(
            {"round": self.round, "accuracy": acc, "hospitals": len(names), "epsilon": self.epsilon() if p["dp"] else 0.0}
        )
        self.pending = {}
        self.add_log("server", f"Aggregated round {self.round} from {len(names)} hospitals, accuracy {acc:.1%}")


@st.cache_resource
def get_fed(version=FED_VERSION):
    return Federation()


if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH.name}. Put disease_data.csv in the same folder as this app.")
    st.stop()

fed = get_fed()
if not hasattr(fed, "classes"):
    st.cache_resource.clear()
    fed = get_fed()


def login_as(username):
    st.session_state.user = username
    fed.add_log(username, "Signed in")


def login_page():
    st.title("Federated Healthcare Platform")
    st.caption("Disease prediction from symptoms, trained across hospitals without sharing patient records.")
    t_in, t_up, t_demo = st.tabs(["Sign in", "Sign up", "Demo accounts"])

    with t_in:
        with st.form("signin"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Sign in", type="primary"):
                username = username.strip().lower()
                rec = fed.users.get(username)
                if rec and rec["pw"] == sha(f"{username}:{password}"):
                    login_as(username)
                    st.rerun()
                else:
                    st.error("Invalid username or password")

    with t_up:
        with st.form("signup"):
            hospital = st.text_input("Hospital name")
            new_user = st.text_input("Choose a username")
            new_pass = st.text_input("Choose a password (6+ characters)", type="password")
            if st.form_submit_button("Create account", type="primary"):
                error = fed.sign_up(new_user, new_pass, hospital)
                if error:
                    st.error(error)
                else:
                    st.success("Account created. Wait for admin approval, then sign in.")

    with t_demo:
        for username, rec in fed.users.items():
            if rec["demo"]:
                label = f"Sign in as {username}" + (f" ({rec['hospital']})" if rec["hospital"] else "")
                st.button(label, key=f"demo_{username}", on_click=login_as, args=(username,), width="stretch")


def admin_page():
    st.title("Admin Console")
    names = list(fed.data)
    pending = [n for n in names if not fed.approved[n]]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Hospitals", len(names))
    c2.metric("Approved", len(names) - len(pending))
    c3.metric("Awaiting approval", len(pending))
    c4.metric("Rounds trained", fed.round)
    if pending:
        st.warning("Waiting for approval: " + ", ".join(pending))

    t_hosp, t_policy, t_log = st.tabs(["Hospitals", "Privacy policy", "Audit log"])

    with t_hosp:
        for n in names:
            owner = next((u for u, r in fed.users.items() if r.get("hospital") == n), "-")
            y = fed.data[n]["y"]
            a, b, c, d = st.columns([3, 2, 2, 2])
            a.write(f"**{n}**" + ("  :orange[NEW]" if n in pending else ""))
            b.write(f"{len(y)} patients, {len(np.unique(y))} diseases")
            c.write(f"Account: {owner}")
            state = d.toggle("Approved", fed.approved[n], key=f"ap_{n}")
            if state != fed.approved[n]:
                fed.approved[n] = state
                fed.pending.pop(n, None)
                fed.add_log("admin", f"{'Approved' if state else 'Revoked'} {n}")
                st.rerun()

    with t_policy:
        for k, v in fed.policy.items():
            st.session_state.setdefault(f"pol_{k}", v)

        def apply_preset(name):
            for k, v in PRESETS[name].items():
                st.session_state[f"pol_{k}"] = v

        st.write("Presets")
        cols = st.columns(len(PRESETS))
        for col, name in zip(cols, PRESETS):
            col.button(name, key=f"pre_{name}", on_click=apply_preset, args=(name,), width="stretch")

        left, right = st.columns(2)
        left.toggle("Differential privacy", key="pol_dp")
        left.toggle("Secure aggregation", key="pol_secure")
        left.slider("Clip norm", 0.1, 5.0, key="pol_clip")
        left.slider("Local epochs", 1, 10, key="pol_epochs")
        right.slider("Noise multiplier", 0.1, 5.0, key="pol_sigma")
        right.slider("Privacy budget (epsilon)", 5.0, 500.0, key="pol_budget")
        right.slider("Learning rate", 0.1, 10.0, key="pol_lr")

        if st.session_state.pol_dp:
            rounds_ok = max_rounds(st.session_state.pol_sigma, st.session_state.pol_budget)
            st.info(f"With these settings the budget lasts about {rounds_ok} rounds.")
            st.caption("More noise means stronger privacy but a less accurate model. Compare the presets on the server's Model quality tab.")
        else:
            st.info("Differential privacy is off: no formal privacy guarantee.")

        if st.button("Save policy", type="primary"):
            fed.policy = {k: st.session_state[f"pol_{k}"] for k in fed.policy}
            fed.add_log("admin", f"Policy updated: {fed.policy}")
            st.success("Policy saved")
        if st.button("Reset model and budget"):
            fed.reset()
            fed.add_log("admin", "Model and privacy budget reset")
            st.rerun()

    with t_log:
        log = pd.DataFrame(fed.log[::-1])
        actor = st.selectbox("Filter by actor", ["All"] + sorted(log["actor"].unique()))
        if actor != "All":
            log = log[log["actor"] == actor]
        st.dataframe(log, hide_index=True, width="stretch")
        st.download_button("Download audit log (CSV)", log.to_csv(index=False), "audit_log.csv")


def server_page():
    st.title("Central Server")
    p = fed.policy
    active = [n for n in fed.data if fed.approved[n]]
    acc = fed.history[-1]["accuracy"] if fed.history else 0.0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Round", fed.round)
    c2.metric("Global accuracy (top-1)", f"{acc:.1%}")
    c3.metric("Updates received", f"{len(fed.pending)} / {len(active)}")
    c4.metric("Privacy spent", f"{fed.epsilon():.1f} / {p['budget']:.0f}" if p["dp"] else "DP off")

    t_train, t_quality, t_privacy = st.tabs(["Training", "Model quality", "Privacy view"])

    with t_train:
        status = pd.DataFrame(
            {
                "Hospital": list(fed.data),
                "Approved": [fed.approved[n] for n in fed.data],
                "Update received": [n in fed.pending for n in fed.data],
                "Rounds contributed": [fed.contrib.get(n, 0) for n in fed.data],
            }
        )
        st.dataframe(status, hide_index=True, width="stretch")

        exhausted = fed.budget_exhausted()
        if exhausted:
            st.warning("Privacy budget exhausted. Ask the admin to raise the budget or reset.")
        ready = len([n for n in fed.pending if fed.approved[n]]) >= 2
        a, b = st.columns(2)
        if a.button("Aggregate round", type="primary", disabled=not ready or exhausted, width="stretch"):
            fed.aggregate()
            st.rerun()
        if b.button("Demo: all hospitals train and aggregate", disabled=len(active) < 2 or exhausted, width="stretch"):
            for n in active:
                fed.train_local(n)
            fed.aggregate()
            st.rerun()

        if fed.history:
            hist = pd.DataFrame(fed.history).set_index("round")
            left, right = st.columns(2)
            left.subheader("Global accuracy")
            left.line_chart(hist["accuracy"])
            right.subheader("Rounds contributed")
            right.bar_chart(pd.Series(fed.contrib))
            st.download_button("Download training history (CSV)", hist.to_csv(), "history.csv")

    with t_quality:
        if fed.round == 0:
            st.info("Train at least one round to see model quality.")
        else:
            per = fed.per_disease()
            m1, m2, m3 = st.columns(3)
            m1.metric("Top-1 accuracy", f"{fed.accuracy(fed.w, fed.Xte, fed.yte):.1%}")
            m2.metric("Correct disease in top 3", f"{fed.top_k_accuracy(3):.1%}")
            m3.metric("Diseases recognised 80%+", f"{int((per['Correct'] >= 0.8).sum())} / {len(per)}")
            st.subheader("Hardest diseases for the model")
            st.dataframe(
                per.head(10).assign(Correct=lambda d: d["Correct"].map("{:.0%}".format)),
                hide_index=True,
                width="stretch",
            )
            st.subheader("Most influential symptoms")
            strength = pd.Series(np.linalg.norm(fed.w[:-1], axis=1), index=fed.feature_names)
            st.bar_chart(strength.sort_values(ascending=False).head(10))
            model = dict(diseases=fed.classes, symptoms=fed.feature_names, weights=fed.w.tolist())
            st.download_button("Download global model (JSON)", json.dumps(model), "global_model.json")

    with t_privacy:
        st.write(f"Differential privacy: **{'on' if p['dp'] else 'off'}**. Secure aggregation: **{'on' if p['secure'] else 'off'}**.")
        if fed.last_view:
            label = "Masked update size" if p["secure"] else "Raw update size"
            st.subheader("What the server saw last round")
            st.dataframe(
                pd.DataFrame({"Hospital": list(fed.last_view), label: list(fed.last_view.values())}),
                hide_index=True,
                width="stretch",
            )
        else:
            st.info("No round has been aggregated yet.")
        if p["dp"] and fed.history:
            st.subheader("Privacy budget over time")
            st.line_chart(pd.DataFrame(fed.history).set_index("round")["epsilon"])


def hospital_page(name):
    d = fed.data[name]
    X, y, df = d["X"], d["y"], d["df"]
    approved = fed.approved[name]
    st.title(name)
    if approved:
        st.success("Approved to participate.")
    else:
        st.error("Waiting for admin approval.")

    t_over, t_data, t_train, t_risk = st.tabs(["Overview", "Patient data", "Training", "Diagnosis check"])

    with t_over:
        top_dx = df["diagnosis"].value_counts()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Patients", len(y))
        c2.metric("Diseases seen", len(top_dx))
        c3.metric("Global model on my data", f"{fed.accuracy(fed.w, X, y):.1%}")
        c4.metric("Rounds contributed", fed.contrib.get(name, 0))
        left, right = st.columns(2)
        left.subheader("Most common diagnoses")
        left.bar_chart(top_dx.head(10))
        right.subheader("Most common symptoms")
        right.bar_chart(pd.Series(X[:, :-1].sum(axis=0), index=fed.feature_names).sort_values(ascending=False).head(10))
        st.subheader("My activity")
        mine = pd.DataFrame([e for e in fed.log if e["actor"] == name][::-1])
        if mine.empty:
            st.caption("No activity yet.")
        else:
            st.dataframe(mine[["time", "action"]], hide_index=True, width="stretch")

    with t_data:
        st.caption("Patient records are simulated from the disease and symptom profiles in disease_data.csv.")
        a, b = st.columns(2)
        dx = a.selectbox("Diagnosis", ["All"] + sorted(df["diagnosis"].unique()))
        query = b.text_input("Search patient ID or symptom")
        view = df
        if dx != "All":
            view = view[view["diagnosis"] == dx]
        if query:
            hit = view["patient_id"].str.contains(query, case=False) | view["symptoms"].str.contains(query, case=False)
            view = view[hit]
        st.dataframe(view[KEY_COLS], hide_index=True, width="stretch")
        st.caption(f"{len(view)} of {len(df)} patients shown")
        st.download_button("Export shown rows (CSV)", view.to_csv(index=False), f"{name}_patients.csv")

    with t_train:
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
        info = fed.last_train.get(name)
        if info:
            st.subheader(f"What left your hospital in round {info['round']}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Global model on my data (before)", f"{info['global_acc']:.1%}")
            c2.metric("My model after training", f"{info['local_acc']:.1%}")
            c3.metric("Update size (raw)", f"{info['raw_norm']:.2f}")
            c4.metric("Update size (sent)", f"{info['sent_norm']:.2f}")
            st.caption("Largest changes to the model, summed over diseases for each symptom")
            change = pd.Series(np.linalg.norm(info["update"], axis=1), index=fed.feature_names + ["bias"])
            st.bar_chart(change.sort_values(ascending=False).head(15))

    with t_risk:
        if fed.round == 0:
            st.info("The global model is not trained yet. Ask the server operator to run a round.")
        else:
            pid = st.selectbox("Patient", df["patient_id"])
            i = int(df.index[df["patient_id"] == pid][0])
            st.write(f"**Symptoms:** {df.loc[i, 'symptoms']}")
            probs = fed.predict_proba(fed.w, X[i:i + 1])[0]
            best = np.argsort(-probs)[:3]
            st.subheader("Most likely diseases")
            for k in best:
                st.progress(float(probs[k]), text=f"{fed.classes[k]}: {probs[k]:.0%}")
            actual = df.loc[i, "diagnosis"]
            predicted = fed.classes[best[0]]
            st.write(f"Model's top choice: **{predicted}**. Recorded diagnosis: **{actual}**.")
            if predicted == actual:
                st.success("Top prediction matches the recorded diagnosis.")
            elif actual in [fed.classes[k] for k in best]:
                st.warning("Recorded diagnosis is among the model's top 3, but not the top choice.")
            else:
                st.error("Prediction differs from the recorded diagnosis.")
            contrib = pd.Series(X[i] * fed.w[:, best[0]], index=fed.feature_names + ["bias"])
            contrib = contrib[X[i] != 0]
            top = contrib.reindex(contrib.abs().sort_values(ascending=False).head(6).index)
            st.subheader(f"Why the model leaned towards {predicted}")
            st.bar_chart(top)


if st.session_state.get("user") not in fed.users:
    st.session_state.pop("user", None)
    login_page()
    st.stop()

user = fed.users[st.session_state.user]
with st.sidebar:
    st.write(f"Signed in as **{st.session_state.user}**")
    st.write(user["role"] + (f" at {user['hospital']}" if user["hospital"] else ""))
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
