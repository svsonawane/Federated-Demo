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

FED_VERSION = 1
DELTA = 1e-5
KEY_COLS = ["patient_id", "diagnosis", "mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness"]
HOSPITAL_NAMES = ["City General", "Riverside Medical", "Northside Clinic", "St. Mary's", "Lakeview Health"]
DEFAULT_POLICY = dict(dp=True, secure=True, clip=1.0, sigma=1.0, lr=0.2, epochs=3, budget=40.0)
PRESETS = {
    "No privacy": dict(dp=False, secure=False, clip=1.0, sigma=1.0, budget=40.0),
    "Balanced": dict(dp=True, secure=True, clip=1.0, sigma=1.0, budget=40.0),
    "Strong privacy": dict(dp=True, secure=True, clip=0.5, sigma=2.0, budget=20.0),
}


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def max_rounds(sigma, budget):
    L = np.log(1 / DELTA)
    rho = (np.sqrt(L + budget) - np.sqrt(L)) ** 2
    return int(rho * 2 * sigma ** 2)


class Federation:
    def __init__(self):
        d = load_breast_cancer()
        self.feature_names = list(d.feature_names)
        raw = pd.DataFrame(d.data, columns=self.feature_names)
        raw.insert(0, "patient_id", [f"P-{i:04d}" for i in range(len(raw))])
        raw["diagnosis"] = np.where(d.target == 1, "Benign", "Malignant")
        scaled = np.hstack([StandardScaler().fit_transform(d.data), np.ones((len(raw), 1))])
        train_idx, test_idx = train_test_split(np.arange(len(raw)), test_size=0.25, random_state=0, stratify=d.target)
        self.Xte, self.yte = scaled[test_idx], d.target[test_idx]
        rng = np.random.default_rng(1)
        order = train_idx[np.argsort(d.target[train_idx] + rng.normal(0, 0.9, len(train_idx)))]
        shards = [
            dict(X=scaled[p], y=d.target[p], df=raw.iloc[p].reset_index(drop=True))
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
        self.w = np.zeros(self.Xte.shape[1])
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

    def accuracy(self, w, X, y):
        return float(((sigmoid(X @ w) > 0.5) == y).mean())

    def epsilon(self, rho=None):
        rho = self.rho if rho is None else rho
        return 0.0 if rho == 0 else rho + 2 * np.sqrt(rho * np.log(1 / DELTA))

    def round_cost(self):
        return 1 / (2 * self.policy["sigma"] ** 2)

    def budget_exhausted(self):
        return self.policy["dp"] and self.epsilon(self.rho + self.round_cost()) > self.policy["budget"]

    def confusion(self):
        pred = (sigmoid(self.Xte @ self.w) > 0.5).astype(int)
        y = self.yte
        tp = int(((pred == 0) & (y == 0)).sum())
        fn = int(((pred == 1) & (y == 0)).sum())
        fp = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 1) & (y == 1)).sum())
        return tp, fn, fp, tn

    def train_local(self, name):
        p = self.policy
        X, y = self.data[name]["X"], self.data[name]["y"]
        w = self.w.copy()
        for _ in range(p["epochs"]):
            w = w - p["lr"] * X.T @ (sigmoid(X @ w) - y) / len(y)
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


fed = get_fed()
if not hasattr(fed, "users"):
    st.cache_resource.clear()
    fed = get_fed()


def login_as(username):
    st.session_state.user = username
    fed.add_log(username, "Signed in")


def login_page():
    st.title("Federated Healthcare Platform")
    st.caption("Hospitals learn together without ever sharing patient records.")
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
        st.caption("Register a new hospital. An admin must approve it before it can join training.")
        with st.form("signup"):
            hospital = st.text_input("Hospital name")
            new_user = st.text_input("Choose a username")
            new_pass = st.text_input("Choose a password (6+ characters)", type="password")
            if st.form_submit_button("Create account", type="primary"):
                error = fed.sign_up(new_user, new_pass, hospital)
                if error:
                    st.error(error)
                else:
                    st.success("Account created. Open the Sign in tab. An admin must approve your hospital before you can train.")

    with t_demo:
        st.caption("One click to explore each profile.")
        for username, rec in fed.users.items():
            if not rec["demo"]:
                continue
            c1, c2 = st.columns([1, 3])
            c1.button(f"Sign in as {username}", key=f"demo_{username}", on_click=login_as, args=(username,), width="stretch")
            if rec["role"] == "Hospital":
                c2.write(f"Hospital: {rec['hospital']}. Local patients, training, risk checks.")
            elif rec["role"] == "Admin":
                c2.write("Admin: privacy policy, hospital approvals, audit log.")
            else:
                c2.write("Central Server: aggregates updates, tracks model and privacy budget.")


def how_it_works_page():
    st.title("How it works")
    st.markdown(
        """
**Federated learning.** Each hospital keeps its patient records. Instead of sending data to a central place, every hospital trains the model on its own patients and sends back only a small list of numbers (a model update). The central server averages the updates into a better global model and sends it back. Repeat for many rounds.

**The three profiles**
- **Hospital**: owns private patient data, trains locally, submits updates, uses the global model for risk checks.
- **Central Server**: collects updates, runs aggregation, tracks accuracy and the privacy budget. It never sees patient records.
- **Admin**: sets the privacy policy, approves hospitals, and reads the audit log.

**Differential privacy (DP).** Two steps protect against someone learning about one patient from the model. *Clipping* limits how much any single hospital can move the model. *Noise* is then added to the combined update. The more noise, the more privacy and the lower the accuracy.

**Privacy budget (epsilon).** Every round leaks a little. Epsilon measures the total leak. A smaller number means stronger privacy. When the budget is used up, training stops.

**Secure aggregation.** Hospitals add random masks to their updates that cancel out when everything is summed. The server learns the sum but not any single hospital's update.

**Reading the model.** The dataset is the Wisconsin breast cancer set. Malignant detection rate (sensitivity) matters most, because missing a malignant case is the costly mistake.

**Limits of this demo.** Secure aggregation is simulated inside one program, noise is added by a trusted server, and the epsilon is an approximate accounting. State resets when the app restarts.
        """
    )


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
            b.write(f"{len(y)} patients, {(1 - y.mean()):.0%} malignant")
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
        right.slider("Noise multiplier", 0.3, 5.0, key="pol_sigma")
        right.slider("Privacy budget (epsilon)", 5.0, 200.0, key="pol_budget")
        right.slider("Learning rate", 0.01, 1.0, key="pol_lr")

        if st.session_state.pol_dp:
            rounds_ok = max_rounds(st.session_state.pol_sigma, st.session_state.pol_budget)
            st.info(f"With these settings the budget lasts about {rounds_ok} rounds.")
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
    c2.metric("Global accuracy", f"{acc:.1%}")
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
        if not ready:
            st.caption("Aggregation needs updates from at least 2 approved hospitals.")

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
            tp, fn, fp, tn = fed.confusion()
            m1, m2, m3 = st.columns(3)
            m1.metric("Malignant detection rate", f"{tp / max(tp + fn, 1):.1%}")
            m2.metric("Benign correctly cleared", f"{tn / max(tn + fp, 1):.1%}")
            m3.metric("Missed malignant cases", fn)
            st.dataframe(
                pd.DataFrame(
                    [[tp, fn], [fp, tn]],
                    index=["Actually malignant", "Actually benign"],
                    columns=["Predicted malignant", "Predicted benign"],
                ),
                width="stretch",
            )
            w = pd.Series(fed.w[:-1], index=fed.feature_names)
            top = w.reindex(w.abs().sort_values(ascending=False).head(8).index)
            st.subheader("Most influential features")
            st.bar_chart(top)
            st.caption("Positive weights push predictions toward benign, negative toward malignant.")
            st.download_button("Download global model (JSON)", json.dumps(fed.w.tolist()), "global_model.json")

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
            st.caption("With secure aggregation the masks hide each hospital's real update. Only the sum is meaningful.")
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
        st.success("Approved to participate. Patient records never leave this hospital.")
    else:
        st.error("Waiting for admin approval. You can explore your data but cannot train yet.")

    t_over, t_data, t_train, t_risk = st.tabs(["Overview", "Patient data", "Training", "Risk check"])

    with t_over:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Patients", len(y))
        c2.metric("Malignant cases", f"{(1 - y.mean()):.0%}")
        c3.metric("Global model on my data", f"{fed.accuracy(fed.w, X, y):.1%}")
        c4.metric("Rounds contributed", fed.contrib.get(name, 0))
        left, right = st.columns(2)
        left.subheader("Diagnoses")
        left.bar_chart(df["diagnosis"].value_counts())
        right.subheader("Radius vs texture")
        right.scatter_chart(df, x="mean radius", y="mean texture", color="diagnosis")
        st.subheader("My activity")
        mine = pd.DataFrame([e for e in fed.log if e["actor"] == name][::-1])
        if mine.empty:
            st.caption("No activity yet.")
        else:
            st.dataframe(mine[["time", "action"]], hide_index=True, width="stretch")

    with t_data:
        st.caption("This table exists only at your hospital. It is never sent to the server.")
        a, b, c = st.columns(3)
        dx = a.selectbox("Diagnosis", ["All", "Malignant", "Benign"])
        query = b.text_input("Search patient ID")
        full = c.toggle("Show all 30 features")
        view = df
        if dx != "All":
            view = view[view["diagnosis"] == dx]
        if query:
            view = view[view["patient_id"].str.contains(query, case=False)]
        st.dataframe(view if full else view[KEY_COLS], hide_index=True, width="stretch")
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
        p = fed.policy
        st.caption(
            f"Policy: DP {'on' if p['dp'] else 'off'}, secure aggregation {'on' if p['secure'] else 'off'}, "
            f"clip {p['clip']}, noise {p['sigma']}, {p['epochs']} local epochs"
        )
        info = fed.last_train.get(name)
        if info:
            st.subheader(f"What left your hospital in round {info['round']}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Global model on my data (before)", f"{info['global_acc']:.1%}")
            c2.metric("My model after training", f"{info['local_acc']:.1%}")
            c3.metric("Update size (raw)", f"{info['raw_norm']:.2f}")
            c4.metric("Update size (sent)", f"{info['sent_norm']:.2f}")
            st.bar_chart(pd.Series(info["update"], index=fed.feature_names + ["bias"]))
            st.caption(f"Only these {len(info['update'])} numbers are shared, not your {len(y)} patient records.")

    with t_risk:
        if fed.round == 0:
            st.info("The global model is not trained yet. Ask the server operator to run a round.")
        else:
            pid = st.selectbox("Patient", df["patient_id"])
            i = int(df.index[df["patient_id"] == pid][0])
            benign = float(sigmoid(X[i] @ fed.w))
            st.progress(benign, text=f"Predicted probability benign: {benign:.0%}")
            actual = df.loc[i, "diagnosis"]
            predicted = "Benign" if benign > 0.5 else "Malignant"
            st.write(f"Model says **{predicted}**. Recorded diagnosis: **{actual}**.")
            if predicted == actual:
                st.success("Prediction matches the recorded diagnosis.")
            else:
                st.error("Prediction differs from the recorded diagnosis.")
            contrib = pd.Series(X[i] * fed.w, index=fed.feature_names + ["bias"])
            top = contrib.reindex(contrib.abs().sort_values(ascending=False).head(6).index)
            st.subheader("Why the model decided this")
            st.bar_chart(top)
            st.caption("Positive bars push toward benign, negative toward malignant.")


if st.session_state.get("user") not in fed.users:
    st.session_state.pop("user", None)
    login_page()
    st.stop()

user = fed.users[st.session_state.user]
with st.sidebar:
    st.write(f"Signed in as **{st.session_state.user}**")
    st.write(user["role"] + (f" at {user['hospital']}" if user["hospital"] else ""))
    page = st.radio("Page", ["Workspace", "How it works"])
    if st.button("Refresh"):
        st.rerun()
    if st.button("Sign out"):
        fed.add_log(st.session_state.user, "Signed out")
        del st.session_state.user
        st.rerun()

if page == "How it works":
    how_it_works_page()
elif user["role"] == "Admin":
    admin_page()
elif user["role"] == "Central Server":
    server_page()
else:
    hospital_page(user["hospital"])
