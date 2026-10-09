import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Federated Healthcare", layout="wide")
st.title("Federated Learning for Healthcare")
st.caption("Hospitals train locally on breast cancer data. Only model updates are shared, never patient records.")

with st.sidebar:
    st.header("Settings")
    n_hosp = st.slider("Hospitals", 3, 10, 5)
    rounds = st.slider("Training rounds", 5, 50, 20)
    epochs = st.slider("Local epochs", 1, 10, 3)
    lr = st.slider("Learning rate", 0.01, 1.0, 0.2)
    non_iid = st.checkbox("Non-IID data across hospitals", True)
    use_dp = st.checkbox("Differential privacy", True)
    clip = st.slider("Clip norm", 0.1, 5.0, 1.0)
    sigma = st.slider("Noise multiplier", 0.1, 5.0, 1.0)
    secure = st.checkbox("Secure aggregation", True)
    delta = 1e-5
    seed = st.number_input("Seed", 0, 999, 42)


@st.cache_data
def load():
    d = load_breast_cancer()
    X = StandardScaler().fit_transform(d.data)
    return train_test_split(X, d.target, test_size=0.25, random_state=0, stratify=d.target)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def add_bias(X):
    return np.hstack([X, np.ones((len(X), 1))])


def accuracy(w, X, y):
    return float(((sigmoid(X @ w) > 0.5) == y).mean())


def split(X, y, n, non_iid, rng):
    idx = rng.permutation(len(y))
    if non_iid:
        idx = idx[np.argsort(y[idx] + rng.normal(0, 0.8, len(y)))]
    return [(X[p], y[p]) for p in np.array_split(idx, n)]


def local_update(w, X, y):
    w0 = w.copy()
    for _ in range(epochs):
        w = w - lr * X.T @ (sigmoid(X @ w) - y) / len(y)
    return w - w0


def clip_update(u):
    return u * min(1.0, clip / (np.linalg.norm(u) + 1e-12))


def mask_updates(updates, rng):
    n, dim = len(updates), len(updates[0])
    masked = [u.copy() for u in updates]
    for i in range(n):
        for j in range(i + 1, n):
            m = rng.normal(0, 5, dim)
            masked[i] += m
            masked[j] -= m
    return masked


def epsilon(R):
    rho = R / (2 * sigma ** 2)
    return rho + 2 * np.sqrt(rho * np.log(1 / delta))


Xtr, Xte, ytr, yte = load()
Xtr, Xte = add_bias(Xtr), add_bias(Xte)

if st.button("Start federated training", type="primary"):
    rng = np.random.default_rng(int(seed))
    clients = split(Xtr, ytr, n_hosp, non_iid, rng)
    w = np.zeros(Xtr.shape[1])
    history, leak_gap = [], 0.0

    for r in range(rounds):
        updates = [local_update(w, X, y) for X, y in clients]
        if use_dp:
            updates = [clip_update(u) for u in updates]
        sent = mask_updates(updates, rng) if secure else updates
        if r == 0:
            leak_gap = float(np.linalg.norm(sent[0] - updates[0]))
        total = np.sum(sent, axis=0)
        if use_dp:
            total = total + rng.normal(0, sigma * clip, total.shape)
        w = w + total / n_hosp
        history.append(accuracy(w, Xte, yte))

    central = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    central_acc = central.score(Xte, yte)
    local_accs = []
    for X, y in clients:
        if len(np.unique(y)) > 1:
            local_accs.append(LogisticRegression(max_iter=1000).fit(X, y).score(Xte, yte))
        else:
            local_accs.append(float((yte == y[0]).mean()))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Federated accuracy", f"{history[-1]:.1%}")
    c2.metric("Centralized (no privacy)", f"{central_acc:.1%}")
    c3.metric("Avg hospital alone", f"{np.mean(local_accs):.1%}")
    c4.metric("Privacy budget (ε)", f"{epsilon(rounds):.1f}" if use_dp else "None")

    left, right = st.columns(2)
    with left:
        st.subheader("Accuracy per round")
        st.line_chart(pd.DataFrame({"Federated": history, "Centralized": central_acc}))
    with right:
        st.subheader("Hospital data")
        st.dataframe(
            pd.DataFrame(
                {
                    "Hospital": [f"Hospital {i + 1}" for i in range(n_hosp)],
                    "Patients": [len(y) for _, y in clients],
                    "Malignant %": [round(100 * (1 - y.mean()), 1) for _, y in clients],
                    "Alone accuracy": [round(a, 3) for a in local_accs],
                }
            ),
            hide_index=True,
            width="stretch",
        )

    st.subheader("Privacy status")
    st.write(f"Differential privacy: {'on' if use_dp else 'off'}")
    if secure:
        st.write(f"Secure aggregation: on. A masked update differs from the real one by {leak_gap:.1f}, but masks cancel in the sum.")
    else:
        st.write("Secure aggregation: off. The server sees each hospital's raw update.")
else:
    st.info("Choose settings on the left and press Start.")
