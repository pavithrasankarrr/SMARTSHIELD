import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="SmartShield IDS", layout="wide")
st.title("🔐 SmartShield: ML-based Intrusion Detection for Wireless Networks")

# ✅ Step 1: Load and preprocess the dataset
@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv("ISS.csv")
    X = df.drop("label", axis=1)
    y = df["label"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return df, X, y, X_scaled, scaler

df, X, y, X_scaled, scaler = load_and_prepare_data()
y = np.array(y)  #Fix writeable flag issue

# ✅ Step 2: Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# ✅ Step 3: Sidebar Input
st.sidebar.header("📡 Simulate Node Input")
packet_size = st.sidebar.slider("Packet Size", 10, 300, 50)
send_rate = st.sidebar.slider("Send Rate (packets/sec)", 1, 50, 5)
neighbor_count = st.sidebar.slider("Neighbor Count", 1, 10, 3)
latency = st.sidebar.slider("Latency (ms)", 5, 100, 20)
node_id = st.sidebar.slider("Node ID", 1, 30, 1)

input_data = pd.DataFrame([[node_id, packet_size, send_rate, neighbor_count, latency]],
                          columns=["node_id", "packet_size", "send_rate", "neighbor_count", "latency"])

input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)[0]
proba = model.predict_proba(input_scaled)[0][1]

# ✅ Step 4: Show Results with Attack Type Classification
st.subheader("🧠 Real-Time Intrusion Prediction")
st.write("**Input Data**:")
st.dataframe(input_data)

if prediction == 1:
    st.error(f"🚨 Intrusion Detected! (Confidence: {proba:.2f})")

    # 🔍 Classify type of attack using feature-based logic
    attack_type = "Unknown"
    if send_rate > 30 and packet_size > 150:
        attack_type = "Flooding / DoS"
    elif send_rate < 2 and latency > 50:
        attack_type = "Blackhole"
    elif neighbor_count > 8 and latency > 60:
        attack_type = "Sinkhole / Wormhole"
    elif packet_size > 200 and send_rate > 20:
        attack_type = "Data Injection"
    elif node_id in [0, 9999]:
        attack_type = "Spoofing"

    st.warning(f"🔍 Likely Attack Type: **{attack_type}**")

else:
    st.success(f"✅ Normal Behavior Detected (Confidence: {1 - proba:.2f})")

# ✅ Step 5: Graph
st.subheader("📊 Data Distribution (Send Rate vs Packet Size)")
df['label'] = df['label'].map({0: 'Normal', 1: 'Anomaly'})
fig, ax = plt.subplots()
sns.scatterplot(data=df, x='send_rate', y='packet_size', hue='label', ax=ax)
plt.xlabel("Send Rate (packets/sec)")
plt.ylabel("Packet Size (bytes)")
plt.grid(True)
st.pyplot(fig)
