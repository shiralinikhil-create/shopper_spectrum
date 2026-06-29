import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide"
)

# ══════════════════════════════════════════════════════════════════════════
# CUSTOM CSS STYLING
# ══════════════════════════════════════════════════════════════════════════
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .title {
        font-size: 42px;
        font-weight: bold;
        color: #1a1a2e;
        text-align: center;
        padding: 10px;
    }
    .subtitle {
        font-size: 18px;
        color: #555;
        text-align: center;
        margin-bottom: 30px;
    }
    .card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 15px;
    }
    .segment-badge {
        font-size: 24px;
        font-weight: bold;
        padding: 15px 30px;
        border-radius: 10px;
        text-align: center;
        margin: 10px 0;
    }
    .high-value  { background-color: #d4edda; color: #155724; }
    .regular     { background-color: #cce5ff; color: #004085; }
    .occasional  { background-color: #fff3cd; color: #856404; }
    .at-risk     { background-color: #f8d7da; color: #721c24; }
    .rec-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 12px 18px;
        border-radius: 8px;
        margin: 6px 0;
        font-size: 15px;
        font-weight: 500;
    }
    </style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# LOAD MODELS
# ══════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_models():
    similarity_df = joblib.load('similarity_matrix.pkl')
    kmeans        = joblib.load('kmeans_model.pkl')
    product_list  = joblib.load('product_list.pkl')
    return similarity_df, kmeans, product_list

# Scaler params hardcoded (no need to upload scaler.pkl)
SCALER_MEAN  = [3.8307338678832883, 1.3455823418447785, 6.593627140775707]
SCALER_SCALE = [1.3401066638714403, 0.6830249359732609, 1.2574326702312202]

def scale_input(recency, frequency, monetary):
    log_vals = np.log1p([recency, frequency, monetary])
    scaled   = [(log_vals[i] - SCALER_MEAN[i]) / SCALER_SCALE[i] for i in range(3)]
    return np.array(scaled).reshape(1, -1)

similarity_df, kmeans, product_list = load_models()

@st.cache_data
def load_data():
    df = pd.read_csv("online_retail_cleaned.csv.gz", compression="gzip")
    return df

# Cluster → Segment label mapping (based on our clustering results)
SEGMENT_MAP = {0: 'Occasional', 1: 'High-Value', 2: 'Regular', 3: 'At-Risk'}

# ══════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="title">🛒 Shopper Spectrum</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Customer Segmentation & Product Recommendation System</div>',
            unsafe_allow_html=True)
st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════
tab1, tab2 = st.tabs(["🎯 Product Recommendations", "👥 Customer Segmentation"])

# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — PRODUCT RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("## 🎯 Product Recommendation")
    st.markdown("Enter a product name and get **5 similar product recommendations** instantly!")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        # Text input
        product_input = st.text_input(
            "🔍 Enter Product Name",
            placeholder="e.g. WHITE HANGING HEART T-LIGHT HOLDER",
            help="Type part of a product name to search"
        )

        # Show sample products
        with st.expander("💡 Sample Product Names to Try"):
            samples = [
                "WHITE HANGING HEART T-LIGHT HOLDER",
                "JUMBO BAG",
                "LUNCH BAG",
                "METAL SIGN",
                "ROSES REGENCY TEACUP AND SAUCER",
                "SET OF 3 CAKE TINS",
                "VINTAGE UNION JACK",
            ]
            for s in samples:
                st.code(s)

        get_btn = st.button("🚀 Get Recommendations", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📊 System Info")
        st.metric("Total Products", f"{len(product_list):,}")
        st.metric("Algorithm", "Cosine Similarity")
        st.metric("Recommendations", "Top 5")
        st.markdown('</div>', unsafe_allow_html=True)

    # Show recommendations
    if get_btn:
        if not product_input.strip():
            st.warning("⚠️ Please enter a product name!")
        else:
            # Find matching products
            matches = [p for p in similarity_df.index
                       if product_input.upper() in p.upper()]

            if not matches:
                st.error(f"❌ No product found matching **'{product_input}'**. Try a different keyword.")
            else:
                matched = matches[0]
                st.success(f"✅ Matched: **{matched}**")

                # Get top 5 similar products
                similar = (similarity_df[matched]
                           .drop(matched)
                           .sort_values(ascending=False)
                           .head(5))

                st.markdown("### 🎁 Top 5 Recommended Products")
                for i, (product, score) in enumerate(similar.items(), 1):
                    st.markdown(
                        f'<div class="rec-card">#{i} &nbsp; {product} &nbsp;&nbsp; '
                        f'<span style="opacity:0.8; font-size:13px;">Similarity: {score:.4f}</span></div>',
                        unsafe_allow_html=True
                    )

                # Similarity bar chart
                st.markdown("### 📊 Similarity Scores")
                chart_data = pd.DataFrame({
                    'Product': [p[:30] + '...' if len(p) > 30 else p for p in similar.index],
                    'Similarity Score': similar.values
                })
                st.bar_chart(chart_data.set_index('Product'))

# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — CUSTOMER SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## 👥 Customer Segmentation")
    st.markdown("Enter customer **RFM values** to predict which segment they belong to!")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📥 Enter Customer Details")

        recency   = st.number_input(
            "📅 Recency (days since last purchase)",
            min_value=1, max_value=400, value=30,
            help="How many days ago did the customer last purchase?"
        )
        frequency = st.number_input(
            "🔁 Frequency (number of orders)",
            min_value=1, max_value=300, value=5,
            help="How many times has the customer ordered?"
        )
        monetary  = st.number_input(
            "💰 Monetary (total amount spent £)",
            min_value=1.0, max_value=300000.0, value=500.0, step=50.0,
            help="What is the total amount the customer has spent?"
        )

        predict_btn = st.button("🔮 Predict Segment", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🏷️ Segment Guide")

        segments_info = {
            "🔵 High-Value":  "Recent, frequent buyers with high spend",
            "🟢 Regular":     "Steady buyers with moderate spend",
            "🟠 Occasional":  "Infrequent buyers with low spend",
            "🔴 At-Risk":     "Haven't purchased in a long time",
        }
        for seg, desc in segments_info.items():
            st.markdown(f"**{seg}** — {desc}")

        st.markdown("---")
        st.markdown("### 📊 Dataset Segments")
        seg_counts = {"High-Value": 716, "Regular": 1173, "Occasional": 837, "At-Risk": 1612}
        seg_df = pd.DataFrame(list(seg_counts.items()), columns=["Segment", "Customers"])
        st.bar_chart(seg_df.set_index("Segment"))
        st.markdown('</div>', unsafe_allow_html=True)

    # Show prediction result
    if predict_btn:
        # Scale input (using hardcoded scaler params)
        input_scaled = scale_input(recency, frequency, monetary)
        cluster      = kmeans.predict(input_scaled)[0]
        segment      = SEGMENT_MAP.get(cluster, "Unknown")

        # Style per segment
        style_map = {
            "High-Value": ("high-value",  "🔵", "💎 Premium Customer!"),
            "Regular":    ("regular",     "🟢", "👍 Loyal Customer!"),
            "Occasional": ("occasional",  "🟠", "💡 Needs Engagement!"),
            "At-Risk":    ("at-risk",     "🔴", "⚠️ Needs Retention!"),
        }
        css_class, emoji, message = style_map.get(segment, ("occasional", "⚪", ""))

        st.markdown("---")
        st.markdown("### 🎯 Prediction Result")

        st.markdown(
            f'<div class="segment-badge {css_class}">'
            f'{emoji} &nbsp; {segment} Customer &nbsp; {emoji}'
            f'</div>',
            unsafe_allow_html=True
        )
        st.markdown(f"<h4 style='text-align:center'>{message}</h4>", unsafe_allow_html=True)

        # Show input summary
        st.markdown("#### 📋 Input Summary")
        summary = pd.DataFrame({
            "Metric":  ["Recency", "Frequency", "Monetary"],
            "Value":   [f"{recency} days", f"{frequency} orders", f"£{monetary:,.2f}"],
            "Rating":  [
                "🟢 Recent" if recency < 30 else ("🟠 Moderate" if recency < 100 else "🔴 Long ago"),
                "🟢 High"   if frequency > 5 else ("🟠 Medium"  if frequency > 2  else "🔴 Low"),
                "🟢 High"   if monetary > 2000 else ("🟠 Medium" if monetary > 500 else "🔴 Low"),
            ]
        })
        st.table(summary)

        # Business recommendation
        st.markdown("#### 💼 Business Action")
        actions = {
            "High-Value":  "🎁 Offer loyalty rewards, exclusive deals and early access to new products.",
            "Regular":     "📧 Send personalized emails with product recommendations to increase spend.",
            "Occasional":  "🎯 Use targeted ads and discount coupons to increase purchase frequency.",
            "At-Risk":     "🚨 Send win-back campaigns with special offers to re-engage this customer.",
        }
        st.info(actions.get(segment, ""))

# ══════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#888; font-size:13px'>"
    "🛒 Shopper Spectrum | Customer Segmentation & Product Recommendation | E-Commerce Analytics"
    "</div>",
    unsafe_allow_html=True
)
