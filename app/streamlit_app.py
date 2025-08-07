# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("output/model_predictions.csv")  # <-- from app/
    return df

df = load_data()

# Title
st.title("🛍️ Retail Churn Prediction Dashboard")

# Sidebar Filters
st.sidebar.header("Filter Customers")
segment_filter = st.sidebar.multiselect("Select Risk Segment", options=df['Segment'].unique(), default=df['Segment'].unique())
df_filtered = df[df['Segment'].isin(segment_filter)]

# Overview Stats
st.subheader("📊 Customer Segments Summary")
segment_counts = df_filtered['Segment'].value_counts().reset_index()
segment_counts.columns = ['Segment', 'Count']
st.dataframe(segment_counts)

# Plot Segment Distribution
import plotly.express as px
fig = px.pie(segment_counts, values='Count', names='Segment', title='Customer Risk Segment Distribution')
st.plotly_chart(fig)

# Show Prediction Table
st.subheader("🧾 Prediction Results")
st.dataframe(df_filtered.sort_values(by='ChurnProbability', ascending=False).reset_index(drop=True))

# Download filtered results
csv = df_filtered.to_csv(index=False).encode('utf-8')
st.download_button("Download Filtered Predictions", csv, "filtered_predictions.csv", "text/csv")

# Show plots
st.subheader("📉 Evaluation Plots")
st.image("output/confusion_matrix.png", caption="Confusion Matrix", use_column_width=True)
st.image("output/roc_curve.png", caption="ROC Curve", use_column_width=True)
