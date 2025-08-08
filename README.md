# 🛍️ Retail Customer Churn Prediction

Welcome to this hands-on machine learning project where we explore how to predict customer churn in a retail setting. Imagine you’re working at a company like JB Hi-Fi or The Good Guys. One big challenge is knowing **which customers are likely to stop buying** — and what you can do about it.

This project shows how to use real-world transaction data to identify those at-risk customers, predict the likelihood of churn, and suggest actionable strategies to keep them engaged. Along the way, you’ll also learn how to engineer features, build a predictive model, evaluate it properly, and deploy a simple web app to explore the results.

👉 **Try the live dashboard here:**  
[(https://retail-churn-prediction-kpcyarztegv8uslqcuvh8l.streamlit.app/)]

---

## 🎯 What’s the Goal?

In retail, it’s often more expensive to attract new customers than to retain existing ones. So, knowing **who is about to churn (stop buying)** can help businesses take proactive steps — like sending a discount, reminding them of a loyalty program, or reaching out with a personal offer.

The main goals of this project are:

- To identify customers likely to churn using historical transaction data
- To group customers into risk segments (High, Medium, Low)
- To recommend personalized retention actions
- To present everything in a friendly dashboard with filters and visualizations

---

## 📦 What Data Are We Using?

We use the [UCI Online Retail Dataset](https://archive.ics.uci.edu/ml/datasets/Online+Retail), which contains transaction logs from a UK-based online retailer. Each row represents an item purchased in a single invoice. From this, we calculate customer-level metrics using the **RFM framework**:

- **Recency** – How recently did the customer make a purchase?
- **Frequency** – How often do they purchase?
- **Monetary** – How much have they spent?

We also define churn labels. If a customer hasn’t made a purchase in the last 6 months, we label them as **churned (1)**. Otherwise, they’re still active (0).

---

## 🤖 What ML Model Did We Use?

We used a **Random Forest Classifier** — a type of ensemble model that builds multiple decision trees and combines their results. It’s powerful, easy to interpret, and works well with structured data like ours.

The model is trained on the RFM features to predict whether a customer has churned. It also outputs **probabilities**, which we use to classify customers into:

- **High Risk** – Likely to churn (e.g. prob > 0.7)
- **Medium Risk** – Uncertain zone
- **Low Risk** – Probably loyal

Based on these risk levels, we generate business suggestions:
- High Risk → Send discount
- Medium Risk → Send reminder or feedback survey
- Low Risk → Promote loyalty club or upsell

---

## 📊 What Does the Output Look Like?

After the model is trained and evaluated, we generate a final table of predictions for each customer, including:

| CustomerID | Recency | Frequency | Monetary | ChurnProbability | Segment     | SuggestedAction        |
|------------|---------|-----------|----------|------------------|-------------|-------------------------|
| 12345      | 210     | 1         | $120     | 0.91             | High Risk   | Send 20% voucher        |
| 24680      | 40      | 4         | $600     | 0.08             | Low Risk    | Promote loyalty club    |
| 54321      | 185     | 2         | $180     | 0.67             | Medium Risk | Send retention email    |

This output is saved as a CSV file and also visualized through the interactive app.

---

## 🧪 How Did We Evaluate the Model?

To assess how well our model works, we used:

- **Confusion Matrix** – to see where the model gets it right and wrong
- **Classification Report** – shows precision, recall, and F1-score
- **ROC AUC Score** – tells us how well the model ranks churners vs non-churners
- **ROC Curve Plot** – helps us visualize performance across thresholds

All plots are saved under the `/output` folder and are also shown in the app.

---

## 📈 Live Dashboard with Streamlit

We used [Streamlit](https://streamlit.io/) to build a user-friendly dashboard. On the dashboard, you can:

- Filter customers by risk level (High, Medium, Low)
- View a pie chart of customer distribution
- See churn probabilities and suggested actions
- Download filtered results as CSV
- Explore model evaluation plots like ROC curve and confusion matrix

👉 **Launch the app here:**  
[https://retail-churn-prediction-aj325zdzvcvu8wl94bwsdn.streamlit.app/](https://retail-churn-prediction-aj325zdzvcvu8wl94bwsdn.streamlit.app/)

---

## 🛠️ Tech Stack

This project uses:

- Python 3
- pandas & numpy (for data handling)
- scikit-learn (for modeling)
- matplotlib & seaborn (for plots)
- plotly (for interactive charts)
- Streamlit (for the web dashboard)
- Git & GitHub (for version control)

---

## 📁 Project Structure

```
retail-churn-prediction/
├── data/               # Raw and processed data
├── notebooks/          # EDA and RFM feature engineering
├── src/                # model.py and evaluation scripts
├── output/             # CSV results and plots
├── app/                # Streamlit dashboard
├── model/              # Saved model
└── README.md
```

---

## 🧑‍🏫 Want to Learn From It?

This project is built to be a full learning resource. You’ll learn how to:

- Clean and prepare real-world retail data
- Engineer features like Recency, Frequency, and Monetary
- Define churn and build a target variable
- Train a classification model and tune it
- Evaluate model performance visually
- Translate ML output into real business decisions
- Build and deploy a Streamlit dashboard to share your results

---

## 🚀 How to Run Locally

```bash
git clone https://github.com/My-projects-portfolio/retail-churn-prediction.git
cd retail-churn-prediction
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

---

## 👋 About the Author

This project was developed by [Your Name] as part of a machine learning portfolio focused on real-world use cases in retail. The goal was not just to build a good model, but to connect data science with actual business value.

📧 Email: your.email@example.com  
🔗 LinkedIn: [linkedin.com/in/your-profile](https://linkedin.com/in/your-profile)

---

Thanks for reading, and happy predicting!
