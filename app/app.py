"""
AI-Based Market Trend Analysis: Streamlit Application.

Forecasts agricultural product prices from the included synthetic onion-price
dataset. The repository does not include trained model artifacts, so the app
uses transparent statistical baselines for the interactive demo.
"""

from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "onion_prices_synthetic.csv"


st.set_page_config(
    page_title="AI Market Trend Analyzer",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data() -> pd.DataFrame | None:
    """Load the included synthetic onion price dataset."""
    try:
        df = pd.read_csv(DATA_PATH)
        df["date"] = pd.to_datetime(df["date"])
        return df
    except FileNotFoundError:
        st.error(f"Data file not found: {DATA_PATH}")
    except Exception as exc:
        st.error(f"Error loading data: {exc}")
    return None


def predict_price_for_date(df: pd.DataFrame, selected_date) -> float:
    """Estimate price from same-month history plus a simple annual trend."""
    month = selected_date.month
    year = selected_date.year
    monthly_average = df.loc[df["month"] == month, "price_rupees_per_kg"].mean()
    annual_growth = 1 + (year - int(df["year"].min())) * 0.06
    seasonal_adjustment = np.sin(2 * np.pi * selected_date.timetuple().tm_yday / 365.25)
    return float(monthly_average * annual_growth + seasonal_adjustment)


def generate_30_day_forecast(df: pd.DataFrame) -> pd.DataFrame:
    """Generate a deterministic 30-day forecast from recent price movement."""
    last_date = df["date"].max()
    forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=30, freq="D")
    recent_prices = df.tail(30)["price_rupees_per_kg"].to_numpy()
    slope, intercept = np.polyfit(np.arange(len(recent_prices)), recent_prices, 1)

    prices = []
    for offset in range(30):
        trend_price = slope * (len(recent_prices) + offset) + intercept
        seasonal_factor = 2 * np.sin(2 * np.pi * offset / 30)
        prices.append(max(trend_price + seasonal_factor, 0))

    return pd.DataFrame({"Date": forecast_dates, "Predicted Price": prices})


st.markdown('<div class="main-header">AI Market Trend Analyzer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Forecasting agricultural product prices from synthetic market data<br>'
    "Dataset: 2022-2025 | Forecast target: 2026</div>",
    unsafe_allow_html=True,
)

df = load_data()

if df is None:
    st.stop()

st.sidebar.title("Configuration")
st.sidebar.markdown("---")
model_choice = st.sidebar.selectbox(
    "Select Prediction Approach",
    ["Seasonal Baseline", "Recent Trend Forecast"],
    index=0,
)

st.sidebar.markdown("### Dataset Info")
st.sidebar.info(
    f"""
    **Records:** {len(df):,}

    **Date Range:** {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}

    **Features:** {len(df.columns)}
    """
)

st.sidebar.markdown("### Price Statistics")
st.sidebar.metric("Average Price", f"Rs. {df['price_rupees_per_kg'].mean():.2f}/kg")
st.sidebar.metric("Min Price", f"Rs. {df['price_rupees_per_kg'].min():.2f}/kg")
st.sidebar.metric("Max Price", f"Rs. {df['price_rupees_per_kg'].max():.2f}/kg")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Data Overview", "Price Prediction", "Historical Analysis", "About"]
)

with tab1:
    st.header("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Avg Price", f"Rs. {df['price_rupees_per_kg'].mean():.2f}")
    col3.metric("Avg Volume", f"{df['volume_quintals'].mean():.0f} Q")
    promo_pct = (df["promo_flag"].sum() / len(df)) * 100
    col4.metric("Promo Days", f"{promo_pct:.1f}%")

    st.markdown("---")
    st.subheader("Sample Data")
    sample_indices = np.linspace(0, len(df) - 1, 20, dtype=int)
    st.dataframe(df.iloc[sample_indices], use_container_width=True)

    st.subheader("Price Trend Over Time")
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df["date"], df["price_rupees_per_kg"], linewidth=1.5, color="#1f77b4")
    ax.set_xlabel("Date", fontsize=11, fontweight="bold")
    ax.set_ylabel("Price (Rs./kg)", fontsize=11, fontweight="bold")
    ax.set_title("Onion Price Trend (2022-2025)", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Price Distribution")
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(df["price_rupees_per_kg"], bins=40, edgecolor="black", alpha=0.7, color="#2ca02c")
        ax.set_xlabel("Price (Rs./kg)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Frequency", fontsize=10, fontweight="bold")
        ax.set_title("Price Distribution", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.subheader("Monthly Average Prices")
        monthly_avg = df.groupby("month")["price_rupees_per_kg"].mean()
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(monthly_avg.index, monthly_avg.values, color="#ff7f0e", alpha=0.7, edgecolor="black")
        ax.set_xlabel("Month", fontsize=10, fontweight="bold")
        ax.set_ylabel("Average Price (Rs./kg)", fontsize=10, fontweight="bold")
        ax.set_title("Average Price by Month", fontsize=12, fontweight="bold")
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        st.pyplot(fig)

with tab2:
    st.header("Price Prediction")
    st.info(f"Selected approach: {model_choice}")

    pred_option = st.radio(
        "Prediction Mode",
        ["Single Date Prediction", "Next 30 Days Forecast"],
        horizontal=True,
    )

    if pred_option == "Single Date Prediction":
        st.subheader("Select Date for Prediction")
        selected_date = st.date_input(
            "Choose a date",
            value=datetime(2026, 6, 15),
            min_value=datetime(2022, 1, 1),
            max_value=datetime(2027, 12, 31),
        )

        if st.button("Predict Price", type="primary"):
            predicted_price = predict_price_for_date(df, selected_date)
            lower_bound = predicted_price * 0.95
            upper_bound = predicted_price * 1.05

            st.success(f"Predicted Price for {selected_date.strftime('%B %d, %Y')}")
            st.markdown(f"## Rs. {predicted_price:.2f} per kg")
            st.info(
                f"Estimated range: Rs. {lower_bound:.2f} - Rs. {upper_bound:.2f} per kg"
            )
    else:
        st.subheader("Next 30 Days Price Forecast")
        if st.button("Generate Forecast", type="primary"):
            forecast_df = generate_30_day_forecast(df)

            fig, ax = plt.subplots(figsize=(14, 6))
            historical = df.tail(90)
            ax.plot(
                historical["date"],
                historical["price_rupees_per_kg"],
                label="Historical",
                linewidth=2,
                color="#1f77b4",
            )
            ax.plot(
                forecast_df["Date"],
                forecast_df["Predicted Price"],
                label="Forecast",
                linewidth=2,
                color="#ff7f0e",
                linestyle="--",
            )
            ax.axvline(
                x=df["date"].max(),
                color="red",
                linestyle=":",
                linewidth=2,
                label="Forecast Start",
            )
            ax.set_xlabel("Date", fontsize=11, fontweight="bold")
            ax.set_ylabel("Price (Rs./kg)", fontsize=11, fontweight="bold")
            ax.set_title("30-Day Price Forecast", fontsize=13, fontweight="bold")
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)

            st.subheader("Forecast Data")
            forecast_display = forecast_df.copy()
            forecast_display["Predicted Price"] = forecast_display["Predicted Price"].map(
                lambda value: f"Rs. {value:.2f}"
            )
            st.dataframe(forecast_display, use_container_width=True)

            csv = forecast_df.to_csv(index=False)
            st.download_button(
                label="Download Forecast CSV",
                data=csv,
                file_name=f"price_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )

with tab3:
    st.header("Historical Analysis")

    year_filter = st.multiselect(
        "Select Years",
        options=sorted(df["year"].unique()),
        default=sorted(df["year"].unique()),
    )

    filtered_df = df[df["year"].isin(year_filter)]

    st.subheader("Yearly Price Comparison")
    fig, ax = plt.subplots(figsize=(12, 6))
    for year in year_filter:
        year_data = df[df["year"] == year]
        ax.plot(
            year_data["day_of_year"],
            year_data["price_rupees_per_kg"],
            label=str(year),
            linewidth=2,
            alpha=0.7,
        )
    ax.set_xlabel("Day of Year", fontsize=11, fontweight="bold")
    ax.set_ylabel("Price (Rs./kg)", fontsize=11, fontweight="bold")
    ax.set_title("Price Trends by Year", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Yearly Statistics")
    yearly_stats = filtered_df.groupby("year").agg(
        {
            "price_rupees_per_kg": ["mean", "min", "max", "std"],
            "volume_quintals": "mean",
        }
    ).round(2)
    yearly_stats.columns = ["Avg Price", "Min Price", "Max Price", "Std Dev", "Avg Volume"]
    st.dataframe(yearly_stats, use_container_width=True)

    st.subheader("Promotional Impact Analysis")
    col1, col2 = st.columns(2)
    with col1:
        promo_stats = filtered_df.groupby("promo_flag")["price_rupees_per_kg"].agg(["mean", "count"])
        promo_stats.index = ["Non-Promo", "Promo"]
        st.dataframe(promo_stats, use_container_width=True)

    with col2:
        fig, ax = plt.subplots(figsize=(6, 5))
        filtered_df.boxplot(column="price_rupees_per_kg", by="promo_flag", ax=ax)
        ax.set_xlabel("Promo Flag (0=No, 1=Yes)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Price (Rs./kg)", fontsize=10, fontweight="bold")
        ax.set_title("Price: Promo vs Non-Promo", fontsize=11, fontweight="bold")
        plt.suptitle("")
        plt.tight_layout()
        st.pyplot(fig)

with tab4:
    st.header("About This Project")
    st.markdown(
        """
        ### AI-Based Market Trend Analysis

        This application demonstrates agricultural price forecasting using the
        included synthetic onion-price dataset.

        **Included features**

        - Dataset overview and sample records
        - Single-date price estimate
        - 30-day forecast
        - Historical trend analysis
        - Promotional impact analysis
        - Downloadable forecast CSV

        The original README mentions trained Linear Regression, Random Forest,
        and LSTM model files, but those artifacts are not present in this
        checkout. This app therefore uses simple deterministic baselines that
        run with the files currently available.
        """
    )

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888;'>"
    "AI Market Trend Analyzer | Powered by Streamlit"
    "</div>",
    unsafe_allow_html=True,
)
