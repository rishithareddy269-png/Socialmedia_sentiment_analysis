import streamlit as st
import pandas as pd
import joblib
import re
import string
import os
from wordcloud import WordCloud
import matplotlib.pyplot as plt


# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(
    page_title="Social Media Sentiment Analysis",
    page_icon="📊",
    layout="wide"
)


# ==============================
# LOAD MODEL
# ==============================

MODEL_PATH = "model/sentiment_model.pkl"
VECTORIZER_PATH = "model/tfidf_vectorizer.pkl"

try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

except Exception as e:
    st.error(f"Unable to load model files: {e}")
    st.stop()


# ==============================
# TEXT CLEANING
# ==============================

def clean_text(text):

    text = str(text)

    text = text.lower()

    text = re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        text
    )

    text = re.sub(
        r"@\w+",
        "",
        text
    )

    text = re.sub(
        r"#",
        "",
        text
    )

    text = re.sub(
        r"\d+",
        "",
        text
    )

    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ==============================
# SENTIMENT CONVERSION
# ==============================

def sentiment_name(value):

    try:

        value = float(value)

        if value == 1:
            return "Positive"

        if value == 0:
            return "Neutral"

        if value == -1:
            return "Negative"

    except Exception:
        pass

    return str(value)


def sentiment_emoji(sentiment):

    if sentiment == "Positive":
        return "😊"

    if sentiment == "Negative":
        return "😞"

    return "😐"


# ==============================
# HEADER
# ==============================

st.title("📊 Social Media Sentiment Analysis")

st.markdown(
    """
    ### Twitter & Reddit Sentiment Intelligence

    Analyze social media posts using
    **Natural Language Processing and Machine Learning**.
    """
)

st.divider()


# ==============================
# SIDEBAR
# ==============================

st.sidebar.title("⚙️ Project Information")

st.sidebar.markdown(
    """
    **Machine Learning**

    Logistic Regression

    **Feature Extraction**

    TF-IDF

    **Platforms**

    Twitter & Reddit

    **Classes**

    🟢 Positive

    🟡 Neutral

    🔴 Negative
    """
)

st.sidebar.info(
    "This application supports single-text prediction "
    "and CSV batch sentiment analysis."
)


# ==============================
# TABS
# ==============================

tab1, tab2, tab3 = st.tabs(
    [
        "🔍 Single Prediction",
        "📁 CSV Analysis",
        "📈 Dashboard"
    ]
)


# =====================================================
# TAB 1 - SINGLE PREDICTION
# =====================================================

with tab1:

    st.header("🔍 Analyze a Social Media Post")

    text = st.text_area(
        "Enter your Twitter or Reddit post:",
        height=150,
        placeholder="Example: I really love this product!"
    )

    if st.button(
        "🚀 Analyze Sentiment",
        use_container_width=True
    ):

        if not text.strip():

            st.warning(
                "Please enter some text first."
            )

        else:

            cleaned = clean_text(text)

            vector = vectorizer.transform(
                [cleaned]
            )

            prediction = model.predict(
                vector
            )[0]

            sentiment = sentiment_name(
                prediction
            )

            st.divider()

            st.subheader("Prediction Result")

            emoji = sentiment_emoji(
                sentiment
            )

            if sentiment == "Positive":

                st.success(
                    f"{emoji} {sentiment} Sentiment"
                )

            elif sentiment == "Negative":

                st.error(
                    f"{emoji} {sentiment} Sentiment"
                )

            else:

                st.warning(
                    f"{emoji} {sentiment} Sentiment"
                )

            # Probability
            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(
                    vector
                )[0]

                classes = model.classes_

                probability_data = {}

                for cls, probability in zip(
                    classes,
                    probabilities
                ):

                    label = sentiment_name(cls)

                    probability_data[label] = (
                        probability * 100
                    )

                st.subheader(
                    "📊 Prediction Confidence"
                )

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "😊 Positive",
                    f"{probability_data.get('Positive', 0):.2f}%"
                )

                col2.metric(
                    "😐 Neutral",
                    f"{probability_data.get('Neutral', 0):.2f}%"
                )

                col3.metric(
                    "😞 Negative",
                    f"{probability_data.get('Negative', 0):.2f}%"
                )

            st.divider()

            st.write("**Original Text:**")

            st.write(text)

            st.write("**Processed Text:**")

            st.write(cleaned)


# =====================================================
# TAB 2 - CSV ANALYSIS
# =====================================================

with tab2:

    st.header("📁 Bulk CSV Sentiment Analysis")

    st.write(
        "Upload a CSV file containing social media posts."
    )

    uploaded_file = st.file_uploader(
        "Choose CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            uploaded_data = pd.read_csv(
                uploaded_file
            )

            st.success(
                "CSV uploaded successfully!"
            )

            st.subheader("Preview")

            st.dataframe(
                uploaded_data.head(10),
                use_container_width=True
            )

            text_column = st.selectbox(
                "Select the text column:",
                uploaded_data.columns
            )

            if st.button(
                "🚀 Analyze Uploaded CSV",
                use_container_width=True
            ):

                with st.spinner(
                    "Analyzing posts..."
                ):

                    cleaned_text = uploaded_data[
                        text_column
                    ].astype(str).apply(
                        clean_text
                    )

                    vectors = vectorizer.transform(
                        cleaned_text
                    )

                    predictions = model.predict(
                        vectors
                    )

                    uploaded_data[
                        "Predicted Sentiment"
                    ] = [
                        sentiment_name(prediction)
                        for prediction in predictions
                    ]

                st.success(
                    "Sentiment analysis completed!"
                )

                st.subheader(
                    "📋 Prediction Results"
                )

                st.dataframe(
                    uploaded_data,
                    use_container_width=True
                )

                counts = uploaded_data[
                    "Predicted Sentiment"
                ].value_counts()

                st.subheader(
                    "📊 Sentiment Distribution"
                )

                st.bar_chart(counts)

                csv_data = uploaded_data.to_csv(
                    index=False
                )

                st.download_button(
                    label="⬇️ Download Results",
                    data=csv_data,
                    file_name="sentiment_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        except Exception as e:

            st.error(
                f"Error processing CSV: {e}"
            )


# =====================================================
# TAB 3 - DASHBOARD
# =====================================================

with tab3:

    st.header("📈 Sentiment Dashboard")

    combined_path = (
        "data/combined_social_media.csv"
    )

    if os.path.exists(combined_path):

        df = pd.read_csv(
            combined_path
        )

        df["Sentiment"] = df[
            "sentiment"
        ].apply(
            sentiment_name
        )

        total_posts = len(df)

        positive_count = (
            df["Sentiment"] == "Positive"
        ).sum()

        neutral_count = (
            df["Sentiment"] == "Neutral"
        ).sum()

        negative_count = (
            df["Sentiment"] == "Negative"
        ).sum()

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Total Posts",
            f"{total_posts:,}"
        )

        col2.metric(
            "😊 Positive",
            f"{positive_count:,}"
        )

        col3.metric(
            "😐 Neutral",
            f"{neutral_count:,}"
        )

        col4.metric(
            "😞 Negative",
            f"{negative_count:,}"
        )

        st.divider()

        st.subheader(
            "Overall Sentiment Distribution"
        )

        sentiment_counts = (
            df["Sentiment"]
            .value_counts()
        )

        st.bar_chart(
            sentiment_counts
        )

        st.divider()

        st.subheader(
            "Twitter vs Reddit Sentiment"
        )

        platform_sentiment = pd.crosstab(
            df["platform"],
            df["Sentiment"]
        )

        st.dataframe(
            platform_sentiment,
            use_container_width=True
        )

        st.bar_chart(
            platform_sentiment
        )

        st.divider()

        st.subheader(
            "☁️ Social Media Word Cloud"
        )

        all_text = " ".join(
            df["text"].astype(str)
        )

        if all_text.strip():

            wordcloud = WordCloud(
                width=1000,
                height=500,
                background_color="white",
                max_words=150
            ).generate(all_text)

            fig, ax = plt.subplots(
                figsize=(12, 6)
            )

            ax.imshow(
                wordcloud,
                interpolation="bilinear"
            )

            ax.axis("off")

            st.pyplot(fig)

            plt.close(fig)

    else:

        st.warning(
            "combined_social_media.csv was not found."
        )


# ==============================
# FOOTER
# ==============================

st.divider()

st.caption(
    "Social Media Sentiment Analysis | "
    "NLP + TF-IDF + Machine Learning"
)