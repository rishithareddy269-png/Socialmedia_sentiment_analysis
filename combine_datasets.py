import pandas as pd

# Load Twitter dataset
twitter = pd.read_csv("data/Twitter_Data.csv")

# Load Reddit dataset
reddit = pd.read_csv("data/Reddit_Data.csv")


# Select required columns from Twitter
twitter = twitter[["clean_text", "category"]]

# Rename Twitter columns
twitter = twitter.rename(columns={
    "clean_text": "text",
    "category": "sentiment"
})

# Add platform
twitter["platform"] = "Twitter"


# Select required columns from Reddit
reddit = reddit[["clean_comment", "category"]]

# Rename Reddit columns
reddit = reddit.rename(columns={
    "clean_comment": "text",
    "category": "sentiment"
})

# Add platform
reddit["platform"] = "Reddit"


# Combine both datasets
combined = pd.concat(
    [twitter, reddit],
    ignore_index=True
)


# Remove missing text
combined = combined.dropna(subset=["text"])


# Remove empty text
combined = combined[
    combined["text"].astype(str).str.strip() != ""
]


# Shuffle the dataset
combined = combined.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# Arrange columns
combined = combined[
    ["platform", "text", "sentiment"]
]


# Save combined dataset
combined.to_csv(
    "data/combined_social_media.csv",
    index=False
)


# Display information
print("Datasets combined successfully!")

print("\nCombined dataset shape:")
print(combined.shape)

print("\nFirst 5 rows:")
print(combined.head())

print("\nSentiment distribution:")
print(combined["sentiment"].value_counts())

print("\nPlatform distribution:")
print(combined["platform"].value_counts())