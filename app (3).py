#!/usr/bin/env python
# coding: utf-8

# JAMES NDUNGU
# MUSIC STREAMING USAGE PLATFORM ANALYZER SYSTEM PROJECT
# DDA-01-0002/2024
# START

# In[1]:


# """ importing necessary libraries"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json 
import os


# In[2]:


# """loading the dataset"""
df=pd.read_csv("C:\\Users\\ADMIN\\Downloads\\Spotify Most Streamed Songs.csv")
df


# In[3]:


print(df[['released_year', 'released_month', 'released_day']].isnull().sum())


# In[4]:


df[['released_year', 'released_month', 'released_day']] = df[['released_year', 'released_month', 'released_day']].fillna(1).astype(int)


# In[5]:


df['release_date'] = pd.to_datetime(
    df['released_year'].astype(str) + '-' + 
    df['released_month'].astype(str) + '-' + 
    df['released_day'].astype(str),
    errors='coerce'
)


# In[6]:


print(df.columns)


# In[7]:


##dropping unwanted columns
df=df.drop(columns=['cover_url'])


# In[8]:


#checking for null values or missing values
df.isnull().sum()


# In[9]:


#filling the missing values 
df=df.fillna(method= 'bfill')


# In[10]:


df.isnull().sum().sum()


# In[11]:


df


# checking if ther is any correlation btwn variables

# In[12]:


df['artist(s)_name'].value_counts()


# In[13]:


df['in_spotify_charts'].value_counts()


# In[14]:


"""importing necessary libraries"""
import scipy
from scipy.stats import pearsonr


# In[15]:


pearsonr(df['danceability_%'],df['energy_%'])


# In[16]:


import scipy
from scipy.stats import spearmanr


# In[17]:


spearmanr(df['danceability_%'],df['energy_%'])


# In[18]:


df.groupby('mode')['track_name'].describe().head()


# In[19]:


#tracks and their modes 
top_streams=df.groupby('mode')['track_name'].describe()
print(top_streams)


# In[20]:


df['danceability_%']


# checking if dancebility affects speechiness

# In[21]:


plt.scatter(df['danceability_%'],df['speechiness_%'])
"""Add a regression line to visualize the trend"""
sns.regplot(x='danceability_%', y='speechiness_%', data=df, scatter=False, color='red')

# Add titles and labels
plt.title('Scatter Plot of danceability vs speechiness_%')
plt.xlabel('danceability')
plt.ylabel('speechiness_%')
plt.grid(True)
plt.show()


# In[30]:


import pandas as pd
import plotly.subplots as sp
import plotly.graph_objects as go
import plotly.express as px

# Ensure 'streams' column is numeric
df["streams"] = pd.to_numeric(df["streams"], errors="coerce")

# Drop NaN values that may arise from conversion issues
df = df.dropna(subset=["streams"])

# Create a subplot grid
fig = sp.make_subplots(
    rows=4, cols=2,
    subplot_titles=(
        "Streams Over Time", "Top 10 Songs by Streams",
        "Distribution of BPM", "BPM vs Streams",
        "Energy vs Streams", "Distribution of Streams",
        "Distribution Across Platforms", "Correlation Heatmap"
    ),
    specs=[
        [{"type": "scatter"}, {"type": "bar"}],
        [{"type": "box"}, {"type": "scatter"}],
        [{"type": "scatter"}, {"type": "histogram"}],
        [{"type": "pie"}, {"type": "heatmap"}]
    ]
)

# 1. Line Chart: Streams Over Time
if 'release_date' in df.columns and 'streams' in df.columns:
    fig_time = px.line(
        df,
        x="release_date",
        y="streams",
        labels={"release_date": "Release Date", "streams": "Streams"}
    )
    for trace in fig_time.data:
        fig.add_trace(trace, row=1, col=1)

# 2. Bar Chart: Top 10 Songs by Streams
if "track_name" in df.columns and "streams" in df.columns:
    top_songs = df.nlargest(10, "streams")
    fig_bar = px.bar(
        top_songs,
        x="track_name",
        y="streams",
        labels={"track_name": "Track Name", "streams": "Streams"},
    )
    for trace in fig_bar.data:
        fig.add_trace(trace, row=1, col=2)

# 3. Box Plots: Audio Features Distribution
features = [
    'danceability_%', 'valence_%', 'energy_%',
    'acousticness_%', 'instrumentalness_%', 'liveness_%', 'speechiness_%'
]
for feature in features:
    if feature in df.columns:
        fig_box = px.box(df, y=feature, labels={feature: feature})
        for trace in fig_box.data:
            fig.add_trace(trace, row=2, col=1)

# 4. Scatter Plots
if 'bpm' in df.columns and 'streams' in df.columns:
    fig_scatter_bpm = px.scatter(
        df,
        x="bpm",
        y="streams",
        labels={"bpm": "Beats Per Minute", "streams": "Streams"},
        trendline="ols"
    )
    for trace in fig_scatter_bpm.data:
        fig.add_trace(trace, row=2, col=2)

if 'energy_%' in df.columns and 'streams' in df.columns:
    fig_scatter_energy = px.scatter(
        df,
        x="energy_%",
        y="streams",
        labels={"energy_%": "Energy", "streams": "Streams"},
        trendline="ols"
    )
    for trace in fig_scatter_energy.data:
        fig.add_trace(trace, row=3, col=1)

# 5. Histograms
if 'bpm' in df.columns:
    fig_hist_bpm = px.histogram(
        df,
        x="bpm",
        nbins=20,
        labels={"bpm": "Beats Per Minute"}
    )
    for trace in fig_hist_bpm.data:
        fig.add_trace(trace, row=3, col=2)

if 'streams' in df.columns:
    fig_hist_streams = px.histogram(
        df,
        x="streams",
        nbins=20,
        labels={"streams": "Streams"}
    )
    for trace in fig_hist_streams.data:
        fig.add_trace(trace, row=3, col=2)  # Now in the correct subplot type

# 6. Pie Chart: Distribution Across Platforms
platforms = {
    "Spotify Playlists": "in_spotify_playlists",
    "Spotify Charts": "in_spotify_charts",
    "Apple Playlists": "in_apple_playlists",
    "Apple Charts": "in_apple_charts",
    "Deezer Playlists": "in_deezer_playlists",
    "Deezer Charts": "in_deezer_charts",
    "Shazam Charts": "in_shazam_charts"
}
platform_values = {name: df[col].sum() for name, col in platforms.items() if col in df.columns}
if platform_values:
    platform_df = pd.DataFrame({
        "Platform": list(platform_values.keys()),
        "Count": list(platform_values.values())
    })
    fig_pie = px.pie(
        platform_df,
        names="Platform",
        values="Count"
    )
    for trace in fig_pie.data:
        fig.add_trace(trace, row=4, col=1)  # Pie chart now correctly placed

# 7. Correlation Heatmap
numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
if len(numeric_cols) > 1:
    corr = df[numeric_cols].corr()
    heatmap = go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale="balance",
        zmin=-1,
        zmax=1
    )
    fig.add_trace(heatmap, row=4, col=2)

# Update layout
fig.update_layout(
    height=1200,
    width=1200,
    title_text="Spotify Most Streamed Songs Dashboard",
    showlegend=False
)

# Show the dashboard
fig.show()


# In[31]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import streamlit as st

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("C:\\Users\\ADMIN\\Downloads\\Spotify Most Streamed Songs.csv")

df = load_data()

# Ensure "streams" column is numeric
if "streams" in df.columns:
    df["streams"] = pd.to_numeric(df["streams"], errors="coerce")
    df = df.dropna(subset=["streams"])  # Remove NaN values
    df["streams"] = df["streams"].astype(int)

# Feature Selection
features = [
    'danceability_%', 'valence_%', 'energy_%', 'acousticness_%',
    'instrumentalness_%', 'liveness_%', 'speechiness_%', 'bpm',
    'in_spotify_playlists', 'in_spotify_charts', 'in_apple_playlists',
    'in_apple_charts', 'in_deezer_playlists', 'in_deezer_charts',
    'in_shazam_charts', 'artist(s)_name'
]
target = 'streams'

# Filter the dataset
df = df[features + [target]]

# Clean numerical columns: Remove commas and convert to numeric
numerical_features = [col for col in features if col != 'artist(s)_name']
for col in numerical_features:
    if df[col].dtype == object:  # Check if the column contains strings
        df[col] = df[col].str.replace(",", "", regex=True)  # Remove commas
        df[col] = pd.to_numeric(df[col], errors="coerce")  # Convert to numeric

# Handle missing values
df = df.dropna()

# Encode categorical variables
categorical_features = ['artist(s)_name']

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Split the data
X = df.drop(columns=[target])
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display evaluation metrics
st.subheader("Model Evaluation Metrics")
st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
st.write(f"**R-squared (R2):** {r2:.2f}")

# Add a section for predictions
st.subheader("Predict Streams for New Data")

# Input fields for new data
st.write("Enter the features for prediction:")
danceability = st.number_input("Danceability (%)", min_value=0, max_value=100, value=50)
valence = st.number_input("Valence (%)", min_value=0, max_value=100, value=50)
energy = st.number_input("Energy (%)", min_value=0, max_value=100, value=50)
acousticness = st.number_input("Acousticness (%)", min_value=0, max_value=100, value=50)
instrumentalness = st.number_input("Instrumentalness (%)", min_value=0, max_value=100, value=50)
liveness = st.number_input("Liveness (%)", min_value=0, max_value=100, value=50)
speechiness = st.number_input("Speechiness (%)", min_value=0, max_value=100, value=50)
bpm = st.number_input("Beats Per Minute (BPM)", min_value=0, max_value=200, value=120)
spotify_playlists = st.number_input("Spotify Playlists", min_value=0, value=10)
spotify_charts = st.number_input("Spotify Charts", min_value=0, value=10)
apple_playlists = st.number_input("Apple Playlists", min_value=0, value=10)
apple_charts = st.number_input("Apple Charts", min_value=0, value=10)
deezer_playlists = st.number_input("Deezer Playlists", min_value=0, value=10)
deezer_charts = st.number_input("Deezer Charts", min_value=0, value=10)
shazam_charts = st.number_input("Shazam Charts", min_value=0, value=10)
artist = st.selectbox("Artist", df["artist(s)_name"].unique())

# Create a DataFrame for the new input
new_data = pd.DataFrame({
    'danceability_%': [danceability],
    'valence_%': [valence],
    'energy_%': [energy],
    'acousticness_%': [acousticness],
    'instrumentalness_%': [instrumentalness],
    'liveness_%': [liveness],
    'speechiness_%': [speechiness],
    'bpm': [bpm],
    'in_spotify_playlists': [spotify_playlists],
    'in_spotify_charts': [spotify_charts],
    'in_apple_playlists': [apple_playlists],
    'in_apple_charts': [apple_charts],
    'in_deezer_playlists': [deezer_playlists],
    'in_deezer_charts': [deezer_charts],
    'in_shazam_charts': [shazam_charts],
    'artist(s)_name': [artist]
})

# Predict streams for the new data
if st.button("Predict Streams"):
    # Transform input using the same pipeline
    prediction = model.predict(new_data)
    st.write(f"### 🎵 Predicted Streams: {int(prediction[0])}")


# In[ ]:




