# """ importing necessary libraries"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json 
import os

# """loading the dataset"""
df=pd.read_csv("C:\\Users\\ADMIN\\Downloads\\Spotify Most Streamed Songs.csv")
df

print(df[['released_year', 'released_month', 'released_day']].isnull().sum())

df[['released_year', 'released_month', 'released_day']] = df[['released_year', 'released_month', 'released_day']].fillna(1).astype(int)

df['release_date'] = pd.to_datetime(
    df['released_year'].astype(str) + '-' + 
    df['released_month'].astype(str) + '-' + 
    df['released_day'].astype(str),
    errors='coerce'
)

print(df.columns)

##dropping unwanted columns
df=df.drop(columns=['cover_url'])


#checking for null values or missing values
df.isnull().sum()

#filling the missing values 
df=df.fillna(method= 'bfill')

df.isnull().sum().sum()

df

df['artist(s)_name'].value_counts()

df['in_spotify_charts'].value_counts()

"""importing necessary libraries"""
import scipy
from scipy.stats import pearsonr

pearsonr(df['danceability_%'],df['energy_%'])

import scipy
from scipy.stats import spearmanr

spearmanr(df['danceability_%'],df['energy_%'])

df.groupby('mode')['track_name'].describe().head()

#tracks and their modes 
top_streams=df.groupby('mode')['track_name'].describe()
print(top_streams)


df['danceability_%']

plt.scatter(df['danceability_%'],df['speechiness_%'])
"""Add a regression line to visualize the trend"""
sns.regplot(x='danceability_%', y='speechiness_%', data=df, scatter=False, color='red')

# Add titles and labels
plt.title('Scatter Plot of danceability vs speechiness_%')
plt.xlabel('danceability')
plt.ylabel('speechiness_%')
plt.grid(True)
plt.show()

import plotly.subplots as sp
import plotly.graph_objects as go

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
if 'track_name' in df.columns and 'streams' in df.columns:
    top_songs = df.nlargest(10, "streams")
    fig_bar = px.bar(
        top_songs,
        x="track_name",
        y="streams",
        labels={"track_name": "Track Name", "streams": "Streams"}
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
        fig_box = px.box(
            df,
            y=feature,
            labels={feature: feature}
        )
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
        fig.add_trace(trace, row=4, col=1)

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
        fig.add_trace(trace, row=4, col=2)

# 7. Correlation Heatmap
numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
if len(numeric_cols) > 0:
    corr = df[numeric_cols].corr()
    heatmap = go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale="coolwarm",
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

! pip install streamlit

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Load dataset (Ensure the file is in the same directory or provide a correct path)
@st.cache_data
def load_data():
    return pd.read_csv("C:\\Users\\ADMIN\\Downloads\\Spotify Most Streamed Songs.csv")  # Update with correct path

df = load_data()

# Ensure "streams" column is numeric
if "streams" in df.columns:
    df["streams"] = pd.to_numeric(df["streams"], errors="coerce")
    df = df.dropna(subset=["streams"])  # Remove NaN values
    df["streams"] = df["streams"].astype(int)

# Streamlit App Layout
st.title("Music Streaming Platform Usage Analyzer")
st.sidebar.header("Filters")

# User Filters
selected_artist = st.sidebar.selectbox("Select Artist", ["All"] + list(df["artist(s)_name"].unique()))

# Apply Filters
filtered_df = df.copy()
if selected_artist != "All":
    filtered_df = filtered_df[filtered_df["artist(s)_name"] == selected_artist]

# Display Data
st.subheader("Top 10 Songs by Streams")
st.dataframe(filtered_df.nlargest(10, "streams"))

# Visualization 1: Streams Distribution by Artist
st.subheader("Streams Distribution by Artist")
fig1 = go.Figure()
fig1.add_trace(go.Bar(x=filtered_df["track_name"], y=filtered_df["streams"], name="Streams"))
st.plotly_chart(fig1, use_container_width=True)

# Visualization 2: Streams Over Time (Line Chart)
if "release_date" in filtered_df.columns:
    st.subheader("Streams Over Time")
    fig2 = px.line(filtered_df, x="release_date", y="streams", title="Streams Over Time")
    st.plotly_chart(fig2, use_container_width=True)

# Visualization 3: Distribution of BPM (Histogram)
if "bpm" in filtered_df.columns:
    st.subheader("Distribution of BPM")
    fig3 = px.histogram(filtered_df, x="bpm", nbins=20, title="Distribution of BPM")
    st.plotly_chart(fig3, use_container_width=True)

# Visualization 4: BPM vs Streams (Scatter Plot)
if "bpm" in filtered_df.columns and "streams" in filtered_df.columns:
    st.subheader("BPM vs Streams")
    fig4 = px.scatter(filtered_df, x="bpm", y="streams", trendline="ols", title="BPM vs Streams")
    st.plotly_chart(fig4, use_container_width=True)

# Visualization 5: Distribution Across Platforms (Pie Chart)
platforms = {
    "Spotify Playlists": "in_spotify_playlists",
    "Spotify Charts": "in_spotify_charts",
    "Apple Playlists": "in_apple_playlists",
    "Apple Charts": "in_apple_charts",
    "Deezer Playlists": "in_deezer_playlists",
    "Deezer Charts": "in_deezer_charts",
    "Shazam Charts": "in_shazam_charts"
}

# Ensure only numeric columns are summed
platform_values = {}
for name, col in platforms.items():
    if col in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[col]):
        platform_values[name] = filtered_df[col].sum()

if platform_values:
    st.subheader("Distribution Across Platforms")
    platform_df = pd.DataFrame({
        "Platform": list(platform_values.keys()),
        "Count": list(platform_values.values())
    })
    fig5 = px.pie(platform_df, names="Platform", values="Count", title="Distribution Across Platforms")
    st.plotly_chart(fig5, use_container_width=True)

# Visualization 6: Correlation Heatmap
numeric_cols = filtered_df.select_dtypes(include=["float64", "int64"]).columns
if len(numeric_cols) > 0:
    st.subheader("Correlation Heatmap")
    corr = filtered_df[numeric_cols].corr()
    fig6 = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale="Viridis",  # Use a valid Plotly colorscale
        zmin=-1,
        zmax=1
    ))
    st.plotly_chart(fig6, use_container_width=True)

