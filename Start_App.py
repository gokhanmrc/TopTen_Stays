import streamlit as st
import pandas as pd
import numpy as np
import math
import scipy.stats as sct
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.logo(
    "images/icon1.png",
    size="large",
    icon_image="images/icon_s_1.png"
)
st.set_page_config(layout="wide", page_title="TopTen Stays", page_icon="🏨")
st.title(":orange[TopTen Stays]🥂🌇")
home_tab, data_tab, recomandations_tab = st.tabs(["Home", "Data", "Recomandations"])
Empty_1, image_left, image_right, Empty_2 = home_tab.columns([4,6,5,4], gap="small")

image_left.image("images/dikdötgen.png",width=360)
image_right.image("images/karcik.png",width=450)
image_right.subheader(":orange[  Discover the top 10 hotels in the 6 best capitals of Europe!]")
image_right.markdown(""" 

Welcome to **_TopTen Stays_**, where you’ll find reviews and ratings for nearly 1,500 hotels across the top 6 capitals of Europe. 

Whether you're looking for accommodations rated by travelers from your own country or want to explore the best options in popular destinations, we’ve got you covered.

For trips to the **_United Kingdom, Spain, France, the Netherlands, Austria, or Italy_**, we’ll recommend the top 10 hotels in each capital to make your stay unforgettable.

""")
image_right.image("images/toplar.png",width=400)


# Title and short introduction
st.sidebar.title(":orange[Discover Your Ideal Stay!] 🌍")
st.sidebar.markdown("Explore top-rated hotels across Europe’s best destinations.")

# Informative visual
st.sidebar.image("images/otel.png", caption="Find the best hotels for an unforgettable stay.", use_container_width=True)

# Popular Destinations
st.sidebar.markdown("### Top Destinations")
st.sidebar.markdown("""
- **London** - Explore the heart of the UK.
- **Paris** - The city of lights awaits.
- **Madrid** - Sun, culture, and lively streets.
- **Amsterdam** - Charming canals and rich history.
- **Vienna** - Elegant streets and imperial palaces.
- **Rome** - Walk through ancient history.
""")

# Information about the application
st.sidebar.markdown("### About")
st.sidebar.info(
    "This hotel recommendation system helps you find the top 10 hotels across Europe’s best capitals. "
    "Use the filters to tailor recommendations to your needs and plan a memorable stay."
)

####_Data_Tab_####


data_tab.subheader("This dataset contains 515,000 customer reviews and scoring of 1493 luxury hotels across Europe. After processing the available data, here is the list of the top 10 hotels selected from each of the 6 countries.")

@st.cache_data
def get_data(nation_x = ["United Kingdom"]):
    df1 = pd.read_csv("part1.csv")
    df2 = pd.read_csv("part2.csv")
    df3 = pd.read_csv("part3.csv")

    df = pd.concat([df1, df2, df3], ignore_index=True)

    df = df.drop(columns=["Review_Date", "Review_Total_Negative_Word_Counts", "Review_Total_Positive_Word_Counts",
                          "Total_Number_of_Reviews_Reviewer_Has_Given","Tags" , "lat", "lng"], axis=1)

    scaler = MinMaxScaler(feature_range=(0, 5))
    df['Reviewer_Score'] = scaler.fit_transform(df[['Reviewer_Score']])
    df['Average_Score'] = scaler.fit_transform(df[['Average_Score']])

    df["days_since_review"] = (df["days_since_review"]
                               .str.replace(" days", "")
                               .str.replace(" day", "")
                               .astype(int))

    df["summary_min"] = df.groupby("Hotel_Name")["days_since_review"].transform("min")
    df["summary_q1"] = df.groupby("Hotel_Name")["days_since_review"].transform(lambda x: x.quantile(0.25))
    df["summary_q2"] = df.groupby("Hotel_Name")["days_since_review"].transform("median")
    df["summary_q3"] = df.groupby("Hotel_Name")["days_since_review"].transform(lambda x: x.quantile(0.75))
    df["summary_max"] = df.groupby("Hotel_Name")["days_since_review"].transform("max")

    def weighted_average_by_time(df_group):
        q1, q2, q3 = df_group["summary_q1"].iloc[0], df_group["summary_q2"].iloc[0], df_group["summary_q3"].iloc[0]

        weighted_avg = (
                df_group.loc[df_group["days_since_review"] <= q1, "Reviewer_Score"].mean() * 0.28 +
                df_group.loc[(df_group["days_since_review"] > q1) & (
                        df_group["days_since_review"] <= q2), "Reviewer_Score"].mean() * 0.26 +
                df_group.loc[(df_group["days_since_review"] > q2) & (
                        df_group["days_since_review"] <= q3), "Reviewer_Score"].mean() * 0.24 +
                df_group.loc[df_group["days_since_review"] > q3, "Reviewer_Score"].mean() * 0.22
        )
        return weighted_avg

    hotel_weighted_averages = df.groupby("Hotel_Name").apply(weighted_average_by_time).reset_index(
        name="Recent_Feedback_Score")
    df = df.merge(hotel_weighted_averages, on="Hotel_Name", how="left")

    df["Reviewer_Nationality"] = df["Reviewer_Nationality"].str.strip().str.strip("'\"")

    def weighted_national(df_group, nation):
        selected_nation_score = df_group.loc[df_group["Reviewer_Nationality"].isin(nation), "Reviewer_Score"].mean()
        other_nations_score = df_group.loc[~df_group["Reviewer_Nationality"].isin(nation), "Reviewer_Score"].mean()
        weighted_avg = (selected_nation_score * 0.60) + (other_nations_score * 0.40)

        return weighted_avg

    hotel_weighted_averages_by_nation = df.groupby("Hotel_Name").apply(weighted_national,
                                                                       nation_x).reset_index(
        name="Nation_Based_Weighted_Score")

    df = df.merge(hotel_weighted_averages_by_nation, on="Hotel_Name", how="left")

    bins = [0, 1, 2, 3, 4, 6]
    labels = [1, 2, 3, 4, 5]

    df['Final_star'] = pd.cut(df['Reviewer_Score'], bins=bins, labels=labels, right=False)
    df = pd.get_dummies(df, columns=['Final_star'], prefix='Star', dtype="int64")

    df['Negative_Review'] = df['Negative_Review'].replace([
        'Non', 'No', 'None', 'NA', 'Nothing', 'Nothin', 'Nothinb', 'Nothibg', 'Nothings', 'NOTHNG',
        'Nothink', 'Nothig', 'Nothingseccess', 'Nothimg', 'Nothinh', 'Nothg', 'Nothing7', 'Nothinge',
        'Notheng', 'Nothhing', 'Nothong', 'Nothjng', 'Nothingefs', 'Nothjnng', 'Nothingg', 'Nothung',
        'Nothinf', 'Nothingthe', 'NothingA', 'Nothiing', 'NOTHINGGGGGG', 'nothging', 'Nothting',
        'nothingone', 'Nothubg', 'Nothjg', 'NothingGra', 'Noththing', 'Nothening', 'NothingGreat',
        'Nothingreally', 'Nothlng'
    ], 'No Negative')

    df['Positive_Review'] = df['Positive_Review'].replace([
        'nothing', 'Nothink', 'Nothin', 'Nothings', 'Not', 'Notjing', 'Nothng', 'NA', 'Na', 'nathing'
    ], 'No Positive')

    df['Negative_Flag'] = df['Negative_Review'].apply(lambda x: 0 if x == 'No Negative' else 1)
    df['Positive_Flag'] = df['Positive_Review'].apply(lambda x: 0 if x == 'No Positive' else 1)
    df["Total_count"] = df["Negative_Flag"] + df["Positive_Flag"]
    df = df[df['Total_count'] != 0]

    df_grouped = (df.groupby("Hotel_Name").agg(
        Negative_Review=("Negative_Flag", "sum"),
        Positive_Review=("Positive_Flag", "sum"))
                  .sort_values(by="Positive_Review", ascending=False)
                  ).reset_index()

    df_grouped['Positive_Ratio'] = df_grouped['Positive_Review'] / (
                df_grouped['Negative_Review'] + df_grouped['Positive_Review'])
    df = df.merge(df_grouped, on="Hotel_Name", how="left")

    Final_df = (
        df.groupby('Hotel_Name').agg(
            Hotel_Address=('Hotel_Address', 'first'),
            Recent_Feedback_Score=('Recent_Feedback_Score', 'first'),
            Nation_Based_Weighted_Score=('Nation_Based_Weighted_Score', 'first'),
            Total_Number_of_Reviews=('Total_Number_of_Reviews', 'first'),
            Star_Rating=('Reviewer_Score', 'mean'),
            Star_1=('Star_1', 'sum'),
            Star_2=('Star_2', 'sum'),
            Star_3=('Star_3', 'sum'),
            Star_4=('Star_4', 'sum'),
            Star_5=('Star_5', 'sum'),
            Negative_Review=("Negative_Flag", "sum"),
            Positive_Review=("Positive_Flag", "sum"),
            Positive_Ratio=("Positive_Ratio", 'first')
        )
        .sort_values(by="Recent_Feedback_Score", ascending=False)
        .reset_index()
    )

    Final_df["Comment_Count_Score"] = Final_df["Total_Number_of_Reviews"] - Final_df["Negative_Review"] + Final_df[
        "Positive_Review"]
    Final_df["Log_Comment_Count_Score"] = np.log1p(Final_df["Comment_Count_Score"])
    scaler = MinMaxScaler(feature_range=(1, 5))
    Final_df["Comment_Count_Score"] = scaler.fit_transform(Final_df[["Log_Comment_Count_Score"]])

    def weighted_sorting_score(dataframe, w1=30, w2=70, ):
        return (dataframe["Comment_Count_Score"] * w1 / 100 +
                dataframe["Nation_Based_Weighted_Score"] * w2 / 100)

    Final_df["weighted_sorting_score"] = weighted_sorting_score(Final_df)

    def bayesian_average_rating(n, confidence=0.95):
        if sum(n) == 0:
            return 0
        K = len(n)
        z = sct.norm.ppf(1 - (1 - confidence) / 2)
        N = sum(n)
        first_part = 0.0
        second_part = 0.0
        for k, n_k in enumerate(n):
            first_part += (k + 1) * (n[k] + 1) / (N + K)
            second_part += (k + 1) * (k + 1) * (n[k] + 1) / (N + K)
        score = first_part - z * math.sqrt((second_part - first_part * first_part) / (N + K + 1))
        return score

    Final_df["bar_score"] = Final_df.apply(lambda x: bayesian_average_rating(x[["Star_1",
                                                                                "Star_2",
                                                                                "Star_3",
                                                                                "Star_4",
                                                                                "Star_5"]]), axis=1)

    def hybrid_sorting_score(dataframe, bar_w=60, wss_w=40):
        bar_score = dataframe.apply(lambda x: bayesian_average_rating(x[["Star_1",
                                                                         "Star_2",
                                                                         "Star_3",
                                                                         "Star_4",
                                                                         "Star_5"]]), axis=1)
        wss_score = weighted_sorting_score(dataframe)

        return (bar_score * bar_w / 100) + (wss_score * wss_w / 100)

    Final_df["hybrid_sorting_score"] = hybrid_sorting_score(Final_df)

    Final_df['Country_name'] = Final_df['Hotel_Address'].apply(
        lambda x: 'United Kingdom' if 'United Kingdom' in x else x.split()[-1])


    return Final_df

df= get_data()
df_first10 = df.head(10).sort_values('hybrid_sorting_score', ascending=False).reset_index(drop=True)
data_tab.dataframe(df_first10)

data_tab.title("Hotel Data Visualizations")
graph1, grap2 = data_tab.columns([2,2], gap="small")
#Graph1
graph1.subheader("Number of Hotels by Country")
plt.figure(figsize=(5, 4))
ax = sns.countplot(data=df, x='Country_name', palette='viridis')
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='baseline', fontsize=8, color='black', xytext=(0, 5), textcoords='offset points')
plt.ylim(0, 500)
plt.title("Number of Hotels by Country", fontsize=12)
plt.xlabel("")
plt.ylabel("Hotel Count", fontsize=10)
plt.xticks(rotation=45, fontsize=8)
plt.yticks(fontsize=8)
graph1.pyplot(plt.gcf())

#Graph2
grap2.subheader("Hotels by Blended Sorting Score")
plt.figure(figsize=(6, 4))  # Daha küçük bir boyut seçildi
df_first10 = df_first10.sort_values(by="hybrid_sorting_score", ascending=False)
bars = plt.bar(df_first10['Hotel_Name'], df_first10['hybrid_sorting_score'], color="skyblue", width=0.6)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.1, round(yval, 2), ha='center', va='bottom', fontsize=8)
plt.ylim(0, max(df_first10['hybrid_sorting_score']) + 2)
plt.ylabel("Blended Sorting Score", fontsize=10)
plt.title("Hotels by Blended Sorting Score", fontsize=12)
plt.xticks(rotation=90, fontsize=8)
plt.yticks(fontsize=8)
grap2.pyplot(plt.gcf())

#graph 3
fig1 = px.scatter(df,
                  x= "Total_Number_of_Reviews",
                  y= "Positive_Ratio",
                  color= "Country_name",
                  title= "Hotel Ratings and Their Positive Feedback Ratios",
                  hover_data=["Hotel_Name","hybrid_sorting_score"])
data_tab.plotly_chart(fig1, use_container_width=True)


####recomandations_tab####
@st.cache_data
def get_data_1():
    df1 = pd.read_csv("part1.csv")
    df2 = pd.read_csv("part2.csv")
    df3 = pd.read_csv("part3.csv")

    df = pd.concat([df1, df2, df3], ignore_index=True)

    df = df.drop(columns=["Review_Date", "Review_Total_Negative_Word_Counts", "Review_Total_Positive_Word_Counts",
                          "Total_Number_of_Reviews_Reviewer_Has_Given","Tags" , "lat", "lng"], axis=1)

    scaler = MinMaxScaler(feature_range=(0, 5))
    df['Reviewer_Score'] = scaler.fit_transform(df[['Reviewer_Score']])
    df['Average_Score'] = scaler.fit_transform(df[['Average_Score']])

    df["days_since_review"] = (df["days_since_review"]
                               .str.replace(" days", "")
                               .str.replace(" day", "")
                               .astype(int))

    df["summary_min"] = df.groupby("Hotel_Name")["days_since_review"].transform("min")
    df["summary_q1"] = df.groupby("Hotel_Name")["days_since_review"].transform(lambda x: x.quantile(0.25))
    df["summary_q2"] = df.groupby("Hotel_Name")["days_since_review"].transform("median")
    df["summary_q3"] = df.groupby("Hotel_Name")["days_since_review"].transform(lambda x: x.quantile(0.75))
    df["summary_max"] = df.groupby("Hotel_Name")["days_since_review"].transform("max")

    def weighted_average_by_time(df_group):
        q1, q2, q3 = df_group["summary_q1"].iloc[0], df_group["summary_q2"].iloc[0], df_group["summary_q3"].iloc[0]

        weighted_avg = (
                df_group.loc[df_group["days_since_review"] <= q1, "Reviewer_Score"].mean() * 0.28 +
                df_group.loc[(df_group["days_since_review"] > q1) & (
                        df_group["days_since_review"] <= q2), "Reviewer_Score"].mean() * 0.26 +
                df_group.loc[(df_group["days_since_review"] > q2) & (
                        df_group["days_since_review"] <= q3), "Reviewer_Score"].mean() * 0.24 +
                df_group.loc[df_group["days_since_review"] > q3, "Reviewer_Score"].mean() * 0.22
        )
        return weighted_avg

    hotel_weighted_averages = df.groupby("Hotel_Name").apply(weighted_average_by_time).reset_index(
        name="Recent_Feedback_Score")
    df = df.merge(hotel_weighted_averages, on="Hotel_Name", how="left")


    bins = [0, 1, 2, 3, 4, 6]
    labels = [1, 2, 3, 4, 5]

    df['Final_star'] = pd.cut(df['Reviewer_Score'], bins=bins, labels=labels, right=False)
    df = pd.get_dummies(df, columns=['Final_star'], prefix='Star', dtype="int64")

    df['Negative_Review'] = df['Negative_Review'].replace([
        'Non', 'No', 'None', 'NA', 'Nothing', 'Nothin', 'Nothinb', 'Nothibg', 'Nothings', 'NOTHNG',
        'Nothink', 'Nothig', 'Nothingseccess', 'Nothimg', 'Nothinh', 'Nothg', 'Nothing7', 'Nothinge',
        'Notheng', 'Nothhing', 'Nothong', 'Nothjng', 'Nothingefs', 'Nothjnng', 'Nothingg', 'Nothung',
        'Nothinf', 'Nothingthe', 'NothingA', 'Nothiing', 'NOTHINGGGGGG', 'nothging', 'Nothting',
        'nothingone', 'Nothubg', 'Nothjg', 'NothingGra', 'Noththing', 'Nothening', 'NothingGreat',
        'Nothingreally', 'Nothlng'
    ], 'No Negative')

    df['Positive_Review'] = df['Positive_Review'].replace([
        'nothing', 'Nothink', 'Nothin', 'Nothings', 'Not', 'Notjing', 'Nothng', 'NA', 'Na', 'nathing'
    ], 'No Positive')

    df['Negative_Flag'] = df['Negative_Review'].apply(lambda x: 0 if x == 'No Negative' else 1)
    df['Positive_Flag'] = df['Positive_Review'].apply(lambda x: 0 if x == 'No Positive' else 1)
    df["Total_count"] = df["Negative_Flag"] + df["Positive_Flag"]
    df = df[df['Total_count'] != 0]

    df_grouped = (df.groupby("Hotel_Name").agg(
        Negative_Review=("Negative_Flag", "sum"),
        Positive_Review=("Positive_Flag", "sum"))
                  .sort_values(by="Positive_Review", ascending=False)
                  ).reset_index()

    df_grouped['Positive_Ratio'] = df_grouped['Positive_Review'] / (
                df_grouped['Negative_Review'] + df_grouped['Positive_Review'])
    df = df.merge(df_grouped, on="Hotel_Name", how="left")

    Final_df = (
        df.groupby('Hotel_Name').agg(
            Hotel_Address=('Hotel_Address', 'first'),
            Recent_Feedback_Score=('Recent_Feedback_Score', 'first'),
            Total_Number_of_Reviews=('Total_Number_of_Reviews', 'first'),
            Star_Rating=('Reviewer_Score', 'mean'),
            Star_1=('Star_1', 'sum'),
            Star_2=('Star_2', 'sum'),
            Star_3=('Star_3', 'sum'),
            Star_4=('Star_4', 'sum'),
            Star_5=('Star_5', 'sum'),
            Negative_Review=("Negative_Flag", "sum"),
            Positive_Review=("Positive_Flag", "sum"),
            Positive_Ratio=("Positive_Ratio", 'first')
        )
        .sort_values(by="Recent_Feedback_Score", ascending=False)
        .reset_index()
    )

    Final_df["Comment_Count_Score"] = Final_df["Total_Number_of_Reviews"] - Final_df["Negative_Review"] + Final_df[
        "Positive_Review"]
    Final_df["Log_Comment_Count_Score"] = np.log1p(Final_df["Comment_Count_Score"])
    scaler = MinMaxScaler(feature_range=(1, 5))
    Final_df["Comment_Count_Score"] = scaler.fit_transform(Final_df[["Log_Comment_Count_Score"]])

    def weighted_sorting_score(dataframe, w1=30, w2=70, ):
        return (dataframe["Comment_Count_Score"] * w1 / 100 +
                dataframe["Recent_Feedback_Score"] * w2 / 100)

    Final_df["weighted_sorting_score"] = weighted_sorting_score(Final_df)

    def bayesian_average_rating(n, confidence=0.95):
        if sum(n) == 0:
            return 0
        K = len(n)
        z = sct.norm.ppf(1 - (1 - confidence) / 2)
        N = sum(n)
        first_part = 0.0
        second_part = 0.0
        for k, n_k in enumerate(n):
            first_part += (k + 1) * (n[k] + 1) / (N + K)
            second_part += (k + 1) * (k + 1) * (n[k] + 1) / (N + K)
        score = first_part - z * math.sqrt((second_part - first_part * first_part) / (N + K + 1))
        return score

    Final_df["bar_score"] = Final_df.apply(lambda x: bayesian_average_rating(x[["Star_1",
                                                                                "Star_2",
                                                                                "Star_3",
                                                                                "Star_4",
                                                                                "Star_5"]]), axis=1)

    def hybrid_sorting_score(dataframe, bar_w=60, wss_w=40):
        bar_score = dataframe.apply(lambda x: bayesian_average_rating(x[["Star_1",
                                                                         "Star_2",
                                                                         "Star_3",
                                                                         "Star_4",
                                                                         "Star_5"]]), axis=1)
        wss_score = weighted_sorting_score(dataframe)

        return (bar_score * bar_w / 100) + (wss_score * wss_w / 100)

    Final_df["hybrid_sorting_score"] = hybrid_sorting_score(Final_df)

    Final_df['Country_name'] = Final_df['Hotel_Address'].apply(
        lambda x: 'United Kingdom' if 'United Kingdom' in x else x.split()[-1])


    return Final_df

df_1= get_data_1()

@st.cache_data
def df_Creator_for_nation ():
    df1 = pd.read_csv("part1.csv")
    df2 = pd.read_csv("part2.csv")
    df3 = pd.read_csv("part3.csv")

    df = pd.concat([df1, df2, df3], ignore_index=True)

    df = df.drop(columns=["Review_Date", "Review_Total_Negative_Word_Counts", "Review_Total_Positive_Word_Counts",
                          "Total_Number_of_Reviews_Reviewer_Has_Given", "lat", "lng"], axis=1)
    df["Reviewer_Nationality"] = df["Reviewer_Nationality"].str.strip().str.strip("'\"")
    return df

df_reviews = df_Creator_for_nation()
country_counts_head = (df_reviews["Reviewer_Nationality"]
                           .value_counts()
                           .sort_values(ascending=False)
                           .head(30)
                           .reset_index())


@st.cache_data
def top_10_hotels_by_country_filtered(dataframe, Country_name, nation_selected=False):
    Country_hotels = dataframe[dataframe['Country_name'] == Country_name]

    if nation_selected:
        selected_columns = [
            "Hotel_Name", 'Hotel_Address', 'Nation_Based_Weighted_Score',
            'Total_Number_of_Reviews', 'Negative_Review', 'Positive_Review',
            'hybrid_sorting_score', 'Country_name'
        ]
    else:
        selected_columns = [
            "Hotel_Name", 'Hotel_Address',
            'Total_Number_of_Reviews', 'Negative_Review', 'Positive_Review',
            'hybrid_sorting_score', 'Country_name'
        ]

    top_10_hotels = Country_hotels[selected_columns].sort_values(by="hybrid_sorting_score", ascending=False).head(10)


    if 'Nation_Based_Weighted_Score' in top_10_hotels.columns:
        top_10_hotels['Nation_Based_Weighted_Score'] = top_10_hotels['Nation_Based_Weighted_Score'].round(2)
    top_10_hotels['hybrid_sorting_score'] = top_10_hotels['hybrid_sorting_score'].round(2)

    rename_columns = {
        "Hotel_Name": "Hotel Name",
        'Hotel_Address': 'Address',
        'Nation_Based_Weighted_Score': 'Nation Score',
        'Total_Number_of_Reviews': 'Total Reviews',
        'Negative_Review': 'Negative Reviews',
        'Positive_Review': 'Positive Reviews',
        'hybrid_sorting_score': 'Score',
        'Country_name': 'Country'
    }
    top_10_hotels = top_10_hotels.rename(columns={k: v for k, v in rename_columns.items() if k in top_10_hotels.columns})

    top_10_hotels.reset_index(drop=True, inplace=True)
    top_10_hotels.index = top_10_hotels.index + 1

    return top_10_hotels

# User selections
user_selected_nation = recomandations_tab.selectbox(
    "Select a nation",
    options=["All Nations"] + country_counts_head.Reviewer_Nationality.unique().tolist()
)

if user_selected_nation == "All Nations":
    user_selected_nation = None  # "All Nations" seçilmişse nation seçimini devre dışı bırak

user_selected_country = recomandations_tab.selectbox("Select a country",
                                                     options=df
                                                     .Country_name
                                                     .unique())

if recomandations_tab.button("List Hotels", use_container_width=True, icon="🥂"):
    if user_selected_nation:  # Eğer bir nation seçilmişse
        df_reviews_filtered = get_data(nation_x=[user_selected_nation])
        top_hotels = top_10_hotels_by_country_filtered(
            df_reviews_filtered,
            Country_name=user_selected_country,
            nation_selected=True)

    else:
        df_reviews_filtered = df_1.copy()
        top_hotels = top_10_hotels_by_country_filtered(
            df_reviews_filtered,
            Country_name=user_selected_country,
            nation_selected=False)

   # Create a table to show hotels to the user
    recomandations_tab.write(f"Top 10 Hotels for {user_selected_country} - Feedback from {user_selected_nation if user_selected_nation else 'All Nations'}")
    recomandations_tab.dataframe(top_hotels, use_container_width=True)








