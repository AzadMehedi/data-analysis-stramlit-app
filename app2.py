# Import Dependencies
##############################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go


import re
import string
import streamlit as st
import warnings 
warnings.filterwarnings('ignore')

import sys
import os
from PIL import Image


# Load the Dataset
##############################################
st.title('Data Analysis App')
st.write('Created by `Mehedi Hasan`, `Statistical Analyst & Data Scientist`')
st.write('`B.Sc.` in `CSE` & `M.Sc.` in `Applied Statistics & Data Science`')

# image upload
img = Image.open("imgs/image.png")  
# resized_img = img.resize((800, 550))
# st.image(resized_img)
st.image(img)



st.write('---')
st.markdown('### Choose a CSV or Excel file')


# File upload widget
uploaded_file = st.file_uploader("Upload your file here", type=["csv", "xlsx"])
if uploaded_file is None:
    st.stop()  

if uploaded_file.name.endswith('.csv'):
    df = pd.read_csv(uploaded_file)
elif uploaded_file.name.endswith('.xlsx'):
    df = pd.read_excel(uploaded_file)

# # Showing DataFrame 
# st.write("Uploaded Dataset:")
# st.write(df)

# Checkbox of showing the dataset
st.markdown('---')
st.subheader('💾Dataset✅')
show_dataset = st.checkbox('👈Show Uploaded Dataset')
if show_dataset:
    st.dataframe(df)

'---'


###################################################################################################

num_cols = df.select_dtypes(include=['number']).columns.tolist()
cat_cols = df.select_dtypes(exclude=['number']).columns.tolist()


######################################################################################

# Display basic dataset info
col1, col2 = st.columns(2)
col1.success(f'✅Total rows: {df.shape[0]}')
col2.success(f'✅Total columns: {df.shape[1]}')
col1.success(f'✅Total Duplicate Values: {df.duplicated().sum()}')
col2.success(f'✅Total Missing Values: {df.isna().sum().sum()}')


col1.success(f'✅Numeric columns: {len(num_cols)}')
for i, col in enumerate(num_cols, 1):
    col1.write(f"{i}. {col}")

col2.success(f'✅Categorical columns: {len(cat_cols)}')
for i, col in enumerate(cat_cols, 1):
    col2.write(f"{i}. {col}")

st.write('-----------------------------------------')   

# Filtering operations
#####################################################################################
st.sidebar.markdown('## 🔎Filtering Options')
st.sidebar.write('------------------------')

# Remove columns with only one unique value from numeric columns
num_cols = [col for col in num_cols if df[col].nunique() > 1]
numeric_filters = {}
for col in num_cols:
    min_val = float(df[col].min())
    max_val = float(df[col].max())
    numeric_filters[col] = st.sidebar.slider(
        f"{col}", min_value=min_val, max_value=max_val, value=(min_val, max_val)
    )

# multiselect & radio filter for Categorical columns 
categorical_filters = {}
for col in cat_cols:
    unique_vals = df[col].dropna().unique().tolist()
    if len(unique_vals) > 3:
        # Multiselect options
        categorical_filters[col] = st.sidebar.multiselect(
            f"{col}", options=unique_vals, default=unique_vals
        )
    else:
        # Radio button 
        categorical_filters[col] = st.sidebar.radio(
            f"{col}", options=['All'] + unique_vals, index=0
        )

# Applying filteres on a new coppied dataset
filtered_df = df.copy()
for col, (min_val, max_val) in numeric_filters.items():
    filtered_df = filtered_df[(filtered_df[col] >= min_val) & (filtered_df[col] <= max_val)]

for col, selected_values in categorical_filters.items():
    if isinstance(selected_values, list) and selected_values:  # Multiselect ফিল্টার
        filtered_df = filtered_df[filtered_df[col].isin(selected_values)]
    elif selected_values != 'All':  # Radio ফিল্টার
        filtered_df = filtered_df[filtered_df[col] == selected_values]


st.subheader('💾Filterable Dataset✅')
show_dataset = st.checkbox('👈Show Filterable Dataset',  key='filter dataset')
if show_dataset:
    st.dataframe(filtered_df)

'---'



# Data Cleaning Section
##############################################################################
# Unique values of columns
st.subheader('✅Unique values of the dataset columns')
unique_values = df[df.columns].nunique()
unique_values_df = pd.DataFrame(unique_values).T
st.write(unique_values_df)


'---'
# Duplicate check
total_duplicate = df.duplicated().sum()
st.subheader(f'✅Dataset has total duplicate records : `{total_duplicate}`')

show_duplicate = st.checkbox('👈Show Duplicated Records')
if show_duplicate:
    st.success('💾Duplicated Records✅👇')
    if total_duplicate > 0:
        duplicate_records = df[df.duplicated()]
        duplicate_records.to_csv('duplicates.csv', index=None)
        st.dataframe(duplicate_records)

'---'
# Missing value check
total_missing_value = df.isnull().sum().sum()
missing_value_columns = df.isnull().sum()

missing_value_df = pd.DataFrame({
    'Column': missing_value_columns[missing_value_columns > 0].index,
    'Missing Values': missing_value_columns[missing_value_columns > 0].values
})

with st.container():
    st.subheader(f'✅Dataset has total missing values: `{total_missing_value}`')
    st.dataframe(missing_value_df)
    st.write('-----------------------------------------')



# **🖥️ ডেটা ক্লিনিং steps**
# columns cleaning
cleaned_df = df.copy()
cleaned_df.columns = df.columns.str.strip()  # শুরুর ও শেষের স্পেস সরানো
cleaned_df.columns = df.columns.str.replace(r'^[0-9._/]+|[0-9._/]+$', '', regex=True)  # নাম্বার ও আনওয়ান্টেড ক্যারেক্টার সরানো
cleaned_df = cleaned_df.dropna()
cleaned_df = cleaned_df.drop_duplicates()
cleaned_df = cleaned_df.loc[:, ~cleaned_df.columns.duplicated()]  # ডুপ্লিকেট কলাম ড্রপ করা
# # নাল (NaN) মান mode value দিয়ে করা
# df[column] = df[column].fillna(df[column].mode()[0])  # ফিলিং NaN মান মোড দিয়ে


# **🖥️ ক্লিন করা ডেটা দেখানো**
st.subheader("📝Cleaned Dataset")
show_dataset = st.checkbox('👈Show Cleaned Dataframe', key='cleaned Dataset')
if show_dataset:
    st.dataframe(cleaned_df)
csv = cleaned_df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download Cleaned Dataset", csv, file_name='cleaned_df.csv')

##########################################################################################

'---'
# Descriptive analysis of Numerical Columns
st.subheader('📝Descriptive Analysis of Numerical Columns👇')
univariate_stat = {}
for col in num_cols:
    univariate_stat[col] = {
        'count': df[col].count(),
        'mean': df[col].mean(),
        'std': df[col].std(),
        'min': df[col].min(),
        '25%': df[col].quantile(0.25),
        '50% (median)': df[col].median(),
        '75%': df[col].quantile(0.75),
        'max': df[col].max()
    }
univariate_df = pd.DataFrame(univariate_stat).T
st.write(univariate_df)

# Descriptive analysis of Categorical Columns
st.subheader('📝Descriptive Analysis of Categorical Columns👇')
univariate_stat = {}
for col in cat_cols[:-1]:  # Adjusting to avoid index error on last categorical column
    univariate_stat[col] = {
        'Unique categories': df[col].nunique(),
        'frequency': df[col].mode().values[0],
        'Percentage Distribution': df[col].value_counts(normalize=True) * 100 
    }
univariate_df = pd.DataFrame(univariate_stat).T
st.write(univariate_df)

#####################################################################################




'---'
# Select preffered visualization
st.subheader("Visualization Types")
visualization_type = st.radio(
    "`Select a Visualization Type`",
    ["👇", "Univariate Numerical", "Univariate Categorical", "Bivariate", "Pair PLot", "Heatmap"]
)

if visualization_type == "Univariate Numerical":
    # ইউনিভ্যারিয়েট নিউমেরিক্যাল ভিজ্যুয়ালাইজেশন
    st.success("Univariate Numerical Visualization")

    col1, col2 = st.columns(2)
    with col1.container():
        numeric_column = st.selectbox("`Select a Numerical Column`", ['Select a column👉']+num_cols)  # df.select_dtypes(include=['float64', 'int64']).columns
    with col2.container():   
        plot_type = st.radio(
            "`Select a plot type`",
            ["👇", "histogram", "box plot", "line plot"]
        )

    if plot_type == "histogram":
        st.success(f"{numeric_column}'s Histogram PLot")
        fig = px.histogram(df, x=numeric_column)
        st.plotly_chart(fig)
    
    elif plot_type == "box plot":
        st.success(f"{numeric_column}'s Box PLot")
        fig = px.box(df, y=numeric_column)
        st.plotly_chart(fig)
    
    elif plot_type == "line plot":
        st.success(f"{numeric_column}'s Line PLot")
        fig = px.line(df, y=numeric_column)
        st.plotly_chart(fig)

    # নিউমেরিক্যাল ভ্যারিয়েবলের পরিসংখ্যানিক তথ্য
    st.subheader(f"`Statistical Information of` : `{numeric_column}`")
    col1,col2,col3 = st.columns(3)
    with col1.container():
        st.write(f"`Total Count`: {df[numeric_column].count()}")
        st.write(f"`Total Unic Value`: {df[numeric_column].nunique()}")
        st.write(f"`Minimum Value`: {df[numeric_column].min()}")
        
    with col2.container():
        st.write(f"`Maximum Value`: {df[numeric_column].max()}")
        st.write(f"`Mean Value`: {df[numeric_column].mean()}")
        st.write(f"`Std`: {df[numeric_column].std()}") 
    with col3.container():
        st.write(f"`Median`: {df[numeric_column].median()}")
        st.write(f"`Unic Values`: {df[numeric_column].unique().tolist()}")

elif visualization_type == "Univariate Categorical":
    # ইউনিভ্যারিয়েট ক্যাটেগরিক্যাল ভিজ্যুয়ালাইজেশন
    st.success("Univariate Categorical Visualization")
    col1, col2 = st.columns(2)
    with col1.container(): 
        categorical_column = st.selectbox("`Select a Categorical Column`", ['Select a column👉']+cat_cols)  # df.select_dtypes(include=['object', 'category']).columns
    with col2.container():
        plot_type = st.radio(
            "`Select a plot type`",
            ["👇", "Bar plot", "Pie plot", "count plot"]
        )
    
    if plot_type == "Bar plot":
        st.success(f"{categorical_column}'s Bar PLot")
        fig = px.bar(df, x=categorical_column)
        st.plotly_chart(fig)
    
    elif plot_type == "Pie plot":
        st.success(f"{categorical_column}'s Pie PLot")
        fig = px.pie(df, names=categorical_column)
        st.plotly_chart(fig)
    
    elif plot_type == "count plot":
        st.success(f"{categorical_column}'s Count PLot")
        fig = px.histogram(df, x=categorical_column)
        st.plotly_chart(fig)

    # ক্যাটেগরিক্যাল ভ্যারিয়েবলের পরিসংখ্যানিক তথ্য
    st.success(f"Statistical Information of : {categorical_column}")
    col1,col2 = st.columns(2)
    with col1.container():
        st.write(f"`Total Count`: {df[categorical_column].count()}")
        st.write(f"`Total Unic Value`: {df[categorical_column].nunique()}")
        st.write(f"`Unic Values`: {df[categorical_column].unique().tolist()}")
        st.write(f"`Mode`: {df[categorical_column].mode().tolist()}")

    with col2.container():
        freq_table = df[categorical_column].value_counts().reset_index()
        freq_table.columns = [categorical_column, "Frequency"]
        st.write('`Frequency`')
        st.table(freq_table)

elif visualization_type == "Bivariate":
    # বাইভ্যারিয়েট ভিজ্যুয়ালাইজেশন
    st.success("Bivariate Visualization")

    # adding "None" as a placeholder option in the dropdowns
    x_options = ["Select one 👉"] + list(df.columns)
    y_options = ["Select one 👉"] + list(df.columns)

    col1, col2 = st.columns(2)
    with col1.container(): 
        x_axis = st.selectbox("`Select a Column for X-Axis`", x_options)
        y_axis = st.selectbox("`Select a Column for Y-Axis`", y_options)
    with col2.container():
        plot_type = st.radio(
            "`Select a Plot Type`",
            ["None", "scatter plot", "line plot", "box plot","bar plot", "histogram"]
        )
    if plot_type == 'None':
        st.write('`Please select a plot type first`👆')
    
    elif plot_type == "scatter plot":
        fig = px.scatter(df, x=x_axis, y=y_axis, title=f"{x_axis} vs {y_axis} Scatter Plot")
        st.plotly_chart(fig)
    
    elif plot_type == "line plot":
        fig = px.line(df, x=x_axis, y=y_axis, title=f"{x_axis} vs {y_axis} Line Plot")
        st.plotly_chart(fig)
    
    elif plot_type == "box plot":
        fig = px.box(df, x=x_axis, y=y_axis, title=f"{x_axis} vs {y_axis} Box Plot")
        st.plotly_chart(fig)
    
    elif plot_type == "bar plot":
        fig = px.bar(df, x=x_axis, y=y_axis, title=f"{x_axis} vs {y_axis} Bar Chart")
        st.plotly_chart(fig)
    
    elif plot_type == "histogram":
        fig = px.histogram(df, x=x_axis, y=y_axis, title=f"{x_axis} vs {y_axis} Histogram")
        st.plotly_chart(fig)

elif visualization_type == "Pair PLot":
    st.success("Pair PLot Visualization")
    fig = sns.pairplot(filtered_df)
    st.pyplot(fig)

elif visualization_type == "Heatmap":
    st.success("Heatmap Visualization")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df.select_dtypes(include=["float64", "int64"]).corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
    st.pyplot(fig)
        
#####################################################################################


    


# Report Creation and download
'---'
st.subheader('📜Ultimate Report of the Dataset✅')
# 🎯 Univariate Analysis
if st.checkbox('👈Show Report'):
    st.success("📊 Univariate Analysis")
    report_text = "# Ultimate Data Analysis Report\n\n"
    report_text += "## Univariate Analysis\n\n"

    # **Numeric Columns Analysis**
    numeric_observations = {}
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    for col in num_cols:
        desc = df[col].describe()
        observation = f"`Mean`: {desc['mean']:.2f}, `Median`: {df[col].median():.2f}, `Std`: {desc['std']:.2f}, `Min`: {desc['min']}, `Max`: {desc['max']}"
        st.write(f"📌 **{col}** - {observation}")
        numeric_observations[col] = observation
        report_text += f"**{col}**: {observation}\n\n"

    # **Categorical Columns Analysis**
    categorical_observations = {}
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        value_counts = df[col].value_counts()
        observation = f"`Top Category`: {value_counts.idxmax()} ({value_counts.max()} times), `Unique Values`: {df[col].nunique()}"
        st.write(f"📌 **{col}** - {observation}")
        categorical_observations[col] = observation
        report_text += f"**{col}**: {observation}\n\n"
    '---'
    # 🎯 Bivariate Analysis (Correlation & Crosstab)
    st.success("🔗 Bivariate Analysis")
    st.write('`Numeric vs Numeric column`')
    report_text += "## Bivariate Analysis\n\n"

    # **Numeric Correlations**
    bivariate_observations = {}
    if len(num_cols) > 1:
        correlation_matrix = df[num_cols].corr()
        for i in range(len(num_cols)):
            for j in range(i + 1, len(num_cols)):
                col1, col2 = num_cols[i], num_cols[j]
                corr_value = correlation_matrix.loc[col1, col2]
                observation = f"Correlation: {corr_value:.2f}"
                st.write(f"📌 **{col1} & {col2}** - `{observation}`")
                bivariate_observations[(col1, col2)] = observation
                report_text += f"**{col1} & {col2}**: {observation}\n\n"

    # **Categorical Crosstab**
    '---'
    st.write('`Categorical vs Categorical column`')
    if len(cat_cols) > 1:
        for i in range(len(cat_cols)):
            for j in range(i + 1, len(cat_cols)):
                col1, col2 = cat_cols[i], cat_cols[j]
                crosstab = pd.crosstab(df[col1], df[col2])
                observation = f"Crosstab Shape: {crosstab.shape}"
                st.write(f"📌 **{col1} & {col2}** - `{observation}`")
                bivariate_observations[(col1, col2)] = observation
                report_text += f"**{col1} & {col2}**: {observation}\n\n"

    # 📩 **Download Button**
    '---'
    st.success('Download Report & Cleaned Dataset')
    col1, col2 = st.columns(2)
    with col1.container():
        st.subheader("📥Report")
        st.download_button("📜Download Report", report_text, file_name="Data_Analysis_Report.txt", key='report')
    with col2.container():
        st.subheader("📥Clened Dataset")
        st.download_button("💾Download Cleaned Dataset", csv, file_name='cleaned_df.csv', key='dataset')











