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
# import sweetviz as sv
import streamlit as st
import warnings 
warnings.filterwarnings('ignore')

import sys
import os
from PIL import Image


# Load the Dataset
##############################################
st.title('Data Analysis App')
st.write('Created by `Mehedi Hasan`')
# image upload
img = Image.open("imgs/pic.jpg")  
# resized_img = img.resize((800, 550))
# st.image(resized_img)
st.image(img)



st.write('---')
st.markdown('### Choose a CSV or Excel file')

# Function to load dataset with caching
@st.cache_data
def load_dataset():
    # Replace with your dataset loading logic
    df = pd.read_csv('your_large_dataset.csv')  # Example for CSV
    return df

uploaded_file = st.file_uploader('Upload a file', type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    else:
        st.error('Unknown format')

    # Show/hide dataset based on checkbox
    '---'
    show_dataset = st.checkbox('Show Dataset👇')
    if show_dataset:
        st.subheader('💾Dataset✅')
        st.dataframe(df)
    else:
        st.write('👆Check the box above to show the dataset.')
    '---'


    ###################################################################################################

    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(exclude=['number']).columns.tolist()


    # Filtering operations
    #####################################################################################
    st.sidebar.markdown('## 🔎Filtering Options')
    st.sidebar.write('------------------------')

    # Remove columns with only one unique value from numeric columns
    num_cols = [col for col in num_cols if df[col].nunique() > 1]
    numeric_filters = {}
    for col in num_cols:
        min_val, max_val = float(df[col].min()), float(df[col].max())
        numeric_filters[col] = st.sidebar.slider(
            f"Filter {col}", min_value=min_val, max_value=max_val, value=(min_val, max_val)
        )
    
    # multiselect & radio filter for Categorical columns ক
    categorical_filters = {}
    for col in cat_cols:
        unique_vals = df[col].dropna().unique().tolist()
        if len(unique_vals) > 3:
            # Multiselect options
            categorical_filters[col] = st.sidebar.multiselect(
                f"Filter {col}", options=unique_vals, default=unique_vals
            )
        else:
            # Radio button 
            categorical_filters[col] = st.sidebar.radio(
                f"Filter {col}", options=['All'] + unique_vals, index=0
            )
    
    # Applying filteres on a new coppied dataset
    # st.subheader('Filterable Dataset')
    filtered_df = df.copy()
    for col, (min_val, max_val) in numeric_filters.items():
        filtered_df = filtered_df[(filtered_df[col] >= min_val) & (filtered_df[col] <= max_val)]
    
    for col, selected_values in categorical_filters.items():
        if isinstance(selected_values, list) and selected_values:  # Multiselect ফিল্টার
            filtered_df = filtered_df[filtered_df[col].isin(selected_values)]
        elif selected_values != 'All':  # Radio ফিল্টার
            filtered_df = filtered_df[filtered_df[col] == selected_values]


    
    show_dataset = st.checkbox('Show Filterable Dataset👇',  key='filter dataset')
    if show_dataset:
        st.subheader('💾Filterable Dataset✅')
        st.dataframe(filtered_df)
    else:
        st.write('👆Check the box above to show the dataset.')
    '---'

    ######################################################################################

    # Display basic dataset info
    col1, col2 = st.columns(2)
    col1.success(f'✅Total rows: {df.shape[0]}')
    col2.success(f'✅Total columns: {df.shape[1]}')

    col1.success(f'✅Numeric columns: {len(num_cols)}')
    for i, col in enumerate(num_cols, 1):
        col1.write(f"{i}. {col}")
    
    col2.success(f'✅Categorical columns: {len(cat_cols)}')
    for i, col in enumerate(cat_cols, 1):
        col2.write(f"{i}. {col}")
    
    st.write('-----------------------------------------')   

    # Data Cleaning Section
##############################################################################
    # Unique values of the columns
    st.subheader('✅Unique values of the dataset columns')
    unique_values = df[df.columns].nunique()
    unique_values_df = pd.DataFrame(unique_values).T
    st.write(unique_values_df)


    '---'
    # Duplicate check
    total_duplicate = df.duplicated().sum()
    st.subheader(f'✅Dataset has total duplicate records 👇: {total_duplicate}')

    show_duplicate = st.checkbox('Show Duplicated Records')
    if show_duplicate:
        st.success('💾Duplicated Records✅👇')
        if total_duplicate > 0:
            duplicate_records = df[df.duplicated()]
            duplicate_records.to_csv('duplicates.csv', index=None)
            st.dataframe(duplicate_records)
    else:
        st.write('👆Check the box above to show Duplicated Records.')


    '---'
    # Missing value check
    total_missing_value = df.isnull().sum().sum()
    missing_value_columns = df.isnull().sum()

    missing_value_df = pd.DataFrame({
        'Column': missing_value_columns[missing_value_columns > 0].index,
        'Missing Values': missing_value_columns[missing_value_columns > 0].values
    })

    with st.container():
        st.subheader(f'✅Dataset has total missing values: {total_missing_value}')
        st.dataframe(missing_value_df)
        st.write('-----------------------------------------')

##########################################################################################
    # Generate the Sweetviz report
    # Function to generate and cache the Sweetviz report
    # @st.cache_resource
    # def generate_sweetviz_report(df):
    #     sweet_report = sv.analyze(df)
    #     sweet_report_file = "sweet_report.html"
    #     sweet_report.show_html(filepath=sweet_report_file, open_browser=False)
    #     return sweet_report_file

    # # Display the Sweetviz report
    # st.subheader('Sweetviz Report')
    # report_file = generate_sweetviz_report(df)

    # # Read the cached HTML file and display it
    # with open(report_file, 'r') as f:
    #     report_html = f.read()

    # st.components.v1.html(report_html, width=1500, height=600, scrolling=True)

########################################################################################

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

    # Function to plot the distribution of a variable
    def univariate_ploting(df, variable):
        if variable not in df.columns:
            st.write(f"Column '{variable}' not found in the DataFrame.")
            return
        
        # Display unique values and their count
        unique_values = df[variable].unique()
        unique_count = df[variable].nunique()
        value_counts = df[variable].value_counts()

        # Prepare data for plot
        value_counts_df = value_counts.reset_index()
        value_counts_df.columns = [variable, 'count']

        # Create a bar chart with counts displayed above each bar
        fig = px.bar(value_counts_df, 
                     x=variable, y='count', color=variable,
                     labels={variable: f'{variable} Categories', 'count': 'Counts'}, 
                     text='count',  # Add count text on top of bars
                     template='plotly_dark'
                    )

        # Update the layout with title and styling
        fig.update_layout(
            title=f"<b>Count of transactions Distribution of {variable}</b>",
            titlefont={'color': None, 'size': 25, 'family': 'Times New Roman'},
            height=600, 
            width=1200,
        )

        # Automatically display text values on bars
        fig.update_traces(textposition='outside')  # Set position of text to be outside the bars

        # Display the plot in Streamlit
        st.plotly_chart(fig)
##############################################################################################



    '---'
    st.subheader('Univariate Distribution Plot and Observations')


    # creating 2 columns for Numerical and categorical columns
    col1, col2 = st.columns(2)

    # radio button for numerical columns
    with col1:
        st.markdown("#### Select a Numeric Column")
        selected_numeric = st.radio(
            "Numeric Columns:",
            options=[None]+num_cols,
            key="radio_numeric",
            index=0
        )

    # radio button for categorical columns
    with col2:
        st.markdown("#### Select a Categorical Column")
        selected_categorical = st.radio(
            "Categorical Columns:",
            options=[None]+cat_cols,
            key="radio_categorical",
            index=0
        )

    # predefined observations for numerical columns
    numeric_observations = {
        "transaction_qty": '✅Total unique values : `6` ✅Unique values : `[2 1 3 4 8 6]` ✅Mean value : `1.4383` ✅Standard Deviation : `0.5425` ✅Minimum value : `1` ✅Maximum value : `8`',
        "store_id": '✅Total unique values : `3` ✅Unique values : `[5 8 3]` ✅Mean value : `5.3421` ✅Standard Deviation : `2.0742` ✅Minimum value : `3` ✅Maximum value : `8`',
        "product_id": '✅Total unique values : `3` ✅Unique values : `[5 8 3]` ✅1st Quartile : 33 ✅Standard Deviation : 17.93 ✅Median : 47 ✅Minimum value : 1 ✅Maximum value : 87',
        "unit_price": '✅Total unique values : `41` ✅Unique values : `[ 3.    3.1   4.5   2.    4.25  3.5   2.55  3.75  2.5   4.75  3.25  4. 2.2   2.45  0.8   8.95 21.   28.    6.4  19.75 12.   18.    9.5  10. 20.45  9.25 22.5  14.75  7.6  14.   10.95 13.33 15.    4.38  4.06  5.63 4.69 45.   23.    2.1   2.65]` ✅Mean value : `3.3822` ✅Standard Deviation : `2.6587` ✅Minimum value : `0.8` ✅Maximum value : `45`',
        "year": '✅`2023` is the only year',
        "month": '✅Total unique values : `6` ✅The Unique values in are: `[1 2 3 4 5 6]` ✅Maximum(35352) minimum(16359) count occurs in 6th & 2nd month repectively. ✅`Second`(33527) & `Third` positions(25335) belogs to `5th` & `4th` month.',
        "day": '✅Total unique values : `30` ✅Unique values : `[ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31]`',
        "hour": '✅Total unique values : `6` ✅The Unique values in are: `[1 2 3 4 5 6]` ✅`Maximum(35352)` & `minimum(16359)` count occure in `6th` & `2nd` month repectively. ✅`Second(33527)` & `Third positions(25335)` belogs to `5th` & `4th` month. ✅Mean value : `3.9889` ✅Standard Deviation : `1.6731` ✅Minimum value : `1` ✅Maximum value : `6`',
        "minute":'✅Total unique values : `60` ✅Unique values : `[ 6  8 14 20 22 25 33 39 43 44 45 48 52 59  0 11 17 24 29 31 35 41 54 56 57 58  7 10 13 15 19 21 34 40 46 47 50 53 55  3 16 18 27 30 49 51  1  4 9 12 23 28  5 36 38 42  2 32 26 37]` ✅Mean value : `29.6239` ✅Standard Deviation : `17.2914` ✅Minimum value : `0` ✅Maximum value : `59`'
    }

    # predefined observations for categorical columns
    categorical_observations = {
        "store_location": '✅Total unique category : `3` ✅Unique categories : [`Hells Kitchen` `Astoria` `Lower Manhattan`] ✅Frequency/Mode : `Hells Kitchen`',
        "product_category": '✅Total unique values : `9` ✅Unique values : [`Coffee` `Tea` `Drinking Chocolate` `Bakery` `Flavours` `Loose Tea` `Coffee beans` `Packaged Chocolate` `Branded`] ✅Frequency/Mode : `Coffee` ✅Frequency/Mode : `Brewed Chai tea`',
        "product_type": '✅Total unique values : `29` ✅Unique values :[`Gourmet brewed coffee` `Brewed Chai tea` `Hot chocolate` `Drip coffee` & 24 more',
        "product_detail" : '✅Total unique values : `80` ✅Frequency/Mode : `Chocolate Croissant`'
    }

    # using container for shoing plots (numerical & categorical)
    if selected_numeric:
        with st.container():
            st.subheader(f'Distribution Plot and Observations for {selected_numeric} Column')
            univariate_ploting(df, selected_numeric)  # পূর্ণ প্রস্থে প্লট দেখানো
            st.write('### Observations')
            observation = numeric_observations.get(selected_numeric, "No specific observation available for this plot.")
            st.write(observation)
            st.write('---')

    if selected_categorical:
        with st.container():
            st.subheader(f'Distribution Plot and Observations for {selected_categorical} Column')
            univariate_ploting(df, selected_categorical)  # পূর্ণ প্রস্থে প্লট দেখানো
            st.write('### Observations')
            observation = categorical_observations.get(selected_categorical, "No specific observation available for this plot.")
            st.write(observation)
            st.write('---')

    #########################################################################
    # Bivariate Analysis ploting
    '---'
    def plot_bivariate_analysis(df, x_var, y_var):
        # Check if both variables exist in the DataFrame
        if x_var not in df.columns or y_var not in df.columns:
            st.write(f"Columns '{x_var}' or '{y_var}' not found in the DataFrame.")
            return

        # create a bivariate count DataFrame
        bivariate_counts = df.groupby([x_var, y_var]).size().reset_index(name='count')

        # Create a bar chart for bivariate analysis
        fig = px.bar(
            bivariate_counts, 
            x=x_var, y='count', color=y_var, 
            labels={x_var: x_var, y_var: y_var, 'count': 'Count'},
            template='plotly_dark',
            text='count'
        )

        # Updating layout 
        fig.update_layout(
            title=f"<b>Bivariate Analysis of {x_var} and {y_var}</b>",
            titlefont={'color': None, 'size': 25, 'family': 'Times New Roman'},
            height=600,
            width=1200,
            title_x=0.5,
            barmode='group'
        )

        # Displaying text outside bars
        fig.update_traces(textposition='outside')

        # Show the plot
        st.plotly_chart(fig, use_container_width=True)



    st.subheader("Bivariate Analysis with Observations")

    # adding "None" as a placeholder option in the dropdowns
    x_options = ["None"] + list(df.columns)
    y_options = ["None"] + list(df.columns)

    # Dropdowns for selecting X and Y variables
    selected_x_var = st.selectbox("Select X-axis variable", options=x_options)
    selected_y_var = st.selectbox("Select Y-axis variable", options=y_options)

    #  predefined observations for specific variable combinations
    bivariate_observations = {
        ("product_category", "store_location"): '✅In every `store_location` product_category : `Coffe` remains `1st position` ✅In every `store_location` product_category : `Tea` remains `2nd position` ✅In every `store_location` product_category : `Bakery` remains `3rd position` ✅In every `store_location` product_category : `Drinking Chocolate` remains `4th position` ✅In store_location `Astoria` & `Lower Manhattan` product_category : `Package Chocolate` is `last position` ✅In store_location `Hells Kitchen` product_category : `Branded` is `last position` ✅Recomendation: The trend of best selling & slow selling product_catery are same. ✅To boost up slow selling product, the shop can arange some attractive `discount & markeing` untill the selling increase.',
        ("store_id", "store_location"): '✅The store_id & store_location are same',
        ("product_type", "unit_price"): '✅product_type `Premium Bens` got the `maximum unit_price(45)` ✅product_type `Clothing` got the `second maximum unit_price(28)` ✅product_type `Organic Beans` got the `third maximum unit_price(22.5)`',
        ("product_id", "unit_price"): '✅product_id `8` has the `highest unit_price(45)` ✅product_id `81` has the `second highest unit_price(28)` ✅product_id `9` has the `highest unit_price(22.5)`',
        ("product_id", "unit_price"): '✅product_id `8` has the `highest unit_price(45)` ✅product_id `81` has the `second highest unit_price(28)` ✅product_id `9` has the `highest unit_price(22.5)`',
        ("product_type", "product_category"): '✅in `Coffee` product_category `Gourmet brewed coffe` then `Barista Expresso` are `best selling` product_type ✅in `Bakery` product_category `Scone` then `Pastry` are `best selling` product_type ✅in `Tea` product_category `Brewed Chai Tea` then `Brewed Black Tea` are `best selling` product_type ✅in `Flavours` product_category `Regular Syrup` then `Sugar Free Syrup` are `best selling` product_type ✅in `Drinking Chocolate` product_category `Hot Chocolate` are the `only best selling item`',
        ("transaction_qty", "store_location"): '✅store_location at `Asterio` has `1` & `2` transaction_quantity  ✅store_location at `Hells Kitchenerio` has `1` & `2` transaction_quantity  ✅store_location at `Lower Manhattan` has `1`, `2` & `3` transaction_quantity ',
        ("store_location", "product_type"): '✅in store_location `Astoria` : 1.`Brewed Chai Tea`, 2.`Gouemet brewed coffee`, 3.`Barista Expresso` are best selling ✅in store_location `Hells Kitchen` : 1.`Barista Expresso`, 2.`Brewed Chai Tea`, 3.`Gouemet brewed coffee` are best selling ✅in store_location `Lower Manhattan` : 1.`Barista Expresso`, 2.`Gouemet brewed coffee`, 3.`Brewed Chai Tea` are best selling',
        ("year", "product_category"): '✅The only year : 2023, 1.`Coffee`, 2.`Tea`, 3.`Bakery` are `best selling` product_categoey ✅1.`Packaged Chocolate`, .`Branded`, 3.`Loose Tea` are the `lowest selling` product_categoey  ✅Recomendation: lowest selling product_categoey can be discounted, displayed suitable place and special marketing & campaigning can be arranged, or it can be cut-off for future loss. Can intoduce alternative product of those.',
        ("month", "product_category"): '✅in every month 1:`coffee`, 2:`Tea`, 3.`Bakery` & 4.`Dringking Chocolate` are the `best selling` product_category',
        ("day", "product_category"): '✅in every day 1:`coffee`, 2:`Tea`, 3.`Bakery` & 4.`Dringking Chocolate` are the `best selling` product_category',
        ("minute", "product_category"): '✅in every `minute` 1:`coffee`, 2:`Tea`, 3.`Bakery` & 4.`Dringking Chocolate` are the `best selling` product_category',
        ("transaction_qty", "product_category"): '✅`Coffee` is the `higest(29177)`, `Tea` is the ` 2nd higest(22467)` & `Bakery` is the `3rd higest(22404)` single quantity (`1`) item ✅`Coffee` is also the `higest(27646)`, `Tea` is the ` 2nd higest(21676)` & `Drinking Chokolate` is the `3rd higest(22404)` `double quantity(2)` product',
    }

    # Display the plot and observations if both variables are selected
    if selected_x_var and selected_y_var:
        st.write(f"### Bivariate Analysis for {selected_x_var} and {selected_y_var}")
        
        # Calling the bivariate plotting function
        plot_bivariate_analysis(df, selected_x_var, selected_y_var)
        
        # Displaying observations based on selected variables
        st.write("### Observations & Recomendation(if needed)")
        observation = bivariate_observations.get((selected_x_var, selected_y_var), "No specific observation available for this plot.")
        st.write(observation)
    else:
        st.write("Please select both X and Y variables for bivariate analysis.")
        

    # Report Creation and download
    # শিরোনাম
    '---'
    st.subheader("Ultimate Report of Coffe Shope Sales")

    # বাটন ক্লিক করলে রিপোর্ট দেখানোর জন্য

    if st.checkbox("Show Report"):
        
        # Univariate Observations
        st.success("Univariate Observations")
        report_text = "### Univariate Observations\n\n"

    # Numeric Observations
        for column, observation in numeric_observations.items():
            st.write(f"📜 **{column}**")
            st.write(observation)
            report_text += f"**{column}**:\n{observation}\n\n"
            st.markdown('---')  # Divider line
        
        # Categorical Observations
        for column, observation in categorical_observations.items():
            st.write(f"📜 **{column}**")
            st.write(observation)
            report_text += f"**{column}**:\n{observation}\n\n"
            st.markdown('---')  # Divider line
        
        # Bivariate Observations
        st.success("Bivariate Observations")
        report_text += "\n### Bivariate Observations\n\n"
        for (col1, col2), observation in bivariate_observations.items():
            st.write(f"📜 **{col1} & {col2}**")
            st.write(observation)
            report_text += f"**{col1} & {col2}**:\n{observation}\n\n"
            st.markdown('---')  # Divider line


        # Download option for the report
        if st.button:
            st.download_button("Download Report", report_text, file_name="report.txt")  




