import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import plotly.express as px 
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="GetAround",
    page_icon="🚗",
    layout="wide"
)

st.title("🚙 GetAround delay analysis 🕓")

st.subheader("Load data")
### Load data
DATA_URL = "https://full-stack-assets.s3.eu-west-3.amazonaws.com/Deployment/get_around_delay_analysis.xlsx"

@st.cache_data
def load_data():
    data = pd.read_csv('df_getaround_delay.csv')
    return data

data_load_state = st.text('Loading data...')
df_delay = load_data()
data_load_state.text("") # change text from "Loading data..." to "" once the the load_data function has run

# Display raw data if box checked
if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(df_delay) 

st.write('**Total entries :** ', df_delay.shape[0])

st.markdown("***")

st.header("Checkin types")

checkin_type_counts = df_delay['checkin_type'].value_counts()
checkin_type_percentage = (checkin_type_counts / checkin_type_counts.sum()) * 100

fig1 = px.pie(
    df_delay['checkin_type'].value_counts(normalize=True).reset_index(),
    names='checkin_type',
    values='proportion',
    color_discrete_sequence=["#4B9AC7", "#9DD4F3"],
    title='Distribution of checkin types',
)
fig1.update_traces(textposition='inside', textinfo='percent+label')
fig1.update_layout(showlegend=False)

avg_delay_by_checkin = df_delay.groupby(
    'checkin_type')['delay_at_checkout_in_minutes'].mean().reset_index()

fig2 = px.bar(avg_delay_by_checkin, 
             x='checkin_type', 
             y='delay_at_checkout_in_minutes', 
             color='checkin_type', 
             color_discrete_sequence=["#4B9AC7", "#4BE8E0"],
             labels={'checkin_type': 'Checkin type', 
                     'delay_at_checkout_in_minutes': 'Average delay'},
             title='Average delay at checkout by checkin type')

col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(fig1)

with col2:
    st.plotly_chart(fig2)


fig = px.histogram(
    df_delay[~df_delay['delay_interval'].isin(["Early or on time", "Unknown"])],
    x='checkin_type',
    color='delay_interval',
    color_discrete_sequence = [
        "#4B9AC7", "#4BE8E0", "#9DD4F3", "#97FBF6", "#2A7FAF", "#23B1AB", "#0E3449", "#015955"],
    labels = {
        "delay_interval" : "Delay interval", 
        "checkin_type" : "Checkin type", 
        "count" : "Count"},
    title="Distribution of delay by intervals and depending on checkin type")
st.plotly_chart(fig, use_container_width=True)

st.markdown("***")

st.header("Delay analysis")

delay_counts = df_delay['delay_interval'].value_counts()
delay_percentages = (delay_counts / delay_counts.sum()) * 100
late_count = sum(delay_counts[i] for i in range(2, len(delay_counts)))
late_perc = sum(delay_percentages[i] for i in range(2, len(delay_percentages)))

labels = ['Early or on time', 
          '< 30 minutes', 
          '< 1 hour', 
          '1 to 2 hours', 
          '2 to 5 hours', 
          '5 to 24 hours', 
          '1 day or more', 
          'Unknown']

fig1 = go.Figure(data=[
    go.Bar(
        x=delay_counts.index,
        y=delay_counts,
        text=delay_percentages.round(2).astype(str) + '%',
        textposition='outside',
        marker=dict(color="#4BE8E0"),
    )
])

fig1.update_layout(
    title='Distribution of delay by category',
    xaxis=dict(title='Delay', categoryorder='array', categoryarray=labels),
    yaxis=dict(title='Count'),
    showlegend=False,
)

df_impacted = df_delay[df_delay["impacted"] == 1]
impacted_cat = [
    (df_impacted['previous_rental_delay'] < 30),
    (df_impacted['previous_rental_delay'] < 60),
    (df_impacted['previous_rental_delay'] < 120),
    (df_impacted['previous_rental_delay'] < 300),
    (df_impacted['previous_rental_delay'] < 1440),
    (df_impacted['previous_rental_delay'] >= 1440),
]
impacted_labels = ['< 30 minutes', 
                   '< 1 hour', 
                   '1 to 2 hours', 
                   '2 to 5 hours', 
                   '5 to 24 hours', 
                   '1 day or more']

df_impacted['previous_delay_interval'] = np.select(impacted_cat, impacted_labels)

impacted_perc_values = (
    df_impacted['previous_delay_interval'].value_counts(
        normalize=True)[impacted_labels] * 100).round(2)
impacted_perc_text = [f'{p:.2f}%' for p in impacted_perc_values]

fig2 = px.histogram(
    df_impacted,
    x='previous_delay_interval',
    histnorm = 'percent',
    labels = {'previous_delay_interval': 'Previous rental delay interval'},
    title='Distribution of impactful delays intervals')

fig2.update_xaxes(categoryorder='array', categoryarray=impacted_labels)
fig2.update_layout(yaxis_title='Percentage')
fig2.update_traces(marker_color = "#0E3449", hovertemplate='%{y:.2f}%', 
                   text=impacted_perc_text, textposition='outside')

selected_graph = st.selectbox(
    "Select Graph", 
    ["Distribution of all delays by categories", 
     "Distribution of impactful delays by categories"],
     key="selected_graph")

if selected_graph == "Distribution of all delays by categories":
    st.plotly_chart(fig1, use_container_width=True)
elif selected_graph == "Distribution of impactful delays by categories":
    st.plotly_chart(fig2, use_container_width=True)

fig = px.histogram(
    df_impacted,
    x='previous_rental_delay',
    nbins = 100,
    range_x = [0, 1000],
    histnorm = 'percent',
    labels = {'previous_rental_delay': 'Previous rental delay', 'count': 'Count'},
    title='Distribution of impactful delays')

fig.update_traces(marker_color = "#0E3449")
st.plotly_chart(fig, use_container_width=True)

st.header("General statistics : ")
left_column, right_column = st.columns(2)

with left_column:
    st.subheader("Checkout time percentage breakdown :")
    st.write(
        f'''Early/Timely checkouts: {delay_percentages[0]:.2f}%  
        Late checkouts: {late_perc:.2f}%  
        Unknown: {delay_percentages[1]:.2f}%'''
    )

with right_column:
    st.subheader("Impactful delays :")
    st.write(
        f"""Total number of impacted rentals : {df_impacted.shape[0]} 
        (or {df_impacted.shape[0]*100/df_delay.shape[0]:.2f}% of total rentals)  
        Average duration of impactful delay : {df_impacted.previous_rental_delay.mean():.0f} minutes  
        Average waiting time for impacted user : {abs(df_impacted.waiting_time.mean()):.0f} minutes  
        Proportion of impactful delays : {df_impacted.shape[0]*100/late_count:.2f}%"""
    )

st.markdown("***")

st.header("Cancellations")

gen_cxl_rate = df_delay["state"].value_counts(normalize=True)[1]*100
no_delay_cxl_rate = df_delay[df_delay["impacted"] == 0]["state"].value_counts(normalize=True)[1]*100
delay_cxl_rate = df_delay[df_delay["impacted"] == 1]["state"].value_counts(normalize=True)[1]*100

cxl_labels = ['All Users', 'Users without Delay', 'Users with Delay']
cxl_rates = [gen_cxl_rate.round(2), no_delay_cxl_rate.round(2), delay_cxl_rate.round(2)]

fig1 = px.bar(
    x=cxl_labels,
    y=cxl_rates,
    color = cxl_labels,
    color_discrete_sequence=["#4B9AC7", "#9DD4F3", "#0E3449"],
    labels={'x': 'User category', 'y': 'Cancellation rate in %'},
    title='Impact of delays on cancellation rate',
    text=[f'{p:.2f}%' for p in cxl_rates]
)
fig1.update_traces(hovertemplate='%{y:.2f}%', textposition='outside')
fig1.update_layout(showlegend=False)

mobile_cxl_rate = df_delay[df_delay["checkin_type"] == "mobile"]["state"].value_counts(normalize=True)[1] * 100
connect_cxl_rate = df_delay[df_delay["checkin_type"] == "connect"]["state"].value_counts(normalize=True)[1] * 100

cxl_labels = ['All users', 'Users without delay', 'Users with delay']
mobile_cxl_rates = [gen_cxl_rate.round(2), no_delay_cxl_rate.round(2), mobile_cxl_rate.round(2)]
connect_cxl_rates = [gen_cxl_rate.round(2), no_delay_cxl_rate.round(2), connect_cxl_rate.round(2)]

df_grouped = pd.DataFrame({
    'User category': cxl_labels * 2,
    'Cancellation rate': mobile_cxl_rates + connect_cxl_rates,
    'Checkin type': ['mobile'] * 3 + ['connect'] * 3,
})

fig_grouped = px.bar(
    df_grouped,
    x='User category',
    y='Cancellation rate',
    color='Checkin type',  
    barmode='group',  # Set barmode to 'group' for side-by-side bars
    color_discrete_sequence=["#4B9AC7", "#9DD4F3"],
    labels={'User category': 'User category', 'Cancellation rate': 'Cancellation rate in %'},
    title='Cancellation rates for "mobile" and "connect" checkin types',
    text = [f'{p:.2f}%' for p in df_grouped['Cancellation rate']]
)
fig_grouped.update_traces(hovertemplate='%{y:.2f}%', textposition='outside')

col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(fig1)

with col2:
    st.plotly_chart(fig_grouped)

st.markdown("***")

st.header("Threshold impact")

st.subheader("How would a threshold feature affect Getaround car owners profit and problematic delays?")
st.write("""A higher threshold value will decrease problematic rentals frenquency and increase customer satisfaction  
         However, the higher the threshold, the more potential transactions will be forfeit, impacting sales""")
st.write("Would users be impacted differently based on checkin type?")

def analyze_data(df):
    threshold_values = range(0, 301, 15)

    total_rentals = len(df)
    total_delayed = len(df[df["delay_at_checkout_in_minutes"] > 0])
    total_impacted = len(df[df["impacted"] == 1])
    total_cancelled = len(df[df["state"] == "canceled"])

    lost_counts = []
    lost_rates = []
    delayed_counts = []
    delayed_rates = []
    impacted_counts = []
    impacted_rates = []
    impacting_delays_rates = []
    solved_counts = []
    solved_rates = []
    cancelled_counts = []
    cancelled_rates = []

    for threshold in threshold_values:

        lost_rentals = df[df['time_delta_with_previous_rental_in_minutes'] < threshold]
        lost_rentals_count = len(lost_rentals)
        lost_rentals_proportion = lost_rentals_count * 100 / total_rentals

        unaffected = df.drop(lost_rentals.index, axis=0)

        delayed = df[df["delay_at_checkout_in_minutes"] > threshold]
        delayed_count = len(delayed)
        delayed_rate = delayed_count / total_delayed

        impacted = unaffected[unaffected["impacted"] == 1]
        impacted_count = len(impacted)
        impacted_rate = impacted_count / total_impacted
        impacting_delays_rate = impacted_count / total_delayed

        solved = lost_rentals[lost_rentals["impacted"] == 1]
        solved_count = len(solved)
        solved_rate = solved_count / total_impacted

        cancelled_count = unaffected["state"].value_counts()["canceled"]
        cancelled_rate = cancelled_count / total_cancelled

        lost_counts.append(lost_rentals_count)
        lost_rates.append(lost_rentals_proportion)
        delayed_counts.append(delayed_count)
        delayed_rates.append(delayed_rate)
        impacted_counts.append(impacted_count)
        impacted_rates.append(impacted_rate)
        impacting_delays_rates.append(impacting_delays_rate)
        solved_counts.append(solved_count)
        solved_rates.append(solved_rate)
        cancelled_counts.append(cancelled_count)
        cancelled_rates.append(cancelled_rate)

    plot_data = pd.DataFrame({
        'threshold': threshold_values,
        'lost_rentals_count': lost_counts,
        'lost_rentals_proportion': lost_rates,
        'delayed_counts': delayed_counts,
        'delayed_rates': delayed_rates,
        'impacted_counts': impacted_counts,
        'impacted_rates': impacted_rates,
        'impacting_delays_rates': impacting_delays_rates,
        'solved_counts': solved_counts,
        'solved_rates': solved_rates,
        'cancelled_counts': cancelled_counts,
        'cancelled_rates': cancelled_rates
    })

    return plot_data

scope_options = ['All rentals', 'Mobile rental', 'Connect rentals']
selected_scope = st.radio('Scope: ', scope_options, key="selected_scope")
if selected_scope == 'All rentals':
    plot_data = analyze_data(df_delay)
elif selected_scope == 'Mobile rental':
    df_mobile = df_delay[df_delay["checkin_type"] == 'mobile']
    plot_data = analyze_data(df_mobile)
else:
    df_connect = df_delay[df_delay["checkin_type"] == 'connect']
    plot_data = analyze_data(df_connect)

fig = go.Figure()

fig.add_trace(go.Scatter(x=plot_data['threshold'], y=plot_data['impacted_rates'], name='Impacted rate', yaxis='y1'))
fig.add_trace(go.Scatter(x=plot_data['threshold'], y=plot_data['lost_rentals_proportion'], name='Rental loss rate', yaxis='y2'))

# Update layout to include two y-axes
fig.update_layout(
    title='Impact of threshold on rental loss and problem rate',
    xaxis_title='Threshold',
    yaxis=dict(title='Delay impacted rate', side='right', showgrid=False),
    yaxis2=dict(title='Rental loss rate', side='left', overlaying='y'),
    legend=dict(x=0.5, y=1.2)
)

st.plotly_chart(fig, use_container_width=True)

st.write("You can try out several threshold values and see their impact for yourself!")

threshold = st.slider('Delay threshold (minutes)', 
                      min_value=0, 
                      max_value=300,
                      step=15, 
                      value=0, key="delay_threshold")

scope_options = ['All rentals', 'Mobile rental', 'Connect rentals']
selected_scope = st.radio('Scope: ', scope_options, key="selected_scope2")

if selected_scope == 'All rentals':
    plot_data = analyze_data(df_delay)
elif selected_scope == 'Mobile rental':
    df_mobile = df_delay[df_delay["checkin_type"] == 'mobile']
    plot_data = analyze_data(df_mobile)
else:
    df_connect = df_delay[df_delay["checkin_type"] == 'connect']
    plot_data = analyze_data(df_connect)

solved_rate_for_threshold = plot_data.loc[(plot_data["threshold"] == threshold)]['solved_rates'].values[0].round(2)*100
rental_loss_for_threshold = plot_data.loc[(plot_data["threshold"] == threshold)]['lost_rentals_proportion'].values[0].round(2)

st.write(f"""For a threshold of {threshold} minutes, applied to {selected_scope} :  
         {solved_rate_for_threshold}% problematic cases are solved  
         {rental_loss_for_threshold}% rental sales are missed""")