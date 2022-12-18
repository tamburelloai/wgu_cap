import os

import torch
from b_model import LinearRegression
#os.system('pip3 install torch --extra-index-url https://download.pytorch.org/whl/cpu')
import pandas as pd
import numpy as np
import streamlit as st
from data_manager import DataManager
from model.transformer import Transformer
from trainer import TorchTrainer
import plotly.graph_objects as go

# Title
if 'city_selected' in st.session_state:
    st.title(f'WeathFormer ({st.session_state.city_selected})')
else:
    st.title('WeatherFormer')
st.text('A transformer based weather prediction application')


dm = DataManager()
model = Transformer(inpt_features=1,
                    d_model=64,
                    nhead=8,
                    d_hid=64,
                    nlayers=3).load_state_dict(torch.load(os.curdir + '/' + 'model_state.pt'))

regressionModel = LinearRegression(24)

handler = TorchTrainer(model,
                        batch_size=1,
                        bptt=24,
                        alpha=0.0001,
                        num_epochs=1)


#regressionModel.load_state_dict(torch.load('baseline_state.pt'))


def getDisplayCities():
   res = []
   for city in dm.cities:
      city = city.split('_')
      res.append(' '.join([c.capitalize() for c in city]))
   return sorted(res)

with st.sidebar:
    city_selection = getDisplayCities()
    city_selected = st.selectbox('City', city_selection, key='city_selected')
    full_df = dm.getFullDataset(city_selected)
    df = full_df.tail(30*24)

    rained = len(full_df[full_df['precipMM'] > 0])
    all = len(full_df)
    labelsValues = ['Yes', 'No']
    fig = go.Figure(
        data=[go.Pie(labels=labelsValues, values=[rained / all, (all - rained) / all], textinfo='label+percent',
                     insidetextorientation='radial', showlegend=False,

                     )])
    fig.update_layout(title='Precipitation (Probability)', width=400, height=400)

    st.plotly_chart(fig)


tab1, tab2, tab3 = st.tabs(["Historical", "Inference", 'Descriptive Statistics'])
with tab1:
    if 'city_selected' not in st.session_state:
        st.session_state['city_selected'] = 'chicago'
    X = dm.getHistorical(df)
    yhat = handler.predict_historical(X)
    yhat_baseline = regressionModel.predict_historical(X)

    y = dm.toF(df['tempC'].tolist()[1:])
    yhat = dm.toF(yhat)
    yhat_baseline = dm.toF(yhat_baseline)


    xaxis = list(range(0, len(y)))


    col1, col2, col3 = st.columns(3)
    with col1:
        label = 'Low (F)'
        value = int(min(y))
        st.metric(label, value, delta=None, delta_color="normal", help=None)
    with col2:
        label = 'Average (F)'
        value = int(np.mean(y))
        st.metric(label, value, delta=None, delta_color="normal", help=None)

    with col3:
        label = 'High (F)'
        value = int(max(y))
        st.metric(label, value, delta=None, delta_color="normal", help=None)

    f1 = go.Figure(
       data=[
           go.Line(x=xaxis, y=y, name="actual", line={'dash': 'solid', 'color':'rgba(0, 0, 255, 0.9)'}),
           go.Line(x=xaxis, y=yhat, name="prediction", line={'dash': 'dash', 'color':'rgba(255, 0, 0, 0.7)'}),
           go.Line(x=xaxis, y=yhat_baseline, name='baseline')
       ],
       layout={"xaxis": {"title": "observations"}, "yaxis": {"title": "TEMPERATURE (F)"},
               "title": f"Historical ({city_selected})"}
    )

    st.plotly_chart(f1)


with tab2:
    if 'city_selected' not in st.session_state:
        st.session_state['city_selected'] = 'chicago'
    if 'days_ahead' not in st.session_state:
        days_ahead = 1

    #actual
    lastWeekActual = dm.getLastWeek(city_selected)

    #transformer
    lastWeekPrediction = handler.predLastWeek(lastWeekActual)
    if 'days_ahead' not in st.session_state:
        days_ahead = 1
        tomorrowPred = handler.predTomorrow(lastWeekActual, days=int(days_ahead))
    else:
        tomorrowPred = handler.predTomorrow(lastWeekActual, days=int(st.session_state.days_ahead))



    #regression
    lastWeekReg = regressionModel.predLastWeek(lastWeekActual)
    if 'days_ahead' not in st.session_state:
        days_ahead = 1
        tomorrowReg = regressionModel.predTomorrow(lastWeekActual, days=int(days_ahead))
    else:
        tomorrowReg = regressionModel.predTomorrow(lastWeekActual, days=int(st.session_state.days_ahead))



    # all to farenheit
    lastWeekActual = dm.toF(lastWeekActual)
    lastWeekPrediction = dm.toF(lastWeekPrediction)
    tomorrowPred = dm.toF(tomorrowPred)
    lastWeekReg = dm.toF(lastWeekReg)
    tomorrowReg = dm.toF(tomorrowReg)


    past = list(range(0, len(lastWeekActual)))
    future = list(range(len(lastWeekActual), len(lastWeekActual) + len(tomorrowPred)))

    col1, col2 = st.columns(2)
    with col1:
        label = 'Temperature (F)'
        value = int(lastWeekActual[-1])
        delta = None
        st.metric(label, value, delta=delta, delta_color="normal", help=None)

    with col2:
        label = 'Forecast (F)'
        value = int(tomorrowPred[-1].item())
        delta = f'{round(((tomorrowPred[-1].item() - lastWeekActual[-1]) / tomorrowPred[-1].item()) * 100, 2)} (%)'
        st.metric(label, value, delta=delta, delta_color="normal", help=None)

    f2 = go.Figure(
       data=[
           # actual
           go.Line(x=past, y=lastWeekActual, name="Last Week (Actual)", line={'dash': 'solid', 'color': 'blue'}),

           #transformer predictions
           go.Line(x=past, y=lastWeekPrediction, name="Last Week (Prediction)", line={'dash': 'dash', 'color': 'red'}),
           go.Line(x=future, y=tomorrowPred, name='Future Forecast', line={'dash': 'dash', 'color': 'pink'}),

           # linear regression
           go.Line(x=future, y=lastWeekReg, name='Last Week (Regression)', line={'dash': 'dash', 'color': 'yellow'}),
           go.Line(x=future, y=tomorrowReg, name='Future Forecast (Regression)', line={'dash': 'dash', 'color': 'yellow'})

       ],
       layout={"xaxis": {"title": "observations"}, "yaxis": {"title": "TEMPERATURE (F)"}, "title": f"Inference ({city_selected})"})
    f2.update_layout(legend=dict(
        yanchor="bottom",
        y=0.0,
        xanchor="center",
        x=1.0
    ))
    f2.update_layout(width=800)
    f2.update_layout(height=500)

    f2.update_layout(template='streamlit')

    days_ahead = st.select_slider(
        'Forecast Length (Days)',
        options=[1, 2, 3, 4, 5], key='days_ahead')

    st.plotly_chart(f2)


with tab3:
    if 'city_selected' not in st.session_state:
        st.session_state['city_selected'] = 'chicago'

    binContainer = st.container()
    with binContainer:
        labelsValues = list((range(-20, 110, 10)))
        df['tempF'] = df['tempC'] * 9 / 5 + 32
        rangeValues = full_df.groupby(pd.cut(full_df['tempF'], bins=labelsValues)).size()
        labelsValues = [str(label) + 's' for label in labelsValues[:-1]]

        fig = go.Figure([go.Bar(x=labelsValues, y=rangeValues.values)])
        fig.update_layout({'yaxis': {'title': 'Temperature'}})
        st.plotly_chart(fig)


    st.subheader("Distribution Statistics")
    desc_df = full_df[['tempC', 'DewPointC', 'precipMM', 'humidity', 'visibility', 'windspeedKmph']]
    desc_df.columns = ['Temperature (F)', 'Dew Point', 'Precipitation', 'Humidity', 'Visibility', 'Windspeed']
    st.dataframe(desc_df.describe())



    corrContainer = st.container()
    with corrContainer:
        st.subheader("Correlation Matrix")
        corr_df = df[['tempC', 'DewPointC', 'precipMM', 'humidity', 'visibility', 'windspeedKmph']]
        corr_df.columns = ['Temperature (F)', 'Dew Point', 'Precipitation', 'Humidity', 'Visibility', 'Windspeed']
        st.dataframe(corr_df.corr())







