import os
os.system('pip3 install torch --extra-index-url https://download.pytorch.org/whl/cpu')
import streamlit as st
import pandas as pd
import numpy as np
import streamlit as st
from data_utils.data_manager import DataManager
from model.transformer import Transformer
from trainer import TorchTrainer
import plotly.graph_objects as go
import plotly

# Title
st.title('WeathFormer')
st.text('A transformer based weather prediction application')


dm = DataManager()
model = Transformer(inpt_features=1,
                    d_model=64,
                    nhead=8,
                    d_hid=64,
                    nlayers=3)

handler = TorchTrainer(model,
                        batch_size=1,
                        bptt=24,
                        alpha=0.0001,
                        num_epochs=1)

#handler.load_model()


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

    pieContainer = st.container()
    with pieContainer:

        st.header(st.session_state.city_selected)


        labelsValues = list((range(-20, 110, 10)))
        df['tempF'] = df['tempC']*9/5 + 32
        rangeValues = full_df.groupby(pd.cut(full_df['tempF'], bins=labelsValues)).size()
        labelsValues = [str(label)+'s' for label in labelsValues[:-1]]

        fig = go.Figure([go.Bar(x=labelsValues, y=rangeValues.values)])
        fig.update_layout({'yaxis': {'title': 'Temperature'}})
        st.plotly_chart(fig)


    corrContainer = st.container()
    with corrContainer:
        st.subheader("Correlation Matrix")
        corr_df = df[['tempC', 'DewPointC', 'precipMM', 'humidity', 'visibility', 'windspeedKmph']]
        corr_df.columns = ['Temperature (F)', 'Dew Point', 'Precipitation', 'Humidity', 'Visibility', 'Windspeed']
        st.dataframe(corr_df.corr())



tab1, tab2 = st.tabs(["Historical", "Inference"])
with tab1:

    X = dm.getHistorical(df)
    yhat = handler.predict_historical(X)

    y = dm.toF(df['tempC'].tolist()[1:])
    yhat = dm.toF(yhat)
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
       ],
       layout={"xaxis": {"title": "observations"}, "yaxis": {"title": "TEMPERATURE (F)"},
               "title": f"Historical ({city_selected})"}
    )

    st.plotly_chart(f1)

    if 'days_ahead' not in st.session_state:
        st.session_state['days_ahead'] = 3
    with tab2:
        lastWeekActual = dm.getLastWeek(city_selected)
        lastWeekPrediction = handler.predLastWeek(lastWeekActual)
        tomorrowPred = handler.predTomorrow(lastWeekActual, days=int(st.session_state.days_ahead))

        lastWeekActual = dm.toF(lastWeekActual)
        lastWeekPrediction = dm.toF(lastWeekPrediction)
        tomorrowPred = dm.toF(tomorrowPred)




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
               go.Line(x=past, y=lastWeekActual, name="Last Week (Actual)", line={'dash': 'solid', 'color': 'blue'}),
               go.Line(x=past, y=lastWeekPrediction, name="Last Week (Prediction)", line={'dash': 'dash', 'color': 'red'}),
               go.Line(x=future, y=tomorrowPred, name='Future Forecast', line={'dash': 'dash', 'color': 'pink'})
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
        st.plotly_chart(f2)

        day_selection = st.select_slider(
            'Forecast Length (Days)',
            options=[1, 2, 3, 4, 5], key='days_ahead')
        st.write(f'{day_selection} day ({int(day_selection) *24})')

left3, right3 = st.columns(2)
with left3:
    st.subheader("Distribution Statistics")
    desc_df = full_df[['tempC', 'DewPointC', 'precipMM', 'humidity', 'visibility', 'windspeedKmph']]
    desc_df.columns = ['Temperature (F)', 'Dew Point', 'Precipitation', 'Humidity', 'Visibility', 'Windspeed']
    st.dataframe(desc_df.describe())

with right3:
    st.subheader('Precipitation Probability')
    rained = len(full_df[full_df['precipMM']>0])
    all = len(full_df)
    labelsValues = ['Yes', 'No']
    fig = go.Figure(data=[go.Pie(labels=labelsValues, values=rangeValues, textinfo='label+percent',
                                 insidetextorientation='radial', showlegend=False,

                                 )])
    fig.update_layout(width=400, height=400)

    st.plotly_chart(fig)





