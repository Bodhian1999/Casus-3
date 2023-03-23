# In[]:
#Packages importeren
import requests
import pandas as pd
import folium
import streamlit as st
from streamlit_folium import st_folium
from PIL import Image
import calendar
import time
from datetime import datetime as dt
from datetime import date
import plotly.express as px
import plotly.figure_factory as ff
import seaborn as sns
from statsmodels.formula.api import ols
from matplotlib import style
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np 
import urllib.request


# In[]:
HeaderImage = Image.open('Header.png')
st.image(HeaderImage, width=290)

st.header('Groep 2 Casus-3')

st.markdown('Voor onze 3e casus hebben wij, Bodhian, Jasmijn, Yvette en Rojhat, de opdracht gekregen om van data informatie te maken. Deze data hebben wij vooraf gecleaned en daarna toegepast.')

# In[]:
st.header('OpenChargeMap API')

###Inladen API - kijk naar country code en maxresults

st.markdown('**De API van OpenChargeMap inladen met behulp van een key**')
st.code('response = requests.get("https://api.openchargemap.io/v3/poi/?output=json&countrycode=NL&maxresults=15000&key=93b912b5-9d70-4b1f-960b-fb80a4c9c017")')
response = requests.get("https://api.openchargemap.io/v3/poi/?output=json&countrycode=NL&maxresults=15000&key=93b912b5-9d70-4b1f-960b-fb80a4c9c017")

###Omzetten naar dictionary
st.markdown('**Omzetten naar een dictionary**')
st.code('responsejson  = response.json()')
responsejson  = response.json()

###Dataframe bevat kolom die een list zijn. 
#Met json_normalize zet je de eerste kolom om naar losse kolommen
st.markdown('**Eerste kolom omzetten naar losse kolommen met behulp van json_normalize**')
st.code('Laadpalen = pd.json_normalize(responsejson)')
Laadpalen = pd.json_normalize(responsejson)

st.markdown('**Omzetten daar een pandas dataframe**')
st.code('OpenChargeMap = pd.DataFrame(Laadpalen)')
OpenChargeMap = pd.DataFrame(Laadpalen)


OpenChargeMap
#Hier wordt de row met de status type 0 gedropt, deze lader heeft geen werking. Deze nemen wij daarom niet mee
st.markdown('**Laders met een StatusTypeID 0, deze hebben geen werking volgens OpenChargeMap**')
st.write(OpenChargeMap[OpenChargeMap['StatusTypeID'] == 0])

st.markdown('**Hier wordt de row met de status type 0 gedropt, deze lader heeft geen werking. Deze nemen wij daarom niet mee.**')
st.code("OpenChargeMap.drop(OpenChargeMap[OpenChargeMap['StatusTypeID'] == 0].index, inplace=True)")
OpenChargeMap.drop(OpenChargeMap[OpenChargeMap['StatusTypeID'] == 0].index, inplace=True)

st.markdown('**De columns namen vermakkelijken**')
st.code("OpenChargeMap.rename(columns = {'AddressInfo.Latitude':'Latitude', 'AddressInfo.Longitude':'Longitude', 'AddressInfo.StateOrProvince': 'Provincie'}, inplace = True)")
OpenChargeMap.rename(columns = {'AddressInfo.Latitude':'Latitude', 'AddressInfo.Longitude':'Longitude', 'AddressInfo.StateOrProvince': 'Provincie'}, inplace = True)

st.markdown('**NaN-values uit provincie halen om uiteindelijk een map aan te kunnen tonen op basis van provincie**')
st.code("Laadpalen_geen_na_provincie=OpenChargeMap.dropna(subset = ['Provincie'])")
Laadpalen_geen_na_provincie=OpenChargeMap.dropna(subset = ['Provincie'])

st.markdown('**Laadpalen selecteren met provincie informatie bijgevoegd**')
st.code("Laadpalen_alleen_provincie=Laadpalen_geen_na_provincie[Laadpalen_geen_na_provincie.Provincie.isin(li)]")
st.code("li = ['Zuid-Holland', 'Noord-Holland', 'Flevoland', 'Zeeland', 'Noord-Brabant', 'Friesland', 'Groningen', 'Drenthe', 'Overijssel', 'Utrecht', 'Gelderland', 'Limburg']")
li = ['Zuid-Holland', 'Noord-Holland', 'Flevoland', 'Zeeland', 'Noord-Brabant', 'Friesland', 'Groningen', 'Drenthe', 'Overijssel', 'Utrecht', 'Gelderland', 'Limburg']
 
Laadpalen_alleen_provincie=Laadpalen_geen_na_provincie[Laadpalen_geen_na_provincie.Provincie.isin(li)]

# In[]:
st.header('Laad-data')

st.markdown('Deze data hebben wij ontvangen via een CSV-bestand in DLO (brightspace).')

laadpaaldata=pd.read_csv('laadpaaldata.csv')

st.code('pd.read_csv("laadpaaldata.csv")')

st.markdown('**De dataframe ziet er als volgt uit:**')
st.write(laadpaaldata)

st.markdown('**De verschillende dtypes in de dataframe**')
st.write(laadpaaldata.dtypes)

st.markdown('**Foutieve datums ontvangen een NaN-value**')
st.code("laadpaaldata['Started']= pd.to_datetime(laadpaaldata['Started'],errors='coerce')")
st.code("laadpaaldata['Ended']= pd.to_datetime(laadpaaldata['Ended'],errors='coerce')")

laadpaaldata['Started']= pd.to_datetime(laadpaaldata['Started'],errors='coerce')
laadpaaldata['Ended']= pd.to_datetime(laadpaaldata['Ended'],errors='coerce')

st.markdown('**Aantal NaN-values:**')
st.write(laadpaaldata.isna().sum())

st.markdown('**Missing values verwijderen (inplace=True)**')
st.code('laadpaaldata.dropna(inplace=True)')
laadpaaldata.dropna(inplace=True)

st.markdown('**.head functie:**')
st.write(laadpaaldata.head())

st.markdown('**.describe functie:**')
st.write(laadpaaldata.describe())

laadpaaldata = laadpaaldata[laadpaaldata['ChargeTime'] >= 0]

st.markdown('**Alleen de data met een Chargetime boven 0 selecteren**')
st.code("laadpaaldata = laadpaaldata[laadpaaldata['ChargeTime'] >= 0]")

# In[]:
st.header('RDW-data')

st.markdown('**RDW data inladen met behulp van de API. Limit op 200.000 auto.**')
st.code('response2 = requests.get("https://opendata.rdw.nl/resource/qyrd-w56j.json?$limit=2000&$offset=0")')
response2 = requests.get("https://opendata.rdw.nl/resource/qyrd-w56j.json?$limit=2000&$offset=0")

st.markdown('**Omzetten naar json.**')
st.code('responsejson2  = response2.json()')
responsejson2  = response2.json()

st.markdown('**Verspreiden over meerdere kolommen.**')
st.code('rdw = pd.json_normalize(responsejson2)')
rdw = pd.json_normalize(responsejson2)


st.markdown('**RDW brandstof-data van auto inladen met behulp van de API.**')
st.code("response3 = requests.get('https://opendata.rdw.nl/resource/8ys7-d773.json?$limit=200000&$offset=0')")
response3 = requests.get('https://opendata.rdw.nl/resource/8ys7-d773.json?$limit=200000&$offset=0')

st.markdown('**Omzetten naar json.**')
st.code('responsejson3  = response3.json()')
responsejson3  = response3.json()

st.markdown('**Pandas dataframe van de brandstof-data maken.**')
st.code('kenteken_brandstof=pd.json_normalize(responsejson3)')
kenteken_brandstof=pd.json_normalize(responsejson3)

st.markdown('**Enkel kenteken en brandstof omschrijving selecteren.**')
st.code('kenteken_brandstof2=kenteken_brandstof[["kenteken", "brandstof_omschrijving"]]')
kenteken_brandstof2=kenteken_brandstof[["kenteken", "brandstof_omschrijving"]]

st.markdown('**De brandstof data op kenteken samenvoegen.**')
st.code("df_merged=rdw.merge(kenteken_brandstof2, how='left', on='kenteken')")
df_merged=rdw.merge(kenteken_brandstof2, how='left', on='kenteken')

df2=df_merged[['merk','voertuigsoort', 'brandstof_omschrijving','datum_tenaamstelling']]
st.markdown('**Samengevoegde dataframe:**')
st.write(df2.head())

st.markdown('**Aantal NaN-values:**')
st.write(df2.isna().sum())

st.markdown('**De NaN-values verwijderen:**')
st.code('df2.dropna(inplace=True)')
df2.dropna(inplace=True)

df2['datum_tenaamstelling'] = pd.to_datetime(df2['datum_tenaamstelling'].astype(str), format='%Y%m%d')

df2['datum_tenaamstelling'] = df2['datum_tenaamstelling'].dt.strftime('%Y-%m')

df2 = df2.sort_values('datum_tenaamstelling')

df2['value'] = 1

st.write(pd.DataFrame(df2.brandstof_omschrijving.value_counts()))

df2['Total Count'] = df2.groupby('brandstof_omschrijving').value.cumsum()


fig2 = px.line(df2, y='Total Count', x='datum_tenaamstelling', color = 'brandstof_omschrijving', log_y=True)

fig2.update_layout(width=970,height=620,
    xaxis=dict(rangeslider=dict(visible=True), title='Datum Tenaamstelling'),
    title="Hoeveelheid Auto's per Brandstof Type",
    title_x= 0.35,
    yaxis=dict(
    title="Hoeveelheid Auto's"),
    legend=dict(
    title="Brandstof Type"))

# In[]:
st.header('Grafieken')

#  ik heb een functie gevonden op het internet voor het toevoegen van een categorische legenda:
# (bron: https://stackoverflow.com/questions/65042654/how-to-add-categorical-legend-to-python-folium-map)

def add_categorical_legend(folium_map, title, colors, labels):
    if len(colors) != len(labels):
        raise ValueError("colors and labels must have the same length.")

    color_by_label = dict(zip(labels, colors))
    
    legend_categories = ""     
    for label, color in color_by_label.items():
        legend_categories += f"<li><span style='background:{color}'></span>{label}</li>"
        
    legend_html = f"""
    <div id='maplegend' class='maplegend'>
      <div class='legend-title'>{title}</div>
      <div class='legend-scale'>
        <ul class='legend-labels'>
        {legend_categories}
        </ul>
      </div>
    </div>
    """
    script = f"""
        <script type="text/javascript">
        var oneTimeExecution = (function() {{
                    var executed = false;
                    return function() {{
                        if (!executed) {{
                             var checkExist = setInterval(function() {{
                                       if ((document.getElementsByClassName('leaflet-top leaflet-right').length) || (!executed)) {{
                                          document.getElementsByClassName('leaflet-top leaflet-right')[0].style.display = "flex"
                                          document.getElementsByClassName('leaflet-top leaflet-right')[0].style.flexDirection = "column"
                                          document.getElementsByClassName('leaflet-top leaflet-right')[0].innerHTML += `{legend_html}`;
                                          clearInterval(checkExist);
                                          executed = true;
                                       }}
                                    }}, 100);
                        }}
                    }};
                }})();
        oneTimeExecution()
        </script>
      """
   

    css = """

    <style type='text/css'>
      .maplegend {
        z-index:9999;
        float:right;
        background-color: rgba(255, 255, 255, 1);
        border-radius: 5px;
        border: 2px solid #bbb;
        padding: 10px;
        font-size:12px;
        positon: relative;
      }
      .maplegend .legend-title {
        text-align: left;
        margin-bottom: 5px;
        font-weight: bold;
        font-size: 90%;
        }
      .maplegend .legend-scale ul {
        margin: 0;
        margin-bottom: 5px;
        padding: 0;
        float: left;
        list-style: none;
        }
      .maplegend .legend-scale ul li {
        font-size: 80%;
        list-style: none;
        margin-left: 0;
        line-height: 18px;
        margin-bottom: 2px;
        }
      .maplegend ul.legend-labels li span {
        display: block;
        float: left;
        height: 16px;
        width: 30px;
        margin-right: 5px;
        margin-left: 0;
        border: 0px solid #ccc;
        }
      .maplegend .legend-source {
        font-size: 80%;
        color: #777;
        clear: both;
        }
      .maplegend a {
        color: #777;
        }
    </style>
    """

    folium_map.get_root().header.add_child(folium.Element(script + css))

    return folium_map


#Folium map laadpalen per provincie
map = folium.Map(location=[51.858566, 4.293632], zoom_start=7.4)

feature_groups = {}

types = Laadpalen_alleen_provincie['Provincie'].unique()

type_colors = {}


color_dict = {
    'Groningen': 'blue',
    'Friesland': 'green',
    'Drenthe': 'red',
    'Overijssel': 'purple',
    'Flevoland': 'orange',
    'Gelderland': 'darkred',
    'Utrecht': 'pink',
    'Noord-Holland': 'black',
    'Zuid-Holland': 'gray',
    'Zeeland': 'lightgreen',
    'Noord-Brabant': 'cadetblue',
    'Limburg': 'darkblue'
}



def color_producer(provincie):
    return color_dict[provincie]



for index, row in Laadpalen_alleen_provincie.iterrows():
    lat = row['Latitude']
    lng = row['Longitude']
    label = row['Provincie'] 
    color = color_producer(row['Provincie'])
    if label not in feature_groups:
        feature_groups[label] = folium.FeatureGroup(name=label)
    folium.CircleMarker(location=[lat, lng], popup=label, tooltip='<b>Klik hier om de popup te zien</b>', color=color, fill=True, fill_opacity=0.6, radius=7).add_to(feature_groups[label])

for label, feature_group in feature_groups.items():
    feature_group.add_to(map)

for i, t in enumerate(types):
    type_colors[t] = color_producer(t)

colors = [type_colors[t] for t in types]
labels = list(types)

map = add_categorical_legend(map, 'Legenda',
    colors = colors,
    labels = labels)


folium.TileLayer('CartoDB Positron', name='CartoDB').add_to(map) 
folium.LayerControl(position='bottomleft', collapsed=False).add_to(map)

st.markdown('**Kaart van elektrische opladers met een provincie waarde**')
st_data = st_folium(map)

# In[]:
st.markdown('**Kans op oplaadtijd**')

hist_data = [laadpaaldata['ChargeTime'],laadpaaldata['ConnectedTime']]
group_labels = ['Charge Time', 'Connected Time']
colors = ['#FF0000','#00B9FF']

fig = ff.create_distplot(hist_data, group_labels, colors=colors,show_rug=False, bin_size=.25)

fig.update_layout(
    title='Kans op Oplaadtijd',
    title_x= 0.35,
    xaxis=dict(
    title='Oplaadtijd in (uren)'),
    yaxis=dict(
    title='Kansdichtheid'),

    shapes=[
    {'line': {'color': '#FF5733', 'dash': 'dash', 'width': 2},
    'type': 'line',
    'x0': laadpaaldata.ChargeTime.mean(),
    'x1': laadpaaldata.ChargeTime.mean(),
    'xref': 'x',
    'y0': 0,
    'y1': 1,
    'yref': 'paper'},
    {'line': {'color': '#FFC300', 'dash': 'dash', 'width': 2},
    'type': 'line',
    'x0': laadpaaldata.ChargeTime.median(),
    'x1': laadpaaldata.ChargeTime.median(),
    'xref': 'x',
    'y0': 0,
    'y1': 1,
    'yref': 'paper'},
    {'line': {'color': '#00D1FF ', 'dash': 'dash', 'width': 2},
    'type': 'line',
    'x0': laadpaaldata.ConnectedTime.mean(),
    'x1': laadpaaldata.ConnectedTime.mean(),
    'xref': 'x',
    'y0': 0,
    'y1': 1,
    'yref': 'paper'},
    {'line': {'color': '#007CFF', 'dash': 'dash', 'width': 2},
    'type': 'line',
    'x0': laadpaaldata.ConnectedTime.median(),
    'x1': laadpaaldata.ConnectedTime.median(),
    'xref': 'x',
    'y0': 0,
    'y1': 1,
    'yref': 'paper'}],

    # Annotations
    annotations=[
        dict(
            align='left',
            x=0.86,
            y=0.75,
            xref='paper',
            yref='paper',
            text= "Mean Charge Time = {:,.2f}".format(laadpaaldata.ChargeTime.mean()),
            showarrow=True,
            arrowhead=7,
            ax=1,
            ay=1,
            font_color='#FF5733',
        ),
        dict(
            align='left',
            x=0.86,
            y=0.8,
            xref='paper',
            yref='paper',
            text="Median Charge Time = {:,.2f}".format(laadpaaldata.ChargeTime.median()),
            showarrow=True,
            arrowhead=7,
            ax=1,
            ay=1,
            font_color='#FFC300'
        ),
        dict(
            align='left',
            x=0.86,
            y=0.87,
            xref='paper',
            yref='paper',
            text= "Mean Connected Time = {:,.2f}".format(laadpaaldata.ConnectedTime.mean()),
            showarrow=True,
            arrowhead=7,
            ax=1,
            ay=1,
            font_color='#00D1FF ',
        ),
        dict(
            align='left',
            x=0.86,
            y=0.92,
            xref='paper',
            yref='paper',
            text="Median Connected Time = {:,.2f}".format(laadpaaldata.ConnectedTime.median()),
            showarrow=True,
            arrowhead=7,
            ax=1,
            ay=1,
            font_color='#007CFF'
        )
    ]
)

fig.update_layout(xaxis=dict(rangeslider=dict(visible=True)), autosize=False, width=970,height=620)

# In[95]:


df1 = pd.read_csv('car data.csv')
#df1.head()


# In[97]:


#df1.info()


# In[98]:


#df1.isnull().sum()


# In[99]:


#df1.describe()


# In[100]:


#df1.columns


# In[101]:


#print(df1['Fuel_Type'].value_counts())
#print(df1['Seller_Type'].value_counts())
#print(df1['Transmission'].value_counts())


# In[102]:


fuel_type = df1['Fuel_Type']
seller_type = df1['Seller_Type']
transmission_type = df1['Transmission']
selling_price = df1['Selling_Price']


# In[105]:


petrol_data = df1.groupby('Fuel_Type').get_group('Petrol')
petrol_data.describe()


# In[106]:


df1.replace({'Fuel_Type':{'Petrol':0, 'Diesel':1, 'CNG':2}}, inplace=True)


# In[107]:


df1 = pd.get_dummies(df1, columns=['Seller_Type', 'Transmission'],drop_first=True)


# In[110]:


X = df1.drop(['Car_Name','Selling_Price'], axis=1)
y = df1['Selling_Price']


# In[111]:


#print("Shape of X is: ",X.shape)
#print("Shape of y is: ", y.shape)


# In[112]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)


# In[114]:


scaler = StandardScaler()


# In[115]:


X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[116]:


model = LinearRegression()


# In[117]:


model.fit(X_train, y_train)


# In[113]:


#print("X_test shape:", X_test.shape)
#print("X_train shape:", X_train.shape)
#print("y_test shape: ", y_test.shape)
#print("y_train shape:", y_train.shape)


# In[118]:


pred = model.predict(X_test)


# In[120]:


#print("MAE: ", (metrics.mean_absolute_error(pred, y_test)))
#print("MSE: ", (metrics.mean_squared_error(pred, y_test)))
#print("R2 score: ", (metrics.r2_score(pred, y_test)))


# In[121]:


fig3 = sns.regplot(x=pred, y=y_test)
plt.xlabel("Predicted Price")
plt.ylabel('Actual Price')
plt.title("Actual vs predicted price")


st.plotly_chart(fig)

st.markdown("**Auto's per Brandstof Type**")

st.plotly_chart(fig2)

st.markdown("**Regressie Grafiek**")

st.pyplot(fig3)





