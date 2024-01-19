import streamlit as st
from streamlit_option_menu import option_menu
from pathlib import Path


import pickle
import pandas as pd
import datetime as dt
from datetime import datetime
import pytz

import json
import requests
import geopandas as gpd

from urllib.request import urlopen

import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import folium as fs
from streamlit_folium import folium_static



# import datasets


# import spray dataset
spray_path = Path(__file__).parent / 'data/spray2.csv'
spray_df = pd.read_csv(spray_path)

# import combined train and weather (feature engineered) dataset for eda
final_path = Path(__file__).parent / 'data/final_eda.csv'
eda_df = pd.read_csv(final_path)

# import dataset for the model
final_path = Path(__file__).parent / 'data/streamlit_df.csv'
model_df = pd.read_csv(final_path)

# import chicago map data
chicago_path = Path(__file__).parent / './data/chicago-community-areas.geojson'
chicago = gpd.read_file(chicago_path)

with open(chicago_path) as f:
    chicago_geojson = json.load(f)


# load the model
model_path = Path(__file__).parent / 'models/model.pkl'
model = pickle.load(open(model_path, 'rb'))



# streamlit shell (layouts etc)
# set webpage name and icon
st.set_page_config(
    page_title='The West Nile Virus in Chicago',
    page_icon=':mosquito:',
    layout='wide',
    initial_sidebar_state='expanded'
    )

# top navigation bar
selected = option_menu(
    menu_title = None,
    options = ['Facts And Figures', 'WNV in Chicago','Risk In Your Area'],
    icons = ['eyeglasses','bar-chart','search'],
    default_index = 0, # which tab it should open when page is first loaded
    orientation = 'horizontal',
    styles={
        'nav-link-selected': {'background-color': '#FF7F0E'}
        }
    )

if selected == 'Facts And Figures':
    # title
    st.title('Facts And Figures')
    st.subheader('by Eden, Enoch, Sandra, and Wynne')
    style = "<div style='background-color:#FF7F0E; padding:2px'></div>"
    st.markdown(style, unsafe_allow_html = True)



    # comparative bar/line chart 
    # ask for user input
    st.subheader('Historic Data For Weather Patterns And Occurrence of the WNV')
    
    option = st.selectbox('Pick a variable to see the relationship with the presence of the WNV',
                            ('Precipitation', 'Average Temperature', 'Wind Direction', 
                             'Length of Night','Fog', 'Haze', 'Mist','Rain','Thunderstorms'))
    
    # translate the english from the option box into the equivalent variable
    if option == 'Precipitation':
        variable = 'roll_sum_28_PrecipTotal'
    elif option == 'Average Temperature':
        variable = 'roll_mean_28_Tavg'
    elif option == 'Wind Direction':
        variable = 'ResultDir'
    elif option == 'Length of Night':
        variable = 'night_length'
    elif option == 'Fog':
        variable = 'roll_sum_28_isFG'
    elif option == 'Haze':
        variable = 'roll_sum_28_isHZ'
    elif option == 'Mist':
        variable = 'roll_sum_28_isBR'
    elif option == 'Rain':
        variable = 'roll_sum_28_isRA'
    elif option == 'Thunderstorms':
        variable = 'roll_sum_28_isTS'
    

    # create the dataframe for a twinx plot    
    variable_WnvP_df = eda_df[['Month',variable,'WnvPresent']].groupby('Month').mean()
    variable_WnvP_df = variable_WnvP_df.reset_index(col_level=1)
    variable_WnvP_df = variable_WnvP_df.astype({'Month': 'string'})
    
    # draw the line graph for the chosen variable
    fig, ax1 = plt.subplots(figsize=(4,3))
    sns.lineplot(data=variable_WnvP_df,x='Month',y=variable,ax=ax1,
                 color='orange',sort=False).set(ylabel=variable)
    ax1_patch = mpatches.Patch(color='orange', label=variable)
    
    # draw the bar chart for WnvPresent
    ax2 = ax1.twinx()
    sns.barplot(data=variable_WnvP_df,x='Month',y='WnvPresent',color='paleturquoise',ax=ax2, 
                alpha=0.5).set(ylabel='WnvPresent')
    ax2.grid(False)
    ax2_patch = mpatches.Patch(color='paleturquoise', label='WnvPresent')
    
    ax1.legend(handles = [ax1_patch, ax2_patch], loc= 'upper right', bbox_to_anchor = (1.6,1), fontsize = 6)
    
    
    
    fontsize=8
    labelsize=5
    ax1.set_title(str(option) + ' vs WnvPresent', fontsize=fontsize)
    ax1.set_ylabel(str(option),fontsize=fontsize)
    ax2.set_ylabel('WnvPresent',fontsize=fontsize)
    ax1.set_xlabel('Month',fontsize=fontsize)
    ax1.tick_params(labelsize=labelsize)
    
    st.pyplot(fig)
    
    
    # 
     
    


    # Weather patterns in Chicago
    st.header('Patterns Observed')
    st.subheader('Noticeable correlation between humidity, heat, and WNV+ mosquitoes')
  
    # text explaination for  
    st.write('A consistent positive correlation is observed between temperature, humidity and the presence of WNV positive mosquitoes, indicating a higher prevalence of WnvPresent during periods of increased temperature. This is consistent with existing literature. According to Lebl et al. (2013) the observed relationship can be attributed to the temperature-dependent development rates of mosquito life stages, including eggs, larvae, pupae, and adult survival rates. Similarly, according to Drakou et al. (2020), Studies have indicated that high humidity contributes to increased egg production, larval indices, mosquito activity, and overall influences their behavior. Furthermore, research has indicated that an optimal range of humidity, typically between 44% and 69%, stimulates mosquito flight activity, with the most suitable conditions observed at around 65%.')

    st.text("") # add extra line in between
    
    st.header('Mosquito Species As WNV Vectors')
    
    
    # now to create a table for the mosquito information
    species_with_virus = eda_df.pivot_table(values=['NumMosquitos'], index='Species',
                                       columns='WnvPresent', aggfunc='sum')

    # Calculate the overall total number of mosquitos across all species
    overall_total = species_with_virus['NumMosquitos'].sum().sum()
    
    # Calculate the overall total number of mosquitos with virus across all species
    overall_virus = species_with_virus['NumMosquitos',1].sum()
    
    # Create a new column for the percentage
    species_with_virus['Percentage_of_overall'] = species_with_virus[('NumMosquitos', 1)] / overall_total * 100
    species_with_virus['Percentage_of_virus'] = species_with_virus[('NumMosquitos', 1)] / overall_virus * 100
    
    cols = ['orange','paleturquoise']
    
    moz = species_with_virus.plot(kind='barh', stacked=True, figsize=(12,5), color = cols).figure
    plt.title("Number of Mosquitos with and without Virus")
    plt.xlabel("Number of Mosquitos")
    plt.legend(labels=["Without Virus", "With Virus"]);
    
    # visualise the mosquito species bar chart
    st.pyplot(moz)
    

    moz2 = plt.figure(figsize = (12,5))
    sns.barplot(data=eda_df,x='WnvPresent',y='Species', color = 'paleturquoise')
    plt.title("Probability of WNV Being Present By Species")
    
    st.pyplot(moz2)
    
    st.subheader('Culex Pipiens is the most likely vector for WNV')
    st.write("Based on the aforementioned observations, it can be inferred that the mosquito species Culex Pipiens and Culex Restuans are carriers of the West Nile virus, while the remaining species caught do not pose a risk.")
    st.write('Spraying efforts should therefore focus more on areas that have higher incidence of Culex Pipiens and Culex Restuans.')


if selected == 'WNV in Chicago':
    # title
    st.title('Visualisation of the WNV in Chicago')
    st.subheader('by Eden, Enoch, Sandra, and Wynne')
    style = "<div style='background-color:#FF7F0E; padding:2px'></div>"
    st.markdown(style, unsafe_allow_html = True)

    st.header('Chicago')
    
    # explain the animation
    st.write('Pick a date on the slider under the map to see the number of WNV positive mosquitoes, or press play to see the changing values over time.')
    
    # time to make the mosquito dataframe for mapping
    
    eda_df['Date'] = pd.to_datetime(eda_df['Date'])
    eda_df['Year-Month'] = eda_df['Date'].dt.strftime('%Y %m')
    eda_df['Year-Month'] = pd.to_datetime(eda_df['Year-Month'], format='%Y %m').dt.to_period('M')
    
    # Calculate total 'NumMosquitos'
    total_mosquito = eda_df.groupby(['Address','Year-Month'], as_index=False)['NumMosquitos'].sum()
    total_mosquito.sort_values(by='Year-Month', inplace=True)
    
    # Calculate median 'latitude' and 'longitude' for each address
    areas = eda_df.groupby('Address', as_index=False)[['Latitude', 'Longitude']].median()
    
    # Calculate total number of 'WnvPresent'
    virus = eda_df.groupby('Address', as_index=False)['WnvPresent'].sum()
    
    # merge datasets together
    mos_data = pd.merge(total_mosquito, areas, on='Address')
    mos_data = pd.merge(mos_data, virus, on='Address')
    
    # since we no longer need 'Address', drop col
    mos_data.drop('Address', axis = 1, inplace = True)
    
    # sort by 'Year-Month'
    mos_data.sort_values(by='Year-Month', inplace=True)
    
    
    # Convert dataframe to geodataframe
    mos_geo = gpd.GeoDataFrame(mos_data, geometry= gpd.points_from_xy(mos_data.Longitude, mos_data.Latitude))
    
    # Output with community areas added to the mosquito dataframe
    mos_chicago = gpd.sjoin(mos_geo, chicago, op='within')
    
    # Summary with some actionable content
    # community_infections is the df we will use for display because it is the cleanest summary
    # it will be called later with an option to choose how many rows you want to see
    community_infections = mos_chicago[['community','NumMosquitos', 'WnvPresent']].groupby('community').sum()
    community_infections.sort_values('WnvPresent', inplace = True, ascending = False)
    chicago_ltd = chicago[['community', 'geometry']]
    community_infections2 = chicago_ltd.merge(community_infections, on='community')
    community_infections2.sort_values('WnvPresent', inplace = True, ascending = False)
    community_infections2.reset_index(inplace = True)
    
    
    # Now for the actual animated map
    
    # Set the Mapbox access token
    px.set_mapbox_access_token(st.secrets['mapbox_token'])
    
    
    # Create a scatter mapbox
    fig_m = px.scatter_mapbox(mos_chicago, 
                            lat=mos_chicago.geometry.y, 
                            lon=mos_chicago.geometry.x,
                            color='NumMosquitos', size='WnvPresent',
                            color_continuous_scale=px.colors.sequential.Jet,
                            hover_data=['NumMosquitos', 'WnvPresent', 'community'], zoom=9, animation_frame='Year-Month')
    
    
    # Create a layer for the community area boundaries
    layer_chicago = dict(
        sourcetype = 'geojson',
        source = chicago_geojson,
        type='fill',
        color='hsla(0, 100%, 90%, 0.2)',  
        below='traces',
        )
    
    # Add the community area boundaries layer to the scatter map
    fig_m.update_layout(mapbox_layers=[layer_chicago])
    
    # Update the layout
    fig_m.update_layout(mapbox_style= 'stamen-toner',
                      title='WNV+ vs. Mosquito count',
                      autosize=False,
                      width=1200,
                      height=1200,
                      )
    
    # Display the figure
    st.plotly_chart(fig_m)
    
    
    

    
    # Request input for how many rows to show
    header_n = st.slider('Select the top number of communities affected by positive WNV mosquitoes', 1, 20, 5)
    
    st.write('Note: Numbers shown below are cumulative. There are a total of 61 communities in our data set.')
    
    # Display the top n communities in table form
    show_communities = community_infections.head(header_n)
    st.dataframe(show_communities)
    
    # Display the top n communities in map form
    show_communities2 = community_infections2.head(header_n)
    
    m2 = fs.Map(location=[41.881832, -87.623177],tiles = 'Stamen Terrain', zoom_start=10, scrollWheelZoom=False)
    m2.choropleth(geo_data = chicago_geojson, 
                    data = show_communities2,
                    columns = ['community', 'WnvPresent'],
                    key_on = 'feature.properties.community',
                    fill_color = 'YlOrRd', 
                    fill_opacity = 0.7, 
                    line_opacity = 0.2,
                    legend_name = 'Number of WNV Positive Mosquitos per Neighbourhood in the years 2007, 2009, 2011, 2013')
        
    for lat, lon, community, wnv in zip(community_infections2.geometry.centroid.y, community_infections2.geometry.centroid.x , community_infections2.community, community_infections2.WnvPresent):
        fs.CircleMarker(location=[f'{lat}',f'{lon}'], radius = 1, color = 'green', fill = True, tooltip=f'{community}, Total Number of WNV Positive Mosquitoes: {wnv}').add_to(m2)    
    
    st_map = folium_static(m2, width=1200)


if selected == 'Risk In Your Area':
    # title
    st.title('Predicted Risk of WNV in Chicago')
    st.subheader('by Eden, Enoch, Sandra, and Wynne')
    style = "<div style='background-color:#FF7F0E; padding:2px'></div>"
    st.markdown(style, unsafe_allow_html = True)
    
    st.header('Risk in your area')
    
    
    # dictionaries that we will be using
    species_map = {'Culex Pipiens': 0.088922,
                       'Culex Pipiens/Restuans': 0.055135,
                       'Culex Restuans': 0.017883}
    trap_map = {'T001': 0.0,
                     'T002': 0.0972972972972973,
                     'T003': 0.11475409836065574,
                     'T004': 0.0,
                     'T005': 0.14285714285714285,
                     'T006': 0.16666666666666666,
                     'T007': 0.0,
                     'T008': 0.07194244604316546,
                     'T009': 0.08108108108108109,
                     'T011': 0.08270676691729323,
                     'T012': 0.0380952380952381,
                     'T013': 0.09615384615384616,
                     'T014': 0.13333333333333333,
                     'T015': 0.11428571428571428,
                     'T016': 0.10606060606060606,
                     'T017': 0.0,
                     'T018': 0.0,
                     'T019': 0.0,
                     'T025': 0.0,
                     'T027': 0.07526881720430108,
                     'T028': 0.07692307692307693,
                     'T030': 0.06349206349206349,
                     'T031': 0.03424657534246575,
                     'T033': 0.05154639175257732,
                     'T034': 0.0,
                     'T035': 0.041666666666666664,
                     'T036': 0.041666666666666664,
                     'T037': 0.034482758620689655,
                     'T039': 0.05,
                     'T040': 0.0,
                     'T043': 0.0,
                     'T044': 0.0,
                     'T045': 0.08571428571428572,
                     'T046': 0.0,
                     'T047': 0.02631578947368421,
                     'T048': 0.0273972602739726,
                     'T049': 0.011764705882352941,
                     'T050': 0.0,
                     'T051': 0.0,
                     'T054': 0.03680981595092025,
                     'T054C': 0.1111111111111111,
                     'T060': 0.0,
                     'T061': 0.07766990291262135,
                     'T062': 0.012195121951219513,
                     'T063': 0.019230769230769232,
                     'T065': 0.045871559633027525,
                     'T066': 0.05172413793103448,
                     'T067': 0.03333333333333333,
                     'T069': 0.012048192771084338,
                     'T070': 0.1,
                     'T071': 0.0,
                     'T072': 0.0,
                     'T073': 0.03125,
                     'T074': 0.017391304347826087,
                     'T075': 0.0,
                     'T076': 0.0,
                     'T077': 0.030303030303030304,
                     'T078': 0.0,
                     'T079': 0.012048192771084338,
                     'T080': 0.021739130434782608,
                     'T081': 0.06818181818181818,
                     'T082': 0.08163265306122448,
                     'T083': 0.02702702702702703,
                     'T084': 0.04,
                     'T085': 0.030303030303030304,
                     'T086': 0.09302325581395349,
                     'T088': 0.0,
                     'T089': 0.04395604395604396,
                     'T090': 0.046357615894039736,
                     'T091': 0.03125,
                     'T092': 0.0,
                     'T094': 0.03305785123966942,
                     'T094B': 0.0,
                     'T095': 0.05555555555555555,
                     'T096': 0.12,
                     'T097': 0.058823529411764705,
                     'T099': 0.0,
                     'T100': 0.0,
                     'T102': 0.017857142857142856,
                     'T103': 0.07228915662650602,
                     'T107': 0.09090909090909091,
                     'T114': 0.060810810810810814,
                     'T115': 0.07564575645756458,
                     'T128': 0.06875,
                     'T129': 0.0,
                     'T135': 0.04371584699453552,
                     'T138': 0.050955414012738856,
                     'T141': 0.0,
                     'T142': 0.05555555555555555,
                     'T143': 0.1935483870967742,
                     'T144': 0.02702702702702703,
                     'T145': 0.01098901098901099,
                     'T146': 0.0,
                     'T147': 0.04081632653061224,
                     'T148': 0.0,
                     'T149': 0.0,
                     'T150': 0.0,
                     'T151': 0.038461538461538464,
                     'T152': 0.021505376344086023,
                     'T153': 0.0,
                     'T154': 0.09523809523809523,
                     'T155': 0.06976744186046512,
                     'T156': 0.047619047619047616,
                     'T157': 0.0,
                     'T158': 0.04081632653061224,
                     'T159': 0.036036036036036036,
                     'T160': 0.04395604395604396,
                     'T161': 0.0,
                     'T162': 0.047619047619047616,
                     'T200': 0.007751937984496124,
                     'T206': 0.0,
                     'T209': 0.023076923076923078,
                     'T212': 0.019736842105263157,
                     'T215': 0.06666666666666667,
                     'T218': 0.02702702702702703,
                     'T219': 0.0,
                     'T220': 0.04081632653061224,
                     'T221': 0.08035714285714286,
                     'T222': 0.016666666666666666,
                     'T223': 0.10344827586206896,
                     'T224': 0.018518518518518517,
                     'T225': 0.10679611650485436,
                     'T226': 0.05194805194805195,
                     'T227': 0.05128205128205128,
                     'T228': 0.0967741935483871,
                     'T229': 0.0,
                     'T230': 0.1076923076923077,
                     'T231': 0.09523809523809523,
                     'T232': 0.04,
                     'T233': 0.14,
                     'T235': 0.11290322580645161,
                     'T236': 0.030303030303030304,
                     'T237': 0.0,
                     'T238': 0.0,
                     'T900': 0.088,
                     'T903': 0.07142857142857142}

    
    def update_df(df, features_to_fill, updated_features):
        for i, feature in enumerate(features_to_fill):
            df[feature] = updated_features[i]
        return df

    def model_predict(df, model=model):
        df2 = df.copy()
        df2['Trap'] = df2['Trap'].map(trap_map)
        df2.drop(['AddressNumberAndStreet','Latitude', 'Longitude'], axis=1, inplace=True)
        x = model.predict_proba(df2)[:,1]
        df['WnvProbability'] = x
        return df
    
    # disclaimer for non-american audiences; however given the subject matter
    # it is assumed almost everyone using this would be american/based there anyway
    st.markdown('NB: American DateTime format and units have been used.')
    st.markdown('Source: All weather information from weather.gov and sunrisesunset.io')


    # Get the current time in Chicago's timezone
    chicago_timezone = pytz.timezone('America/Chicago')
    current_time = datetime.now(chicago_timezone)

    # Format the date as a string with the desired format
    formatted_date = current_time.strftime("%m-%d-%Y")

    # Format the time as a string with the desired format
    formatted_time = current_time.strftime("%H:%M:%S")

    # Style the date display using HTML/CSS
    date_style = f"""
        <div style="font-size: 50px; font-family: Helvetica, sans-serif; color: #FF7F0E;">
            {formatted_date}
        </div>
    """

    # ditto with the time
    time_style = f"""
        <div style="font-size: 50px; font-family: Helvetica, sans-serif; color: #FF7F0E;">
            {formatted_time}
        </div>
    """

    date_col, space_col, time_col = st.columns(3)

    # display the formatted time and date
    with date_col:
        st.subheader('Current Date')
        st.markdown(date_style, unsafe_allow_html=True)

    with time_col:
        st.subheader('Current Time')
        st.markdown(time_style, unsafe_allow_html=True)

    st.markdown('')
    st.markdown('')

    col1, col2, col3 = st.columns(3)

    # retrieve weather
    # Request url
    forecast_url = 'https://api.weather.gov/points/41.8781,-87.6298'
    req = requests.get(forecast_url)

    # the weather.gov api is weird in that you have to call twice
    # retrieve forecast data
    call1_url = req.json()['properties']['forecast']

    # retrieve weather data for today
    forecast_json = requests.get(call1_url).json()

    # retrieve some values
    temp_now = forecast_json['properties']['periods'][0]['temperature']
    precip_prob_now = (forecast_json['properties']['periods'][0]['probabilityOfPrecipitation']['value'])
    windspeed_now = forecast_json['properties']['periods'][0]['windSpeed']

    # repeat the same display formatting as previously

    temp_style = f"""
        <div style="font-size: 36px; font-family: Helvetica, sans-serif; color: #FF7F0E;">
            {temp_now} °F
        </div>
    """

    precip_style = f"""
        <div style="font-size: 36px; font-family: Helvetica, sans-serif; color: #FF7F0E;">
            {precip_prob_now} %
        </div>
    """

    wind_style = f"""
        <div style="font-size: 36px; font-family: Helvetica, sans-serif; color: #FF7F0E;">
            {windspeed_now}
        </div>
    """


    with col1:
        st.subheader('Temperature')
        st.markdown(temp_style, unsafe_allow_html=True)
    with col2:
        st.subheader('Probability of Precipitation')
        st.markdown(precip_style, unsafe_allow_html=True)
    with col3:
        st.subheader('Wind Speed')
        st.markdown(wind_style, unsafe_allow_html=True)


    st.subheader('Forecast')
    st.write(forecast_json['properties']['periods'][0]['detailedForecast'])

    # now that we've done some decorative stuff, time for modelling

    # retrieve sunrise and sunset times today
    sunrise_url = 'https://api.sunrisesunset.io/json?lat=41.8781&lng=-87.6298'
    sunrise_req = requests.get(sunrise_url)
    sunrise_json = sunrise_req.json()

    # get sunrise, sunset, and day length in dt format
    date = sunrise_json['results']['date']
    sunrise_str = date + ' ' + sunrise_json['results']['sunrise']
    sunset_str = date + ' ' + sunrise_json['results']['sunset']
    sunrise_dt = datetime.strptime(sunrise_str, '%Y-%m-%d %I:%M:%S %p')
    sunset_dt = datetime.strptime(sunset_str, '%Y-%m-%d %I:%M:%S %p')

    daylength = sunrise_json['results']['day_length']
    daylength_dt = datetime.strptime(daylength, '%H:%M:%S')
    daylength_dt_seconds = daylength_dt.second + daylength_dt.minute*60 + daylength_dt.hour*3600

    # convert sunset and sunrise to pure numbers
    sunrise = sunrise_dt.hour * 100 + sunrise_dt.minute
    sunset = sunset_dt.hour * 100 + sunset_dt.minute

    # convert day length to decimal hours + rename for df
    timediff = 24 - daylength_dt_seconds/ 3600

    # time for the frustrating rolling values

    # first some time definitions

    one_week_ago = dt.timedelta(days = 7)
    three_weeks_ago = dt.timedelta(days = 21)
    four_weeks_ago = dt.timedelta(days = 28)

    today = datetime.now()
    roll_7_date = today - one_week_ago
    roll_21_date = today - three_weeks_ago
    roll_28_date = today - four_weeks_ago

    # NOAA access token
    token = st.secrets['NOAA_token']
    # # Midway airport weather station id
    # station_id = 'GHCND:USW00014819'
    # O'Hara airport weather station id
    station_id = 'GHCND:USW00094846'
    # NOAA api base request
    NOAA_url = 'https://www.ncdc.noaa.gov/cdo-web/api/v2/data?'
    # data set 
    datasetid = 'GHCND'
    # data to pull
    datatype_tavg = 'TAVG'
    datatype_precip = 'PRCP'
    datatype_tmin = 'TMIN'

    # end date will always be today
    enddate = datetime.strftime(today, '%Y-%m-%d')
    # start dates for the three lengths we need
    startdate = enddate
    startdate_7 = datetime.strftime(roll_7_date, '%Y-%m-%d')
    startdate_21 = datetime.strftime(roll_21_date, '%Y-%m-%d')
    startdate_28 = datetime.strftime(roll_28_date, '%Y-%m-%d')


    # different urls to call
    url_tavg_roll28 = NOAA_url + 'datasetid=' + datasetid + '&' + 'datatypeid=' + datatype_tavg + '&' + 'limit=1000' + '&' + 'stationid=' + station_id + '&' + 'startdate=' + startdate_28 + '&' + 'enddate=' + enddate
    url_tmin_roll7 = NOAA_url + 'datasetid=' + datasetid + '&' + 'datatypeid=' + datatype_tmin + '&' + 'limit=1000' + '&' + 'stationid=' + station_id + '&' + 'startdate=' + startdate_7 + '&' + 'enddate=' + enddate
    url_tmin_roll28 = NOAA_url + 'datasetid=' + datasetid + '&' + 'datatypeid=' + datatype_tmin + '&' + 'limit=1000' + '&' + 'stationid=' + station_id + '&' + 'startdate=' + startdate_28 + '&' + 'enddate=' + enddate
    url_precip_roll21 = NOAA_url + 'datasetid=' + datasetid + '&' + 'datatypeid=' + datatype_precip + '&' + 'limit=1000' + '&' + 'stationid=' + station_id + '&' + 'startdate=' + startdate_21 + '&' + 'enddate=' + enddate
    url_precip_roll28 = NOAA_url + 'datasetid=' + datasetid + '&' + 'datatypeid=' + datatype_precip + '&' + 'limit=1000' + '&' + 'stationid=' + station_id + '&' + 'startdate=' + startdate_28 + '&' + 'enddate=' + enddate

    # call 'em'
    req1 = requests.get(url_tavg_roll28, headers = {'token':token})
    req2 = requests.get(url_tmin_roll7, headers = {'token':token})
    req3 = requests.get(url_tmin_roll28, headers = {'token':token})
    req4 = requests.get(url_precip_roll21, headers = {'token':token})
    req5 = requests.get(url_precip_roll28, headers = {'token':token})

    # jsons
    tavg_json = req1.json()
    tmin7_json = req2.json()
    tmin28_json = req3.json()
    precip21_json = req4.json()
    precip28_json = req5.json()

    # tavg roll mean 28
    tavg_roll_28 = []
    for tavg in tavg_json['results']:
        tavg_roll_28.append(tavg['value'])
    roll_mean_28_Tavg = sum(tavg_roll_28)/len(tavg_roll_28)

    # tmin roll mean 7
    tmin_roll_7 = []
    for tmin in tmin7_json['results']:
        tmin_roll_7.append(tmin['value'])
    roll_mean_7_Tmin = sum(tmin_roll_7)/len(tmin_roll_7)

    # tmin roll mean 28
    tmin_roll_28 = []
    for tmin in tmin28_json['results']:
        tmin_roll_28.append(tmin['value'])
    roll_mean_28_Tmin = sum(tmin_roll_28)/len(tmin_roll_28)

    # precip roll sum 21
    precip_roll_21 = []
    for precip in precip21_json['results']:
        precip_roll_21.append(precip['value'])
    roll_sum_21_PrecipTotal = sum(precip_roll_21)

    # precip roll sum 28
    precip_roll_28 = []
    for precip in precip28_json['results']:
        precip_roll_28.append(precip['value'])
    roll_sum_28_PrecipTotal = sum(precip_roll_28)

    # now for the departure from daily mean.. this is maybe unnecessarily irritating lol
    # padded numbers are necessary!!!
    month = today.strftime('%m')
    day = today.strftime('%d')
    monthday = '-' + month + '-' + day

    tavg_dict = {}

    # check if there's even a record for today... if not, there's no point running the rest of this
    tavg_today_url = NOAA_url + 'datasetid=' + datasetid + '&datatypeid=' + datatype_tavg + '&limit=1000&stationid=' + station_id + '&startdate=' + str(today.year) + monthday + '&enddate=' + str(today.year) + monthday
    tavg_today_req = requests.get(tavg_today_url, headers={'token': token})
    tavg_today_json = tavg_today_req.json()

    if len(tavg_today_json) == 0:
        depart = 0

    else:
        tavg_today = tavg_today_json['results'][0]['value']
        
        # for each year from 1991-2020 because that's what the US is using for 30 year average right now
        for year in range(1991, 2021):
            year = str(year)

            # make the api call
            url = NOAA_url + 'datasetid=' + datasetid + '&datatypeid=' + datatype_tavg + '&limit=1000&stationid=' + station_id + '&startdate=' + year + monthday + '&enddate=' + year + monthday
            r = requests.get(url, headers={'token': token})
            if r.status_code != 200:
                break
            # load the api response as a json
            d = r.json()

            if len(d) == 0:
                continue

            # get all items in the response which are average temperature readings
            avg_temp = d['results'][0]['value']
            # get the date field from all average temperature readings
            tavg_dict[f'{year}'] = avg_temp
        
        today_30yr_tavg = sum(tavg_dict.values())/len(tavg_dict.values())
        depart = tavg_today - today_30yr_tavg


    ###---this is where we need info from people--###

    st.subheader('Fill in the following to see the risk of WNV in this time period.')
    species = st.selectbox('Species',['Culex Pipiens/Restuans', 'Culex Restuans', 'Culex Pipiens'], index=0)
    species = species_map[species]
    

    codesum = st.multiselect('CodeSum', ['Normal', 'BR', 'HZ', 'RA', 'TS', 'VCTS'])
    codesum = 0.042585423329405826 # Codesum score for normal  


    num_trap = st.number_input('Average number of times checked for each trap a day', min_value=0, step=1)
    roll_sum_14_num_trap = num_trap * 14
    speciesXroll_sum_28_num_trap =  species * (num_trap * 28)

    year = current_time.year


    features_to_fill = ['Species', 'Depart', 'Sunrise', 'Sunset', 'CodeSum',
                        'roll_sum_21_PrecipTotal', 'roll_sum_28_PrecipTotal',
                        'roll_mean_7_Tmin', 'roll_mean_28_Tmin', 'roll_mean_28_Tavg',
                        'Month', 'Year', 'num_trap', 'roll_sum_14_num_trap',
                        'speciesXroll_sum_28_num_trap', 'timediff']
    if num_trap is not None:
        updated_features = [species, depart, sunrise, sunset, codesum,
                            roll_sum_21_PrecipTotal, roll_sum_28_PrecipTotal,
                            roll_mean_7_Tmin, roll_mean_28_Tmin, roll_mean_28_Tavg,
                            month, year, num_trap, roll_sum_14_num_trap,
                            speciesXroll_sum_28_num_trap, timediff]
        update_df(model_df, features_to_fill, updated_features)
        
        model_predict(model_df)

    st.subheader('Map')
    fig = px.scatter_mapbox(model_df,
                            lat='Latitude',
                            lon='Longitude',
                            hover_name='Trap',
                            hover_data=['AddressNumberAndStreet', 'Latitude', 'Longitude', 'WnvProbability'],
                            size='WnvProbability',
                            color='WnvProbability',
                            color_continuous_scale=[[0, 'rgb(255, 200, 200)'], [1, 'rgb(255, 0, 0)']],
                            range_color=[0, 1],
                            zoom=10)
    # Create a layer for the community area boundaries
    layer_chicago = dict(
    sourcetype = 'geojson',
    source = chicago_geojson,
    type='fill',
    color='hsla(0, 100%, 90%, 0.2)',  
    below='traces',
    )


    fig.update_layout(mapbox_layers=[layer_chicago],
                      mapbox_style='carto-positron',
                      margin={'r': 0, 't': 0, 'l': 0, 'b': 0},
                      height=700,
                      width=1000)

    st.plotly_chart(fig)
    
    st.subheader('Areas with the highest probability of WNV')
    high_proba = model_df[model_df['WnvProbability'] == model_df['WnvProbability'].max()][['AddressNumberAndStreet', 'WnvProbability']]
    high_proba.reset_index(inplace = True, drop = True)
    st.dataframe(high_proba, width = 600, height = 400)

    st.subheader('Areas with the lowest probability of WNV')
    low_proba = model_df[model_df['WnvProbability'] == model_df['WnvProbability'].min()][['AddressNumberAndStreet', 'WnvProbability']]
    low_proba.reset_index(inplace = True, drop = True)
    st.dataframe(low_proba, width = 600, height = 400)
    
