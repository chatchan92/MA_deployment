# import yfinance as yf
import streamlit as st
import datetime 
# import talib 
import ta
import pandas as pd
import numpy as np
import os, sys, gc, time, warnings, random, joblib, gzip, pickle
import itertools

import scipy as sc
import scipy.stats as sct
from scipy.stats import skewnorm, skew, gamma, norm
from scipy import stats


import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff

import lightgbm as lgb
from sklearn.model_selection import train_test_split

import shap

from ngboost import NGBRegressor
from ngboost.ngboost import NGBoost
from ngboost.distns import Bernoulli, Normal, LogNormal
from ngboost.scores import MLE
from ngboost.learners import default_tree_learner
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error

from sklearn.tree import DecisionTreeRegressor

################
LOD_STAGE = 1
################

# st.write("""
# BIM early desgin phase assistant system
# """)

st.set_page_config(
     page_title="BIM early desgin phase assistance system",
     page_icon="🧊",
     layout="wide",
     initial_sidebar_state="expanded",
 )

st.sidebar.header('User Input Parameters')

today = datetime.date.today()

# title
# h_col1, h_col2 = st.beta_columns([1,11])
# with h_col1:
#     st.image('digital_brain.png',width=35)
# with h_col2:
st.header("""
            **🧊 BIM early desgin phase assistance system**
            """)

option = st.selectbox(
    'Confirm your design stage (Level-of-Detail)',
    ('LOD1', 'LOD2', 'LOD3'))

st.write('You selected:', option)
LOD_STAGE = int(option[3])

if(LOD_STAGE == 1):

    TOTAL_LIST = [
        'BUILT_FORM',
        'TOTAL_FLOOR_AREA',
        'FLOOR_HEIGHT',
        'GLAZED_AREA',
        'NUMBER_HABITABLE_ROOMS',
        # 'WINDOWS_ENERGY_EFF',
        # 'WALLS_ENERGY_EFF',
        # 'ROOF_ENERGY_EFF',
        'WINDOWS_DESCRIPTION',
        'WALLS_DESCRIPTION',
        'ROOF_DESCRIPTION',

        # 'NUMBER_HEATED_ROOMS',
        # 'NUMBER_OPEN_FIREPLACES',
        # 'MULTI_GLAZE_PROPORTION',
        # 'MAINHEAT_DESCRIPTION',
        # 'MAINHEAT_ENERGY_EFF',

        # 'SECONDHEAT_DESCRIPTION',
        # 'MECHANICAL_VENTILATION',
        # 'MAINS_GAS_FLAG',
        # 'SOLAR_WATER_HEATING_FLAG',
        # 'HOT_WATER_ENERGY_EFF',
        # 'PHOTO_SUPPLY',
     ]

    cat_dict = {

        'BUILT_FORM': {'Detached': 0, 'Enclosed End-Terrace': 1, 'Enclosed Mid-Terrace': 2, 'End-Terrace': 3, 'Mid-Terrace': 4, 'Semi-Detached': 5}, 
        # 'MAINS_GAS_FLAG': {'N': 0, 'Y': 1}, 
        'GLAZED_AREA': {'Less Than Typical': 0, 'More Than Typical': 1, 'Much More Than Typical': 2, 'Normal': 3}, 
        # 'SECONDHEAT_DESCRIPTION': {'None': 0, 'Portable electric heaters': 1, 'Room heaters,': 2, 'Room heaters, LPG': 3, 'Room heaters, anthracite': 4, 'Room heaters, bottled LPG': 5, 'Room heaters, bottled gas': 6, 'Room heaters, coal': 7, 'Room heaters, dual fuel (mineral and wood)': 8, 'Room heaters, electric': 9, 'Room heaters, mains gas': 10, 'Room heaters, oil': 11, 'Room heaters, smokeless fuel': 12, 'Room heaters, wood logs': 13}
        # 'MAINHEAT_DESCRIPTION': {'Boiler and radiators, LPG': 0, 'Boiler and radiators, anthracite': 1, 'Boiler and radiators, bottled gas': 2, 'Boiler and radiators, coal': 3, 'Boiler and radiators, dual fuel (mineral and wood)': 4, 'Boiler and radiators, electric': 5, 'Boiler and radiators, mains gas': 6, 'Boiler and radiators, oil': 7, 'Boiler and radiators, smokeless fuel': 8, 'Boiler and underfloor heating, LPG': 9, 'Boiler and underfloor heating, mains gas': 10, 'Boiler and underfloor heating, oil': 11, 'Community scheme': 12, 'Community scheme with CHP': 13, 'Electric ceiling heating': 14, 'Electric storage heaters': 15, 'Electric underfloor heating': 16, 'No system present: electric heating assumed': 17, 'Portable electric heating assumed for most rooms': 18, 'Room heaters, electric': 19, 'Room heaters, mains gas': 20, 'Room heaters, wood logs': 21, 'Warm air, Electricaire': 22, 'Warm air, electric': 23, 'Warm air, mains gas': 24}
        # 'SOLAR_WATER_HEATING_FLAG': {'N': 0, 'Y': 1}, 
        # 'MECHANICAL_VENTILATION': {'mechanical, extract only': 0, 'mechanical, supply and extract': 1, 'natural': 2}, 
        
        # 'FLAT_TOP_STOREY': {'N': 0, 'Y': 1}, 
        # 'HOTWATER_DESCRIPTION': {'Electric immersion, dual tariff': 0, 'Electric immersion, off-peak': 1, 'Electric immersion, off-peak, no cylinder thermostat': 2, 'Electric immersion, off-peak, no cylinderstat': 3, 'Electric immersion, off-peak, plus solar': 4, 'Electric immersion, standard tariff': 5, 'Electric instantaneous at point of use': 6, 'From main system': 7, 'From main system, no cylinder thermostat': 8, 'From main system, no cylinder thermostat, plus solar': 9, 'From main system, no cylinderstat': 10, 'From main system, plus solar': 11, 'From main system, standard tariff': 12, 'From secondary system': 13, 'From secondary system, no cylinder thermostat': 14, 'From secondary system, no cylinderstat': 15, 'Gas instantaneous at point of use': 16, 'Gas multipoint': 17, 'No system present: electric immersion assumed': 18}, 
        'WINDOWS_DESCRIPTION': {'Fully double glazing': 0, 'Fully triple glazing': 1, 'Mostly double glazing': 2, 'Partial double glazing': 3, 'Single glazing': 4},
        'WALLS_DESCRIPTION': {'Cavity wall, as built, insulated': 0, 'Cavity wall, as built, no insulation': 1, 'Cavity wall, as built, partial insulation': 2, 'Cavity wall, filled cavity': 3, 'Cavity wall, with external insulation': 4, 'Cavity wall, with internal insulation': 5, 'Cob, as built': 6, 'Granite or whinstone, as built, insulated': 7, 'Granite or whinstone, as built, no insulation': 8, 'Granite or whinstone, with external insulation': 9, 'Granite or whinstone, with internal insulation': 10, 'Sandstone, as built, insulated': 11, 'Sandstone, as built, no insulation': 12, 'Sandstone, with internal insulation': 13, 'Solid brick, as built, insulated': 14, 'Solid brick, as built, no insulation': 15, 'Solid brick, as built, partial insulation': 16, 'Solid brick, with external insulation': 17, 'Solid brick, with internal insulation': 18, 'System built, as built, insulated': 19, 'System built, as built, no insulation': 20, 'System built, as built, partial insulation': 21, 'System built, with external insulation': 22, 'System built, with internal insulation': 23, 'Timber frame, as built, insulated': 24, 'Timber frame, as built, no insulation': 25, 'Timber frame, as built, partial insulation': 26, 'Timber frame, with internal insulation': 27},
        # 'FLOOR_DESCRIPTION': {'(other premises below)': 0, 'Solid, insulated': 1, 'Solid, insulated (assumed)': 2, 'Solid, limited insulation (assumed)': 3, 'Solid, no insulation (assumed)': 4, 'Suspended, insulated': 5, 'Suspended, insulated (assumed)': 6, 'Suspended, limited insulation (assumed)': 7, 'Suspended, no insulation (assumed)': 8, 'To external air, insulated': 9, 'To external air, insulated (assumed)': 10, 'To external air, limited insulation (assumed)': 11, 'To external air, no insulation (assumed)': 12, 'To external air, uninsulated (assumed)': 13, 'To unheated space, insulated': 14, 'To unheated space, insulated (assumed)': 15, 'To unheated space, limited insulation (assumed)': 16, 'To unheated space, no insulation (assumed)': 17, 'To unheated space, uninsulated (assumed)': 18}, 
        # 'MAINHEATCONT_DESCRIPTION': {'Appliance thermostats': 0, 'Automatic charge control': 1, 'Charging system linked to the use of community heating prog and TRVs': 2, 'Charging system linked to use of community heating programmer and TRVs': 3, 'Flat rate charging no thermostat control of room temperature': 4, 'Flat rate charging no thermostatic control': 5, 'Flat rate charging no thermostatic control of room temperature': 6, 'Flat rate charging programmer and TRVs': 7, 'Flat rate charging programmer and room thermostat': 8, 'Flat rate charging programmer no room thermostat': 9, 'Flat rate charging room thermostat only': 10, 'Manual charge control': 11, 'No thermostatic control of room temperature': 12, 'No time or thermostatic control of room temp': 13, 'No time or thermostatic control of room temperature': 14, 'None': 15, 'Programmer TRVs and boiler energy manager': 16, 'Programmer TRVs and bypass': 17, 'Programmer and appliance thermostats': 18, 'Programmer and at least 2 room thermostats': 19, 'Programmer and at least two room thermostats': 20, 'Programmer and room thermostat': 21, 'Programmer and room thermostats': 22, 'Programmer no room thermostat': 23, 'Programmer room thermostat and TRVs': 24, 'Room thermostat only': 25, 'Room thermostats only': 26, 'Temperature zone control': 27, 'Time and temperature zone control': 28, 'Unit charging programmer and TRVs': 29}, 
        'ROOF_DESCRIPTION': {'Flat': 0, 'Flat insulated': 1, 'Flat limited insulation': 2, 'Pitched': 3, 'Pitched 100mm loft insulation': 4, 'Pitched 12mm loft insulation': 5, 'Pitched 150mm loft insulation': 6, 'Pitched 200mm loft insulation': 7, 'Pitched 250mm loft insulation': 8, 'Pitched 25mm loft insulation': 9, 'Pitched 270mm loft insulation': 10, 'Pitched 300+mm loft insulation': 11, 'Pitched 300mm loft insulation': 12, 'Pitched 50mm loft insulation': 13, 'Pitched 75mm loft insulation': 14, 'Pitched insulated': 15, 'Pitched insulated at rafters': 16, 'Pitched limited insulation': 17, 'Roof room(s) ceiling insulated': 18, 'Roof room(s) insulated': 19, 'Roof room(s) limited insulation': 20, 'Roof room(s) no insulation': 21, 'Roof room(s) thatched': 22, 'Roof room(s) thatched with additional insulation': 23, 'Thatched': 24, 'Thatched with additional insulation': 25},
        # 'PROPERTY_TYPE': {'Bungalow': 0, 'Flat': 1, 'House': 2, 'Maisonette': 3}, 
        # 'ENERGY_TARIFF': {'Single': 0, 'Unknown': 1, 'dual': 2}, 
        # 'GLAZED_TYPE': {'double glazing installed before 2002': 0, 'double glazing installed during or after 2002': 1, 'double glazing, unknown install date': 2, 'not defined': 3, 'secondary glazing': 4, 'single glazing': 5, 'triple glazing': 6}, 
    }

    rate_dict = {
        # 'HOT_WATER_ENERGY_EFF' : {'Very good': 5, 'Good': 4, 'Averaged': 3, 'Poor': 2, 'Very poor': 1},
        # 'HOT_WATER_ENV_EFF' : {'Very good': 5, 'Good': 4, 'Averaged': 3, 'Poor': 2, 'Very poor': 1},
        # 'WINDOWS_ENERGY_EFF' : {'Very good': 5, 'Good': 4, 'Averaged': 3, 'Poor': 2, 'Very poor': 1},
        # # 'WINDOWS_ENV_EFF' : {'Very good': 5, 'Good': 4, 'Averaged': 3, 'Poor': 2, 'Very poor': 1},
        # 'WALLS_ENERGY_EFF' : {'Very good': 5, 'Good': 4, 'Averaged': 3, 'Poor': 2, 'Very poor': 1},
        # # 'WALLS_ENV_EFF' : {'Very good': 5, 'Good': 4, 'Averaged': 3, 'Poor': 2, 'Very poor': 1},
        # 'ROOF_ENERGY_EFF' : {'Very good': 5, 'Good': 4, 'Averaged': 3, 'Poor': 2, 'Very poor': 1},
        # # 'ROOF_ENV_EFF' : {'Very good': 5, 'Good': 4, 'Averaged': 3, 'Poor': 2, 'Very poor': 1},
        # 'MAINHEAT_ENERGY_EFF' : {'Very good': 5, 'Good': 4, 'Averaged': 3, 'Poor': 2, 'Very poor': 1},
        # 'MAINHEAT_ENV_EFF' : {'Very good': 5, 'Good': 4, 'Averaged': 3, 'Poor': 2, 'Very poor': 1},
        # 'MAINHEATC_ENERGY_EFF' : {'Very good': 5, 'Good': 4, 'Averaged': 3, 'Poor': 2, 'Very poor': 1},
        # 'MAINHEATC_ENV_EFF' : {'Very good': 5, 'Good': 4, 'Averaged': 3, 'Poor': 2, 'Very poor': 1},
        # 'LIGHTING_ENERGY_EFF' : {'Very good': 5, 'Good': 4, 'Averaged': 3, 'Poor': 2, 'Very poor': 1},
        # 'LIGHTING_ENV_EFF' : {'Very good': 5, 'Good': 4, 'Averaged': 3, 'Poor': 2, 'Very poor': 1}
    }

elif(LOD_STAGE == 2):

    TOTAL_LIST = [
        'BUILT_FORM',
        'TOTAL_FLOOR_AREA',
        'FLOOR_HEIGHT',
        'GLAZED_AREA',
        'NUMBER_HABITABLE_ROOMS',
        # 'WINDOWS_ENERGY_EFF',
        # 'WALLS_ENERGY_EFF',
        # 'ROOF_ENERGY_EFF',
        'WINDOWS_DESCRIPTION',
        'WALLS_DESCRIPTION',
        'ROOF_DESCRIPTION',

        'NUMBER_HEATED_ROOMS',
        'NUMBER_OPEN_FIREPLACES',
        'MULTI_GLAZE_PROPORTION',
        'MAINHEAT_DESCRIPTION',
        # 'MAINHEAT_ENERGY_EFF',

     ]

    cat_dict = {

        'BUILT_FORM': {'Detached': 0, 'Enclosed End-Terrace': 1, 'Enclosed Mid-Terrace': 2, 'End-Terrace': 3, 'Mid-Terrace': 4, 'Semi-Detached': 5}, 
        # 'MAINS_GAS_FLAG': {'N': 0, 'Y': 1}, 
        'GLAZED_AREA': {'Less Than Typical': 0, 'More Than Typical': 1, 'Much More Than Typical': 2, 'Normal': 3}, 
        # 'SECONDHEAT_DESCRIPTION': {'None': 0, 'Portable electric heaters': 1, 'Room heaters,': 2, 'Room heaters, LPG': 3, 'Room heaters, anthracite': 4, 'Room heaters, bottled LPG': 5, 'Room heaters, bottled gas': 6, 'Room heaters, coal': 7, 'Room heaters, dual fuel (mineral and wood)': 8, 'Room heaters, electric': 9, 'Room heaters, mains gas': 10, 'Room heaters, oil': 11, 'Room heaters, smokeless fuel': 12, 'Room heaters, wood logs': 13}
        'MAINHEAT_DESCRIPTION': {'Boiler and radiators, LPG': 0, 'Boiler and radiators, anthracite': 1, 'Boiler and radiators, bottled gas': 2, 'Boiler and radiators, coal': 3, 'Boiler and radiators, dual fuel (mineral and wood)': 4, 'Boiler and radiators, electric': 5, 'Boiler and radiators, mains gas': 6, 'Boiler and radiators, oil': 7, 'Boiler and radiators, smokeless fuel': 8, 'Boiler and underfloor heating, LPG': 9, 'Boiler and underfloor heating, mains gas': 10, 'Boiler and underfloor heating, oil': 11, 'Community scheme': 12, 'Community scheme with CHP': 13, 'Electric ceiling heating': 14, 'Electric storage heaters': 15, 'Electric underfloor heating': 16, 'No system present: electric heating assumed': 17, 'Portable electric heating assumed for most rooms': 18, 'Room heaters, electric': 19, 'Room heaters, mains gas': 20, 'Room heaters, wood logs': 21, 'Warm air, Electricaire': 22, 'Warm air, electric': 23, 'Warm air, mains gas': 24},
        # 'SOLAR_WATER_HEATING_FLAG': {'N': 0, 'Y': 1}, 
        # 'MECHANICAL_VENTILATION': {'mechanical, extract only': 0, 'mechanical, supply and extract': 1, 'natural': 2}, 
        
        # 'FLAT_TOP_STOREY': {'N': 0, 'Y': 1}, 
        # 'HOTWATER_DESCRIPTION': {'Electric immersion, dual tariff': 0, 'Electric immersion, off-peak': 1, 'Electric immersion, off-peak, no cylinder thermostat': 2, 'Electric immersion, off-peak, no cylinderstat': 3, 'Electric immersion, off-peak, plus solar': 4, 'Electric immersion, standard tariff': 5, 'Electric instantaneous at point of use': 6, 'From main system': 7, 'From main system, no cylinder thermostat': 8, 'From main system, no cylinder thermostat, plus solar': 9, 'From main system, no cylinderstat': 10, 'From main system, plus solar': 11, 'From main system, standard tariff': 12, 'From secondary system': 13, 'From secondary system, no cylinder thermostat': 14, 'From secondary system, no cylinderstat': 15, 'Gas instantaneous at point of use': 16, 'Gas multipoint': 17, 'No system present: electric immersion assumed': 18}, 
        'WINDOWS_DESCRIPTION': {'Fully double glazing': 0, 'Fully triple glazing': 1, 'Mostly double glazing': 2, 'Partial double glazing': 3, 'Single glazing': 4},
        'WALLS_DESCRIPTION': {'Cavity wall, as built, insulated': 0, 'Cavity wall, as built, no insulation': 1, 'Cavity wall, as built, partial insulation': 2, 'Cavity wall, filled cavity': 3, 'Cavity wall, with external insulation': 4, 'Cavity wall, with internal insulation': 5, 'Cob, as built': 6, 'Granite or whinstone, as built, insulated': 7, 'Granite or whinstone, as built, no insulation': 8, 'Granite or whinstone, with external insulation': 9, 'Granite or whinstone, with internal insulation': 10, 'Sandstone, as built, insulated': 11, 'Sandstone, as built, no insulation': 12, 'Sandstone, with internal insulation': 13, 'Solid brick, as built, insulated': 14, 'Solid brick, as built, no insulation': 15, 'Solid brick, as built, partial insulation': 16, 'Solid brick, with external insulation': 17, 'Solid brick, with internal insulation': 18, 'System built, as built, insulated': 19, 'System built, as built, no insulation': 20, 'System built, as built, partial insulation': 21, 'System built, with external insulation': 22, 'System built, with internal insulation': 23, 'Timber frame, as built, insulated': 24, 'Timber frame, as built, no insulation': 25, 'Timber frame, as built, partial insulation': 26, 'Timber frame, with internal insulation': 27},
        # 'FLOOR_DESCRIPTION': {'(other premises below)': 0, 'Solid, insulated': 1, 'Solid, insulated (assumed)': 2, 'Solid, limited insulation (assumed)': 3, 'Solid, no insulation (assumed)': 4, 'Suspended, insulated': 5, 'Suspended, insulated (assumed)': 6, 'Suspended, limited insulation (assumed)': 7, 'Suspended, no insulation (assumed)': 8, 'To external air, insulated': 9, 'To external air, insulated (assumed)': 10, 'To external air, limited insulation (assumed)': 11, 'To external air, no insulation (assumed)': 12, 'To external air, uninsulated (assumed)': 13, 'To unheated space, insulated': 14, 'To unheated space, insulated (assumed)': 15, 'To unheated space, limited insulation (assumed)': 16, 'To unheated space, no insulation (assumed)': 17, 'To unheated space, uninsulated (assumed)': 18}, 
        # 'MAINHEATCONT_DESCRIPTION': {'Appliance thermostats': 0, 'Automatic charge control': 1, 'Charging system linked to the use of community heating prog and TRVs': 2, 'Charging system linked to use of community heating programmer and TRVs': 3, 'Flat rate charging no thermostat control of room temperature': 4, 'Flat rate charging no thermostatic control': 5, 'Flat rate charging no thermostatic control of room temperature': 6, 'Flat rate charging programmer and TRVs': 7, 'Flat rate charging programmer and room thermostat': 8, 'Flat rate charging programmer no room thermostat': 9, 'Flat rate charging room thermostat only': 10, 'Manual charge control': 11, 'No thermostatic control of room temperature': 12, 'No time or thermostatic control of room temp': 13, 'No time or thermostatic control of room temperature': 14, 'None': 15, 'Programmer TRVs and boiler energy manager': 16, 'Programmer TRVs and bypass': 17, 'Programmer and appliance thermostats': 18, 'Programmer and at least 2 room thermostats': 19, 'Programmer and at least two room thermostats': 20, 'Programmer and room thermostat': 21, 'Programmer and room thermostats': 22, 'Programmer no room thermostat': 23, 'Programmer room thermostat and TRVs': 24, 'Room thermostat only': 25, 'Room thermostats only': 26, 'Temperature zone control': 27, 'Time and temperature zone control': 28, 'Unit charging programmer and TRVs': 29}, 
        'ROOF_DESCRIPTION': {'Flat': 0, 'Flat insulated': 1, 'Flat limited insulation': 2, 'Pitched': 3, 'Pitched 100mm loft insulation': 4, 'Pitched 12mm loft insulation': 5, 'Pitched 150mm loft insulation': 6, 'Pitched 200mm loft insulation': 7, 'Pitched 250mm loft insulation': 8, 'Pitched 25mm loft insulation': 9, 'Pitched 270mm loft insulation': 10, 'Pitched 300+mm loft insulation': 11, 'Pitched 300mm loft insulation': 12, 'Pitched 50mm loft insulation': 13, 'Pitched 75mm loft insulation': 14, 'Pitched insulated': 15, 'Pitched insulated at rafters': 16, 'Pitched limited insulation': 17, 'Roof room(s) ceiling insulated': 18, 'Roof room(s) insulated': 19, 'Roof room(s) limited insulation': 20, 'Roof room(s) no insulation': 21, 'Roof room(s) thatched': 22, 'Roof room(s) thatched with additional insulation': 23, 'Thatched': 24, 'Thatched with additional insulation': 25},
        # 'PROPERTY_TYPE': {'Bungalow': 0, 'Flat': 1, 'House': 2, 'Maisonette': 3}, 
        # 'ENERGY_TARIFF': {'Single': 0, 'Unknown': 1, 'dual': 2}, 
        # 'GLAZED_TYPE': {'double glazing installed before 2002': 0, 'double glazing installed during or after 2002': 1, 'double glazing, unknown install date': 2, 'not defined': 3, 'secondary glazing': 4, 'single glazing': 5, 'triple glazing': 6}, 
         
}

    rate_dict = {
        # 'HOT_WATER_ENERGY_EFF' : {'Very good': 5, 'Good': 4, 'Averaged': 3, 'Poor': 2, 'Very poor': 1},
        # 'HOT_WATER_ENV_EFF' : {'Very good': 5, 'Good': 4, 'Averaged': 3, 'Poor': 2, 'Very poor': 1},
        # 'WINDOWS_ENERGY_EFF' : {'Very good': 5, 'Good': 4, 'Averaged': 3, 'Poor': 2, 'Very poor': 1},
        # # 'WINDOWS_ENV_EFF' : {'Very good': 5, 'Good': 4, 'Averaged': 3, 'Poor': 2, 'Very poor': 1},
        # 'WALLS_ENERGY_EFF' : {'Very good': 5, 'Good': 4, 'Averaged': 3, 'Poor': 2, 'Very poor': 1},
        # # 'WALLS_ENV_EFF' : {'Very good': 5, 'Good': 4, 'Averaged': 3, 'Poor': 2, 'Very poor': 1},
        # 'ROOF_ENERGY_EFF' : {'Very good': 5, 'Good': 4, 'Averaged': 3, 'Poor': 2, 'Very poor': 1},
        # # 'ROOF_ENV_EFF' : {'Very good': 5, 'Good': 4, 'Averaged': 3, 'Poor': 2, 'Very poor': 1},
        # 'MAINHEAT_ENERGY_EFF' : {'Very good': 5, 'Good': 4, 'Averaged': 3, 'Poor': 2, 'Very poor': 1},
        # 'MAINHEAT_ENV_EFF' : {'Very good': 5, 'Good': 4, 'Averaged': 3, 'Poor': 2, 'Very poor': 1},
        # 'MAINHEATC_ENERGY_EFF' : {'Very good': 5, 'Good': 4, 'Averaged': 3, 'Poor': 2, 'Very poor': 1},
        # 'MAINHEATC_ENV_EFF' : {'Very good': 5, 'Good': 4, 'Averaged': 3, 'Poor': 2, 'Very poor': 1},
        # 'LIGHTING_ENERGY_EFF' : {'Very good': 5, 'Good': 4, 'Averaged': 3, 'Poor': 2, 'Very poor': 1},
        # 'LIGHTING_ENV_EFF' : {'Very good': 5, 'Good': 4, 'Averaged': 3, 'Poor': 2, 'Very poor': 1}
    }

elif(LOD_STAGE == 3):

    TOTAL_LIST = [
        'BUILT_FORM',
        'TOTAL_FLOOR_AREA',
        'FLOOR_HEIGHT',
        'GLAZED_AREA',
        'NUMBER_HABITABLE_ROOMS',
        # 'WINDOWS_ENERGY_EFF',
        # 'WALLS_ENERGY_EFF',
        # 'ROOF_ENERGY_EFF',
        'WINDOWS_DESCRIPTION',
        'WALLS_DESCRIPTION',
        'ROOF_DESCRIPTION',

        'NUMBER_HEATED_ROOMS',
        'NUMBER_OPEN_FIREPLACES',
        'MULTI_GLAZE_PROPORTION',
        'MAINHEAT_DESCRIPTION',
        # 'MAINHEAT_ENERGY_EFF',

        'SECONDHEAT_DESCRIPTION',
        'MECHANICAL_VENTILATION',
        'MAINS_GAS_FLAG',
        'SOLAR_WATER_HEATING_FLAG',
        # 'HOT_WATER_ENERGY_EFF',
        'PHOTO_SUPPLY',
     ]

    cat_dict = {


        'BUILT_FORM': {'Detached': 0, 'Enclosed End-Terrace': 1, 'Enclosed Mid-Terrace': 2, 'End-Terrace': 3, 'Mid-Terrace': 4, 'Semi-Detached': 5}, 
        'MAINS_GAS_FLAG': {'N': 0, 'Y': 1}, 
        'GLAZED_AREA': {'Less Than Typical': 0, 'More Than Typical': 1, 'Much More Than Typical': 2, 'Normal': 3}, 
        'SECONDHEAT_DESCRIPTION': {'None': 0, 'Portable electric heaters': 1, 'Room heaters,': 2, 'Room heaters, LPG': 3, 'Room heaters, anthracite': 4, 'Room heaters, bottled LPG': 5, 'Room heaters, bottled gas': 6, 'Room heaters, coal': 7, 'Room heaters, dual fuel (mineral and wood)': 8, 'Room heaters, electric': 9, 'Room heaters, mains gas': 10, 'Room heaters, oil': 11, 'Room heaters, smokeless fuel': 12, 'Room heaters, wood logs': 13},
        'MAINHEAT_DESCRIPTION': {'Boiler and radiators, LPG': 0, 'Boiler and radiators, anthracite': 1, 'Boiler and radiators, bottled gas': 2, 'Boiler and radiators, coal': 3, 'Boiler and radiators, dual fuel (mineral and wood)': 4, 'Boiler and radiators, electric': 5, 'Boiler and radiators, mains gas': 6, 'Boiler and radiators, oil': 7, 'Boiler and radiators, smokeless fuel': 8, 'Boiler and underfloor heating, LPG': 9, 'Boiler and underfloor heating, mains gas': 10, 'Boiler and underfloor heating, oil': 11, 'Community scheme': 12, 'Community scheme with CHP': 13, 'Electric ceiling heating': 14, 'Electric storage heaters': 15, 'Electric underfloor heating': 16, 'No system present: electric heating assumed': 17, 'Portable electric heating assumed for most rooms': 18, 'Room heaters, electric': 19, 'Room heaters, mains gas': 20, 'Room heaters, wood logs': 21, 'Warm air, Electricaire': 22, 'Warm air, electric': 23, 'Warm air, mains gas': 24},
        'SOLAR_WATER_HEATING_FLAG': {'N': 0, 'Y': 1}, 
        'MECHANICAL_VENTILATION': {'mechanical, extract only': 0, 'mechanical, supply and extract': 1, 'natural': 2}, 
        
        # 'FLAT_TOP_STOREY': {'N': 0, 'Y': 1}, 
        # 'HOTWATER_DESCRIPTION': {'Electric immersion, dual tariff': 0, 'Electric immersion, off-peak': 1, 'Electric immersion, off-peak, no cylinder thermostat': 2, 'Electric immersion, off-peak, no cylinderstat': 3, 'Electric immersion, off-peak, plus solar': 4, 'Electric immersion, standard tariff': 5, 'Electric instantaneous at point of use': 6, 'From main system': 7, 'From main system, no cylinder thermostat': 8, 'From main system, no cylinder thermostat, plus solar': 9, 'From main system, no cylinderstat': 10, 'From main system, plus solar': 11, 'From main system, standard tariff': 12, 'From secondary system': 13, 'From secondary system, no cylinder thermostat': 14, 'From secondary system, no cylinderstat': 15, 'Gas instantaneous at point of use': 16, 'Gas multipoint': 17, 'No system present: electric immersion assumed': 18}, 
        'WINDOWS_DESCRIPTION': {'Fully double glazing': 0, 'Fully triple glazing': 1, 'Mostly double glazing': 2, 'Partial double glazing': 3, 'Single glazing': 4},
        'WALLS_DESCRIPTION': {'Cavity wall, as built, insulated': 0, 'Cavity wall, as built, no insulation': 1, 'Cavity wall, as built, partial insulation': 2, 'Cavity wall, filled cavity': 3, 'Cavity wall, with external insulation': 4, 'Cavity wall, with internal insulation': 5, 'Cob, as built': 6, 'Granite or whinstone, as built, insulated': 7, 'Granite or whinstone, as built, no insulation': 8, 'Granite or whinstone, with external insulation': 9, 'Granite or whinstone, with internal insulation': 10, 'Sandstone, as built, insulated': 11, 'Sandstone, as built, no insulation': 12, 'Sandstone, with internal insulation': 13, 'Solid brick, as built, insulated': 14, 'Solid brick, as built, no insulation': 15, 'Solid brick, as built, partial insulation': 16, 'Solid brick, with external insulation': 17, 'Solid brick, with internal insulation': 18, 'System built, as built, insulated': 19, 'System built, as built, no insulation': 20, 'System built, as built, partial insulation': 21, 'System built, with external insulation': 22, 'System built, with internal insulation': 23, 'Timber frame, as built, insulated': 24, 'Timber frame, as built, no insulation': 25, 'Timber frame, as built, partial insulation': 26, 'Timber frame, with internal insulation': 27},
        # 'FLOOR_DESCRIPTION': {'(other premises below)': 0, 'Solid, insulated': 1, 'Solid, insulated (assumed)': 2, 'Solid, limited insulation (assumed)': 3, 'Solid, no insulation (assumed)': 4, 'Suspended, insulated': 5, 'Suspended, insulated (assumed)': 6, 'Suspended, limited insulation (assumed)': 7, 'Suspended, no insulation (assumed)': 8, 'To external air, insulated': 9, 'To external air, insulated (assumed)': 10, 'To external air, limited insulation (assumed)': 11, 'To external air, no insulation (assumed)': 12, 'To external air, uninsulated (assumed)': 13, 'To unheated space, insulated': 14, 'To unheated space, insulated (assumed)': 15, 'To unheated space, limited insulation (assumed)': 16, 'To unheated space, no insulation (assumed)': 17, 'To unheated space, uninsulated (assumed)': 18}, 
        # 'MAINHEATCONT_DESCRIPTION': {'Appliance thermostats': 0, 'Automatic charge control': 1, 'Charging system linked to the use of community heating prog and TRVs': 2, 'Charging system linked to use of community heating programmer and TRVs': 3, 'Flat rate charging no thermostat control of room temperature': 4, 'Flat rate charging no thermostatic control': 5, 'Flat rate charging no thermostatic control of room temperature': 6, 'Flat rate charging programmer and TRVs': 7, 'Flat rate charging programmer and room thermostat': 8, 'Flat rate charging programmer no room thermostat': 9, 'Flat rate charging room thermostat only': 10, 'Manual charge control': 11, 'No thermostatic control of room temperature': 12, 'No time or thermostatic control of room temp': 13, 'No time or thermostatic control of room temperature': 14, 'None': 15, 'Programmer TRVs and boiler energy manager': 16, 'Programmer TRVs and bypass': 17, 'Programmer and appliance thermostats': 18, 'Programmer and at least 2 room thermostats': 19, 'Programmer and at least two room thermostats': 20, 'Programmer and room thermostat': 21, 'Programmer and room thermostats': 22, 'Programmer no room thermostat': 23, 'Programmer room thermostat and TRVs': 24, 'Room thermostat only': 25, 'Room thermostats only': 26, 'Temperature zone control': 27, 'Time and temperature zone control': 28, 'Unit charging programmer and TRVs': 29}, 
        'ROOF_DESCRIPTION': {'Flat': 0, 'Flat insulated': 1, 'Flat limited insulation': 2, 'Pitched': 3, 'Pitched 100mm loft insulation': 4, 'Pitched 12mm loft insulation': 5, 'Pitched 150mm loft insulation': 6, 'Pitched 200mm loft insulation': 7, 'Pitched 250mm loft insulation': 8, 'Pitched 25mm loft insulation': 9, 'Pitched 270mm loft insulation': 10, 'Pitched 300+mm loft insulation': 11, 'Pitched 300mm loft insulation': 12, 'Pitched 50mm loft insulation': 13, 'Pitched 75mm loft insulation': 14, 'Pitched insulated': 15, 'Pitched insulated at rafters': 16, 'Pitched limited insulation': 17, 'Roof room(s) ceiling insulated': 18, 'Roof room(s) insulated': 19, 'Roof room(s) limited insulation': 20, 'Roof room(s) no insulation': 21, 'Roof room(s) thatched': 22, 'Roof room(s) thatched with additional insulation': 23, 'Thatched': 24, 'Thatched with additional insulation': 25},
        # 'PROPERTY_TYPE': {'Bungalow': 0, 'Flat': 1, 'House': 2, 'Maisonette': 3}, 
        # 'ENERGY_TARIFF': {'Single': 0, 'Unknown': 1, 'dual': 2}, 
        # 'GLAZED_TYPE': {'double glazing installed before 2002': 0, 'double glazing installed during or after 2002': 1, 'double glazing, unknown install date': 2, 'not defined': 3, 'secondary glazing': 4, 'single glazing': 5, 'triple glazing': 6}, 
    }

    rate_dict = {
        # 'HOT_WATER_ENERGY_EFF' : {'Very good': 5, 'Good': 4, 'Averaged': 3, 'Poor': 2, 'Very poor': 1},
        # # 'HOT_WATER_ENV_EFF' : {'Very good': 5, 'Good': 4, 'Averaged': 3, 'Poor': 2, 'Very poor': 1},
        # 'WINDOWS_ENERGY_EFF' : {'Very good': 5, 'Good': 4, 'Averaged': 3, 'Poor': 2, 'Very poor': 1},
        # # 'WINDOWS_ENV_EFF' : {'Very good': 5, 'Good': 4, 'Averaged': 3, 'Poor': 2, 'Very poor': 1},
        # 'WALLS_ENERGY_EFF' : {'Very good': 5, 'Good': 4, 'Averaged': 3, 'Poor': 2, 'Very poor': 1},
        # # 'WALLS_ENV_EFF' : {'Very good': 5, 'Good': 4, 'Averaged': 3, 'Poor': 2, 'Very poor': 1},
        # 'ROOF_ENERGY_EFF' : {'Very good': 5, 'Good': 4, 'Averaged': 3, 'Poor': 2, 'Very poor': 1},
        # # 'ROOF_ENV_EFF' : {'Very good': 5, 'Good': 4, 'Averaged': 3, 'Poor': 2, 'Very poor': 1},
        # 'MAINHEAT_ENERGY_EFF' : {'Very good': 5, 'Good': 4, 'Averaged': 3, 'Poor': 2, 'Very poor': 1},
        # 'MAINHEAT_ENV_EFF' : {'Very good': 5, 'Good': 4, 'Averaged': 3, 'Poor': 2, 'Very poor': 1},
        # 'MAINHEATC_ENERGY_EFF' : {'Very good': 5, 'Good': 4, 'Averaged': 3, 'Poor': 2, 'Very poor': 1},
        # 'MAINHEATC_ENV_EFF' : {'Very good': 5, 'Good': 4, 'Averaged': 3, 'Poor': 2, 'Very poor': 1},
        # 'LIGHTING_ENERGY_EFF' : {'Very good': 5, 'Good': 4, 'Averaged': 3, 'Poor': 2, 'Very poor': 1},
        # 'LIGHTING_ENV_EFF' : {'Very good': 5, 'Good': 4, 'Averaged': 3, 'Poor': 2, 'Very poor': 1}
    }


LOD_1_L = [
        'BUILT_FORM',
        'TOTAL_FLOOR_AREA',
        'FLOOR_HEIGHT',
        'GLAZED_AREA',
        'NUMBER_HABITABLE_ROOMS',
        # 'WINDOWS_ENERGY_EFF',
        # 'WALLS_ENERGY_EFF',
        # 'ROOF_ENERGY_EFF',
        'WINDOWS_DESCRIPTION',
        'WALLS_DESCRIPTION',
        'ROOF_DESCRIPTION',
     ]
LOD_2_L = [
        'NUMBER_HEATED_ROOMS',
        'NUMBER_OPEN_FIREPLACES',
        'MULTI_GLAZE_PROPORTION',
        'MAINHEAT_DESCRIPTION',
        # 'MAINHEAT_ENERGY_EFF',
     ]
LOD_3_L = [
        'SECONDHEAT_DESCRIPTION',
        'MECHANICAL_VENTILATION',
        'MAINS_GAS_FLAG',
        'SOLAR_WATER_HEATING_FLAG',
        # 'HOT_WATER_ENERGY_EFF',
        'PHOTO_SUPPLY',
     ]

status_dict = {}
# Known input checked!!!
# st.header('Known inputs:')
with st.beta_expander('Known inputs:'):
    

    col1, col2, col3 = st.beta_columns(3)
    with col1:
        all_check_1 = st.checkbox('I know all the LOD1 inputs!', key="1")
        for each in range(0, len(LOD_1_L)):
            if(all_check_1):
                status_dict[LOD_1_L[each]] = st.checkbox(LOD_1_L[each], value=True)
            else:
                status_dict[LOD_1_L[each]] = st.checkbox(LOD_1_L[each])
    if(LOD_STAGE == 2 or LOD_STAGE == 3):
        with col2:
            all_check_2 = st.checkbox('I know all the LOD2 inputs!', key="2")
            for each in range(0, len(LOD_2_L)):
                if(all_check_2):
                    status_dict[LOD_2_L[each]] = st.checkbox(LOD_2_L[each], value=True)
                else:
                    status_dict[LOD_2_L[each]] = st.checkbox(LOD_2_L[each])  
    if(LOD_STAGE == 3):
        with col3:
            all_check_3 = st.checkbox('I know all the LOD3 inputs!', key="3")
            for each in range(0, len(LOD_3_L)):
                if(all_check_3):
                    status_dict[LOD_3_L[each]] = st.checkbox(LOD_3_L[each], value=True)
                else:
                    status_dict[LOD_3_L[each]] = st.checkbox(LOD_3_L[each]) 
    # if(all_check):
    #     for each in TOTAL_LIST:
    #         status_dict[each] = True
# st.write(status_dict)
# interaction 
def user_input_features(cat_dict, rate_dict, status_dict):
    out_cat_dict = {}
    out_rate_dict = {}
    out_float_dict = {}
    # st.sidebar.text_input(each_cal)
    if(status_dict['TOTAL_FLOOR_AREA'] is True):
        TOTAL_FLOOR_AREA = st.sidebar.slider(
            'TOTAL_FLOOR_AREA',
            0.1, 500.0, (50.0)
        )
        out_float_dict['TOTAL_FLOOR_AREA'] = TOTAL_FLOOR_AREA
    
    if(status_dict['FLOOR_HEIGHT'] is True):
        FLOOR_HEIGHT = st.sidebar.slider(
            'FLOOR_HEIGHT',
            0.1, 20.0, (3.0)
        )
        out_float_dict['FLOOR_HEIGHT'] = FLOOR_HEIGHT

    if(status_dict['NUMBER_HABITABLE_ROOMS'] is True):
        NUMBER_HABITABLE_ROOMS = st.sidebar.slider(
            'NUMBER_HABITABLE_ROOMS',
            1, 25, 3, 1
        )
        out_float_dict['NUMBER_HABITABLE_ROOMS'] = NUMBER_HABITABLE_ROOMS

    if(LOD_STAGE == 2 or LOD_STAGE==3):
        if(status_dict['MULTI_GLAZE_PROPORTION'] is True):
            MULTI_GLAZE_PROPORTION = st.sidebar.slider(
                'MULTI_GLAZE_PROPORTION',
                0.0, 100.0, (10.0)
            )
            out_float_dict['MULTI_GLAZE_PROPORTION'] = MULTI_GLAZE_PROPORTION

        if(status_dict['NUMBER_HEATED_ROOMS'] is True):
            NUMBER_HEATED_ROOMS = st.sidebar.slider(
                'NUMBER_HEATED_ROOMS',
                0, NUMBER_HABITABLE_ROOMS, 0, 1
            )
            out_float_dict['NUMBER_HEATED_ROOMS'] = NUMBER_HEATED_ROOMS

        if(status_dict['NUMBER_OPEN_FIREPLACES'] is True):
            NUMBER_OPEN_FIREPLACES = st.sidebar.slider(
                'NUMBER_OPEN_FIREPLACES',
                0, 100, 1, 1
            )
            out_float_dict['NUMBER_OPEN_FIREPLACES'] = NUMBER_OPEN_FIREPLACES

    if(LOD_STAGE==3):
        if(status_dict['PHOTO_SUPPLY'] is True):
            PHOTO_SUPPLY = st.sidebar.slider(
                'PHOTO_SUPPLY',
                0, 100, 0, 1
            )
            out_float_dict['PHOTO_SUPPLY'] = PHOTO_SUPPLY


    for each_cal in cat_dict:
        if(status_dict[each_cal] is True):
            out_cat_dict[each_cal] = st.sidebar.selectbox(each_cal, 
                tuple(cat_dict[each_cal].keys())
                )
    for each_rate in rate_dict:
        if(status_dict[each_rate] is True):
            out_rate_dict[each_rate] = st.sidebar.selectbox(each_rate, 
                tuple(rate_dict[each_rate].keys())
                )
    # if(status_dict['WIND_TURBINE_COUNT'] is True):
    #     WIND_TURBINE_COUNT = st.sidebar.slider(
    #         'WIND_TURBINE_COUNT',
    #         0, 1, 0, 1
    #     )
    #     out_float_dict['WIND_TURBINE_COUNT'] = WIND_TURBINE_COUNT


    # PROPERTY_TYPE = st.sidebar.selectbox(
    # 'PROPERTY_TYPE',
    # (
    # 'Maisonette', 'Bungalow', 'House', 'Flat'
    # ))
    return out_cat_dict, out_rate_dict, out_float_dict


##############
##############

# model loaded 
def load_model(LOD_STAGE):
    # # CURRENT_ENERGY_EFFICIENCY
    # # target = open(r'C:/Users/msi-/Desktop/RF/ngb_model_ALL_DATA_2020-09-26_18-50-19_v1.bin','rb')#注意此处是rb
    # ngb_CURR = joblib.load('ngb_model_ALL_DATA_2020-09-26_18-50-19_v1.pkl')
    # # LCA CO2_EMISSIONS_CURRENT
    # # target = open(r'C:/Users/msi-/Desktop/RF/ngb_model_ALL_DATA_2020-10-22_12-02-05_v1_LCA.bin','rb')#注意此处是rb
    # ngb_CO2 = joblib.load('ngb_model_ALL_DATA_2020-10-22_12-02-05_v1_LCA.pkl')
    with gzip.open('D_'+str(LOD_STAGE)+'_ENERGY_CONSUMPTION_CURRENT_PER_M2.pklz', 'rb') as ifp:
        ngb_CURR = pickle.load(ifp)
    with gzip.open('D_'+str(LOD_STAGE)+'_CO2_EMISS_CURR_PER_FLOOR_AREA.pklz', 'rb') as ifp:
        ngb_CO2 = pickle.load(ifp)
    with gzip.open('D_'+str(LOD_STAGE)+'_COST_OPE_CURRENT_PER_M2.pklz', 'rb') as ifp:
        ngb_COST = pickle.load(ifp)
    # st.write("""
    #         Model loaded! 
    #         """)
    return ngb_CURR, ngb_CO2, ngb_COST


def point_prediction(SUBJECT, model, mu_list, sigma_list, y_dists, input_f):
    st.header( SUBJECT )

    ############## diagram
    n = 0
    quantitle = [1, 0.32, 0.05, 0.003]
    p_result_df = pd.DataFrame(columns=('', 'Result'))
    for index in range(0,len(quantitle)):
        each_q = quantitle[index]
        left_part  = sct.norm.ppf(q=each_q/2,loc=mu_list[n],scale=sigma_list[n])
        right_part = sct.norm.isf(q=each_q/2,loc=mu_list[n],scale=sigma_list[n])

        left_part = max(left_part, 0)
        right_part = min(right_part, 100)
    #     print(str((1-each_q)*100), '% the value would between: ', (round(left_part, 2) , round(right_part, 2)))
        # st.write((round(left_part, 2) , round(right_part, 2)))
        if(index == 0):
            p_result_df.loc[index] = ['Point prediction', round(left_part, 2)]
            p_mean = left_part
        else:
            # text = 'Probabilistic prediction: ' + str((1-each_q)*100) + '% the value would between'
            text = str((1-each_q)*100) + '%' + ' confidence'
            p_result_df.loc[index] = [text, (round(left_part, 2) , round(right_part, 2))]

    ############## plot
    mu_1 = mu_list[0]
    sigma_1 = sigma_list[0]

    # x = np.linspace(mu_1-5*sigma_1,mu_1+5*sigma_1,1000)
    if(SUBJECT == 'Energy (kWh/m²·year)'):
        x = np.linspace(0, max(p_mean+5, 15), 1000)
    elif(SUBJECT == 'CO2 (kg/m²·year)'):
        x = np.linspace(10, max(p_mean+5, 85), 1000)
    elif(SUBJECT == 'Cost (£/m²·year)'):   
        x = np.linspace(0, max(p_mean+5, 20), 1000)
    
    a1 = norm.pdf(x, loc=mu_1, scale=sigma_1)

    fig, ax = plt.subplots()
    plt.plot(x, a1, color='r',label='Building_1', linewidth=2)
    plt.xlabel(SUBJECT)
    plt.ylabel('Probability Density')
    plt.title(SUBJECT + ' prediction',fontsize=14)
    plt.legend()
    plt.grid(True) #x坐标轴的网格使用主刻度

    ############## feture importance
    input_col = df.columns.values.tolist()
    ## Feature importance for loc trees
    feature_importance_loc = model.feature_importances_[0]

    ## Feature importance for scale trees
    feature_importance_scale = model.feature_importances_[1]

    df_loc = pd.DataFrame({'feature':input_col, 
                           'importance':feature_importance_loc}).sort_values('importance',ascending=False)
    df_scale = pd.DataFrame({'feature':input_col, 
                           'importance':feature_importance_scale}).sort_values('importance',ascending=False)

    ############## presentation
    st.write(y_dists.params.items())
    # col1, col2 = st.beta_columns(2)
    # with col1:
    st.write(p_result_df)
    # with col2:
    plt.tight_layout()
    st.pyplot(fig)

    # decision_plot
    explainer = shap.TreeExplainer(model)
    expected_value = explainer.expected_value
    shap_values = explainer.shap_values(input_f)

    with st.beta_expander(SUBJECT + '\'s detail'):
        # possibility distribution
        # st.pyplot(fig)
        st.write('---')
        # waterfall
        st.write('Waterfall plot:')
        fig_w, ax = plt.subplots(figsize=(6,13)) 
        plt.tight_layout()
        # shap_w = shap_transform_scale(explainer(input_f), p_mean, 0)
        # p_w = shap.plots.waterfall(shap_w)
        shap.plots._waterfall.waterfall_legacy(expected_value[0], shap_values[0], feature_names=input_f.columns)
        st.pyplot(fig_w)

        # decsion plot
        st.write('---')
        st.write('Decsion plot:')
        fig_d, ax = plt.subplots(figsize=(6,13)) 
        plt.tight_layout()
        shap_d = shap.decision_plot(expected_value, shap_values, input_f.columns,feature_order='hclust', return_objects=True)
        st.pyplot(fig_d)

        # feature importance
        st.write('---')
        st.write('Feature importance plot:')
        fig_imp, (ax1, ax2) = plt.subplots(2, 1, figsize=(6,13))
        fig_imp.subplots_adjust(wspace = 0.8)
        # fig_imp.suptitle("Feature importance plot for distribution parameters", fontsize=17)
        plt.tight_layout()
        sns.barplot(x='importance',y='feature',ax=ax1,data=df_loc, color="skyblue").set_title('loc param')
        sns.barplot(x='importance',y='feature',ax=ax2,data=df_scale, color="skyblue").set_title('scale param')
        st.pyplot(fig_imp)

        # 记录shap贡献值，做成dataframe
        shap_df = pd.DataFrame(shap_d.shap_values)
        shap_df.columns = shap_d.feature_names
        st.write(shap_df)


# explain the model again
    return shap_df



def estimation(model, SUBJECT, input_f, cat_dict, warning=False, TOTAL_LIST=TOTAL_LIST):
    
    permutation_list = []
    permutation_col_name = []
    reverse_cat_dict = {}
    for each_key in cat_dict.keys():
        reverse_cat_dict[each_key] = dict((y,x) for x,y in cat_dict[each_key].items())
    
    for each_col in input_f.columns.values.tolist():
        # 如果是缺失值
        if(pd.isnull(input_f[each_col]).item() != False):
            permutation_col_name.append(each_col)
            # 如果属于cat
            if(each_col in cat_dict):
                # st.write(list(cat_dict[each_col].values()))
                permutation_list.append(list(cat_dict[each_col].values()))
            # 如果属于rate
            elif(each_col in rate_dict):
                permutation_list.append(list(rate_dict[each_col].values()))
            # 剩下的
            else:
                if(each_col == 'TOTAL_FLOOR_AREA'):
                    permutation_list.append(list(np.linspace(0.1, 500, 100, endpoint=True)))
                if(each_col == 'FLOOR_HEIGHT'):
                    permutation_list.append(list(np.linspace(0.1, 20, 40, endpoint=True)))
                if(each_col == 'MULTI_GLAZE_PROPORTION'):
                    permutation_list.append(list(np.linspace(0, 100, 101, endpoint=True)))
                if(each_col == 'NUMBER_HABITABLE_ROOMS'):
                    permutation_list.append(list(np.linspace(1, 25, 25, endpoint=True)))
                if(each_col == 'NUMBER_HEATED_ROOMS'):
                    permutation_list.append(list(np.linspace(1, 25, 25, endpoint=True)))
                if(each_col == 'NUMBER_OPEN_FIREPLACES'):
                    permutation_list.append(list(np.linspace(1, 10, 10, endpoint=True)))
                if(each_col == 'WIND_TURBINE_COUNT'):
                    permutation_list.append([0,1])
                if(each_col == 'PHOTO_SUPPLY'):
                    permutation_list.append(list(np.linspace(0, 100, 101, endpoint=True)))
    # calculate permutation length 计算需要遍历的个数
    length = 1
    for each_len in permutation_list:
        # st.write(each_len)
        length = length*len(each_len)
    
    # st.write(permutation_col_name)
    # st.write(length)
    
    if(length >= 8000000):
        if(warning is True):
            st.header('We need more input!')
    else:
        # st.header('Let us make estimation!')
        st.header(SUBJECT)
        target_permutation_df = pd.DataFrame()
        target_permutation_df = pd.DataFrame(list(itertools.product(*permutation_list)), columns=permutation_col_name)
        
        for each_col in input_f.columns.values.tolist():
            # 如果不是缺省项
            if(pd.isnull(input_f[each_col]).item() != True):
                target_permutation_df[each_col] = input_f[each_col].values.tolist()[0]
            else:
                pass
        # 对齐顺序
        target_permutation_df = target_permutation_df[input_f.columns.values.tolist()]

        # down_sampling
        if(len(target_permutation_df)>=500):
            target_permutation_df = target_permutation_df.sample(500)
        
        # estimation model
        # st.write(target_permutation_df)

        y_preds, y_dists, mu_list, sigma_list = open_model_box(model, SUBJECT, target_permutation_df, go_point_prediction=False)

        mu_list = np.array(mu_list).flatten()
        sigma_list = np.array(sigma_list).flatten()

        # plot estimation
        space = (max(y_preds) - min(y_preds))*2
        min_value = max(min(y_preds)-space, 0)
        x = np.linspace(min_value, max(y_preds)+space,1000)

        fig_e, ax = plt.subplots()

        # plt.figure(figsize=(20,18))
        for each_p in range(0, len(mu_list)):
            a1 = norm.pdf(x,loc=mu_list[each_p], scale=sigma_list[each_p])
            if(each_p == 0):
                mix = a1/len(mu_list)
            else:
                mix = mix + a1/len(mu_list)
            # 细线
            if(len(target_permutation_df)<=100):
                if(each_p == 0):
                    plt.plot(x, a1, color='bisque',linewidth=0.5, linestyle="-", label='Probability of a single prediction')
                if(each_p % 5 == 0):
                # else:
                    plt.plot(x, a1, color='bisque',linewidth=0.5, linestyle="-")


        plt.plot(x, mix,linewidth=2, linestyle="-", label='Output probability distribution')


        sns.distplot(y_preds, label='Output Distribution')
        sns.kdeplot(y_preds, label='Kernel density estimate of output',color='r')
        plt.legend(fontsize=6)
        plt.grid()
        plt.xlabel(SUBJECT)
        plt.ylabel('Probability Density')
        plt.tight_layout()
        plt.title(SUBJECT + ' distribution',fontsize=14)


        ######## diagramm 
        # Input uncertainties
        u_result_df = pd.DataFrame(columns=('Input uncertainties', ' '))
        u_result_df.loc[0] = ['Output range', (round(min(y_preds), 1), round(max(y_preds), 2))]
        u_result_df.loc[1] = ['mean', round(np.mean(y_preds), 2)]
        u_result_df.loc[2] = ['median', round(np.median(y_preds), 2)]

        

        # Probability distribution
        p_result_df = pd.DataFrame(columns=('Prob. distribution', ' '))
        # fit curve (gamma)
        mix_change = (mix-np.min(mix))/(np.max(mix)-np.min(mix))*100

        sample_t = pd.DataFrame()
        sample_t['p'] = mix_change
        sample_t['v'] = x

        sample_pool = []
        for index, row in sample_t.iterrows():
            num = int(row['p'])
            value = row['v']

            sample_pool.append([value]*num)

        final_d = [i for item in sample_pool for i in item]
        final_d = np.array(final_d)

        ae, loce, scalee = stats.gamma.fit(final_d)
        a3 = gamma.pdf(x, ae, loce, scalee)
        quantitle = [1, 0.32, 0.05, 0.003]

        for each_q in range(0, len(quantitle)):
            left_part  = stats.gamma.ppf(q=quantitle[each_q]/2,a=ae ,loc=loce,scale=scalee)
            right_part  = stats.gamma.isf(q=quantitle[each_q]/2,a=ae ,loc=loce,scale=scalee)
            if(each_q == 0):
                text = 'Point prediction'
                p_result_df.loc[each_q] = [text, round(left_part, 2)]
            else:
                text = str((1-quantitle[each_q])*100) + '%' + ' confidence'
                p_result_df.loc[each_q] = [text, (round(left_part, 2) , round(right_part, 2))]
        
        # col1, col2 = st.beta_columns(2)
        # with col1:
        st.write(u_result_df)
        st.write(p_result_df)
        # with col2:
        plt.tight_layout()
        st.pyplot(fig_e)
        
        # fig_i, ax = plt.subplots()
        # shap.initjs()
        # explainer = shap.TreeExplainer(model, model_output=0) # use model_output = 1 for scale trees
        # if(len(target_permutation_df) > 25):
        #     target_permutation_df = target_permutation_df.sample(n=25, replace=True)
        # shap_values = explainer.shap_values(target_permutation_df)
        # shap.summary_plot(shap_values, target_permutation_df, feature_names=input_f.columns.values.tolist(), max_display=33)
        # with st.beta_expander('Feature contributions:'):
        #     st.pyplot(fig_i)

                ########### feature importance
        # summary
        fig_i, ax = plt.subplots()
        shap.initjs()
        explainer = shap.TreeExplainer(model, model_output=0) # use model_output = 1 for scale trees
        if(len(target_permutation_df) > 100):
            target_permutation_df = target_permutation_df.sample(n=100, replace=True)
        shap_values = explainer.shap_values(target_permutation_df)
        shap_s = shap.summary_plot(shap_values, target_permutation_df, feature_names=input_f.columns.values.tolist(), max_display=33)
        
        # SHAP 值表格，命名columns
        shap_df = pd.DataFrame(shap_values)
        shap_df.columns = input_f.columns
        # st.write(shap_df)
        # shap.plots.heatmap(explainer(target_permutation_df))
        # if our unknown cols in reverse_cat_cols 
        cat_input_cols = []
        for each_col in permutation_col_name:
            if(each_col in reverse_cat_dict.keys()):
                cat_input_cols.append(each_col)
        # X_cat = target_permutation_df.copy()
        # relationship_decoding = {0: 'Detached', 1: 'Enclosed End-Terrace', 2: 'Enclosed Mid-Terrace', 3: 'End-Terrace', 4: 'Mid-Terrace', 5: 'Semi-Detached' }
        # X_cat["BUILT_FORM"] = X_cat["BUILT_FORM"].map(relationship_decoding)

        # st.write(X_cat.shape)
        with st.beta_expander('Feature contributions:'):
            st.set_option('deprecation.showPyplotGlobalUse', False)
            # SHOW SUMMARY PLOT
            st.pyplot(fig_i)
            # st.write(shap_df)
            df_estimation = pd.DataFrame()
            if(cat_input_cols != []):
                X_cat = target_permutation_df.copy()
                # decoding feature back to string
                for each_col in cat_input_cols:
                    X_cat[each_col] = X_cat[each_col].map(reverse_cat_dict[each_col])
                    # get each row's feature name
                    each_sample_df = X_cat[[each_col]]
                    each_sample_df = each_sample_df.reset_index()
                    # get shap value
                    each_sample_df[each_col+'_Impact'] = shap_df[each_col]
                    # st.write(each_sample_df)
                    # df_estimation = pd.concat([df_estimation, each_sample_df],ignore_index=True)
                    df_estimation[each_col] = each_sample_df[each_col]
                    df_estimation[each_col+'_Impact'] = each_sample_df[each_col+'_Impact']

                    

            df_impact = df_estimation.loc[:,['Impact' in i for i in df_estimation.columns]]
            for i in df_impact.columns:
                # category plot
                fig, ax = plt.subplots()
                plt.rc('xtick', labelsize=8)
                plt.rc('axes', titlesize=8)
                fig = shap.dependence_plot(i[:-7], shap_values, X_cat, interaction_index=None)
                plt.xticks(rotation=45, ha="right", size=8)
                plt.tight_layout()
                st.pyplot(fig)

                st.write(df_estimation[[i[:-7],i]].groupby(i[:-7]).mean())    
                # 记录shap贡献值，做成dataframe
                # st.write(reverse_cat_dict[each_col].values())
                # 去除非重复项
                # sampled_df = X_cat.loc[:, (X_cat != X_cat.iloc[0]).any()]
                # st.write(sampled_df)

                # 各个选项的潜在影响
                
                # st.write(df_estimation.loc[:,['Impact' in i for i in df_estimation.columns]].sum(axis=1))
        with st.beta_expander('Suggestions:'):
            # st.write('---')
            st.write('At the current design scheme:')

            for i in df_impact.columns:
                text = 'By setting **{}** --> **{}** would maximize the prediction around **{}** at the following potential scenario'.format(i[:-7], df_estimation.iloc[df_impact[i].idxmax()][i[:-7]], str(round(df_estimation.iloc[df_impact[i].idxmax()][i], 2)) )
                st.markdown(text)
                st.write(df_estimation.loc[df_impact[i].idxmax()])
            st.markdown('-')
            for i in df_impact.columns:
                text = 'By setting **{}** --> **{}** would minimize the prediction around **{}** at the following potential scenario'.format(i[:-7], df_estimation.iloc[df_impact[i].idxmin()][i[:-7]], str(round(df_estimation.iloc[df_impact[i].idxmin()][i], 2)) )
                st.markdown(text)
                st.write(df_estimation.loc[df_impact[i].idxmin()])

                # st.write()
                # st.write('To minimze the ')
                # st.write(i[:-7])
                # st.write(round(df_estimation.iloc[df_impact[i].idxmin()][i]))
                # st.write(df_estimation.iloc[df_impact[i].idxmin()][i[:-7]])
            # st.write(df_estimation.loc[df_estimation['Value'].idxmax()])
            df_estimation['Overall_Impact'] = df_estimation.loc[:,['Impact' in i for i in df_estimation.columns]].sum(axis=1)
            st.write('---')
            st.write('Overall estimation detail:')
            st.write(df_estimation)

                # shap_df = pd.DataFrame(shap_values)
                # shap_df.columns = X_cat.columns
                # shap_df = shap_df[[sampled_df.columns.values()]]
                # shap_df.columns = ['Impact']
                # shap_df[each_col] = reverse_cat_dict[each_col].values()
                # shap_df.sort_values('Impact', inplace=True)
                
                



def open_model_box(model, SUBJECT, input_f, go_point_prediction=True):

    # 做预测，返回对应的 点预测列，概率预测列，正态分布的均值，标准差
    # prediction
    y_preds = model.predict(input_f)
    y_dists = model.pred_dist(input_f)

    # plot final result distribution
    mu_list = []
    sigma_list = []
    for mu, sigma in zip(y_dists.params['loc'],y_dists.params['scale']):
        mu_list.append(mu)
        sigma_list.append(sigma)

    if(go_point_prediction is True):
        # present
        # st.write(y_preds)
        shap_d = point_prediction(SUBJECT, model, mu_list, sigma_list, y_dists, input_f)
        return shap_d

    else:
        return y_preds, y_dists, mu_list, sigma_list



# input data structure
df = pd.read_csv('D'+str(LOD_STAGE)+'.csv', index_col=0)
# input dataframe
input_f = df.iloc[0:1]
# st.write(input_f)
# input data updating
out_cat_dict, out_rate_dict, out_float_dict = user_input_features(cat_dict, rate_dict, status_dict)

# dict updating
for each_output_f in out_cat_dict:
    out_cat_dict[each_output_f] = cat_dict[each_output_f][out_cat_dict[each_output_f]]
    input_f[each_output_f] = out_cat_dict[each_output_f]

for each_output_f in out_rate_dict:
    out_rate_dict[each_output_f] = rate_dict[each_output_f][out_rate_dict[each_output_f]]
    input_f[each_output_f] = out_rate_dict[each_output_f]

for each_output_f in out_float_dict:
    input_f[each_output_f] = out_float_dict[each_output_f]


for each_input in TOTAL_LIST:
    if(status_dict[each_input] == False):
        input_f[each_input] = np.nan

st.write('Input preview:')
st.write(input_f)
# st.write('Input shape is: ' + str(input_f.shape))
# updating input dataframe
# for 
# st.write(input_f.columns.values.tolist())

model_CURR, model_CO2, model_Cost = load_model(LOD_STAGE)
if(input_f.isnull().values.sum() == 0):
    ################
    # prediction
    st.write('Prediction!')
    col1, col2, col3 = st.beta_columns(3)
    with col1:
        s1 = open_model_box(model_CURR, 'Energy (kWh/m²·year)', input_f)
    with col2:
        s2 = open_model_box(model_CO2, 'CO2 (kg/m²·year)', input_f)
    with col3:
        s3 = open_model_box(model_Cost, 'Cost (£/m²·year)', input_f)
    frames = [s1, s2, s3]
    result = pd.concat(frames)
    st.write(result)
else:
    ################
    # estimation
    st.write('Estimation!')
    # st.write(input_f.isnull().values.sum())
    col1, col2, col3 = st.beta_columns(3)
    with col1:
        estimation(model_CURR, 'Energy (kWh/m²·year)', input_f, cat_dict, warning=True)
    with col2:
        estimation(model_CO2, 'CO2 (kg/m²·year)', input_f, cat_dict)
    with col3:
        estimation(model_Cost, 'Cost (£/m²·year)', input_f, cat_dict)
    
    

with st.beta_expander('Encoding Dictionary：'):
    st.write('Explaination of features:')
    st.write('https://epc.opendatacommunities.org/docs/guidance')
    st.write(cat_dict)
