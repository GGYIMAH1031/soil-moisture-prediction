import os
import numpy as np
import gradio as gr
import pandas as pd
import numpy as sklearn
import numpy as autogluon
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from autogluon.tabular import TabularDataset, TabularPredictor


import pandas as pd
from autogluon.tabular import TabularPredictor

# https://github.com/gradio-app/gradio/issues/3693#issuecomment-1745577523
#!mkdir -m 700 flagged


#model_dir="FinalModel-SoilMoisture"
#model_dir="./"

# Get the absolute path
#absolute_path = os.path.abspath(model_dir)

current_directory = os.getcwd()
current_directory_string = str(current_directory) + '/'


# Load the AutoGluon model
#model = TabularPredictor.load('./')

model = TabularPredictor.load(current_directory_string)
 

def predict(atm_pressure_kPa, precipitation_mm, Soil_conductivity_5cm_S000988, 
            radiation_W_m2, rel_humidity, Temp_2m_Celsius, windspeed_m_s):
    
    # Preprocess the model inputs
    atm_pressure_kPa= atm_pressure_kPa*1.0
    precipitation_mm= precipitation_mm*1.0 
    Soil_conductivity_5cm_S000988= Soil_conductivity_5cm_S000988*1.0 
    radiation_W_m2= radiation_W_m2*1.0 
    rel_humidity= rel_humidity*1.0 
    Temp_2m_Celsius=Temp_2m_Celsius*1.0
    windspeed_m_s= windspeed_m_s*1.0
    
    # Convert inputs into a pandas dataframe
    X_inputs={"atm_pressure_kPa":[atm_pressure_kPa], 
           "precipitation_mm":[precipitation_mm], 
           "Soil_conductivity_5cm_S000988":[Soil_conductivity_5cm_S000988], 
           "radiation_W_m2":[radiation_W_m2], 
           "rel_humidity":[rel_humidity],
           "Temp_2m_Celsius":[Temp_2m_Celsius],
           "windspeed_m_s":[windspeed_m_s]}
    
    X_inputs_df = pd.DataFrame(X_inputs)


    # Run inference using model inputs
    soil_moisture_5cm_S000988 = model.predict(X_inputs_df, model='WeightedEnsemble_L2')  
    
    #return the numeric portion of the model prediction
    return soil_moisture_5cm_S000988[0] 


demo = gr.Interface(
    fn=predict,
    inputs=[gr.Number(value=98.81, label="Atmospheric Pressure (kPa)"), 
            gr.Number(value=0, label="Precipitation (mm)"),
            gr.Number(value=0.069, label="Soil Conductivity at 5cm Depth"),
            gr.Number(value=223.0, label="Radiation (w/m^2)"),
            gr.Slider(0, 1, value=0.541, label="Relative Humidity"),
            gr.Number(value=34.7, label="Temperature at 2m Height (C)"),
            gr.Number(value=1.18, label="Windspeed (m/s)"),
           ],
    outputs=[gr.Number(value=0.22719, label="Predicted Soil Moisture at 5cm Depth")],
    live=True,
    title = "Predicting Soil Moisture at 5cm Depth with Weather Data",
    #thumbnail="./FinalModel-SoilMoisture/moisture-from-weather-data.png"
)

if __name__=='__main__':
    demo.launch(share=True)
