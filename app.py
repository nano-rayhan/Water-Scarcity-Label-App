import pandas as pd
import numpy as np
import pickle
import gradio as gr

with open('water.pkl', 'rb') as f:
    model = pickle.load(f)


def predict(Country, Year, TotalWaterconsumption ,
       PerCapitaWaterUse, AgriculturalWaterUse,
       IndustrialWaterUse, HouseholdWaterUse ,
       RainfallImpact , GroundwaterDepletionRate):
     
    input_df = pd.DataFrame([[Country, Year, TotalWaterconsumption ,
       PerCapitaWaterUse, AgriculturalWaterUse,
       IndustrialWaterUse, HouseholdWaterUse ,
       RainfallImpact , GroundwaterDepletionRate]],
       columns=['Country', 'Year', 'Total Water Consumption (Billion m3)',
       'Per Capita Water Use (L/Day)', 'Agricultural Water Use (%)',
       'Industrial Water Use (%)', 'Household Water Use (%)',
       'Rainfall Impact (mm)', 'Groundwater Depletion Rate (%)'])
    
    
    predicted = model.predict(input_df)[0]
    
    
    labels = {0:"Low",1:"Moderate",2:"High",3:"Critical"}
    return f"Water Scarcity Level: {labels[predicted]}"


countries = ['China', 'India', 'USA', 'Indonesia', 'Pakistan', 'Brazil',
       'Nigeria', 'Bangladesh', 'Russia', 'Mexico', 'Japan', 'Ethiopia',
       'Philippines', 'Egypt', 'Vietnam', 'DR Congo', 'Turkey', 'Iran',
       'Germany', 'Thailand', 'United Kingdom', 'France', 'Italy',
       'Tanzania', 'South Africa', 'Myanmar', 'Kenya', 'South Korea',
       'Colombia', 'Spain', 'Uganda', 'Argentina', 'Algeria', 'Sudan',
       'Ukraine', 'Iraq', 'Afghanistan', 'Poland', 'Canada', 'Morocco',
       'Saudi Arabia', 'Uzbekistan', 'Malaysia', 'Peru', 'Angola',
       'Ghana', 'Mozambique', 'Yemen', 'Nepal', 'Venezuela', 'Madagascar',
       'Cameroon', "Côte d'Ivoire", 'North Korea', 'Australia', 'Niger',
       'Taiwan', 'Sri Lanka', 'Burkina Faso', 'Mali', 'Romania', 'Malawi',
       'Chile', 'Kazakhstan', 'Zambia', 'Guatemala', 'Ecuador', 'Syria',
       'Netherlands', 'Senegal', 'Cambodia', 'Chad', 'Somalia',
       'Zimbabwe', 'Guinea', 'Rwanda', 'Benin', 'Burundi', 'Tunisia',
       'Bolivia', 'Belgium', 'Haiti', 'Cuba', 'South Sudan',
       'Dominican Republic', 'Czech Republic', 'Greece', 'Jordan',
       'Portugal', 'Azerbaijan', 'Sweden', 'Honduras',
       'United Arab Emirates', 'Hungary', 'Tajikistan', 'Belarus',
       'Austria', 'Papua New Guinea', 'Serbia', 'Israel', 'Switzerland',
       'Togo', 'Sierra Leone', 'Hong Kong', 'Laos', 'Paraguay',
       'Bulgaria', 'Libya', 'Lebanon', 'Nicaragua', 'Kyrgyzstan',
       'El Salvador', 'Turkmenistan', 'Singapore', 'Denmark', 'Finland',
       'Slovakia', 'Norway', 'Oman', 'State of Palestine', 'Costa Rica',
       'Liberia', 'Ireland', 'Central African Republic', 'New Zealand',
       'Mauritania', 'Panama', 'Kuwait', 'Croatia', 'Moldova', 'Georgia',
       'Eritrea', 'Uruguay', 'Bosnia and Herzegovina', 'Mongolia',
       'Armenia', 'Jamaica', 'Qatar', 'Albania', 'Lithuania', 'Namibia',
       'Gambia', 'Botswana', 'Gabon', 'Lesotho', 'North Macedonia',
       'Slovenia', 'Guinea-Bissau', 'Latvia', 'Bahrain']
inputs = [
    gr.Dropdown(choices=countries, label="Country", filterable=True),
    gr.Number(label= 'Year'),
    gr.Number(label='Total Water Consumption (Billion m3)'),
    gr.Number(label='Per Capita Water Use (L/Day)'),
    gr.Number(label='Agricultural Water Use (%)'),
    gr.Number(label='Industrial Water Use (%)'),
    gr.Number(label='Household Water Use (%)'),
    gr.Number(label='Rainfall Impact (mm)'),
    gr.Number(label='Groundwater Depletion Rate (%)')
]


app = gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs='text',
    title='Water scarcity label'
)

app.launch(share=True)
