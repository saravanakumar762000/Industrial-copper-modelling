import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelBinarizer
import streamlit as st
import re
import warnings
import pickle
warnings.filterwarnings('ignore')
st.set_page_config(page_title= "Industrial Copper Modelling",
                   
                   layout= "wide",
                   initial_sidebar_state= "expanded",
                   menu_items={'About': """# This Project is created by *Saravana*!"""})
st.markdown("<h1 style='text-align: center; color: white;'>Industrial Copper Modelling</h1>", unsafe_allow_html=True)
tab1,tab2 = st.tabs(["Predict Selling price", "Predict Status"])
with tab1:
    status_options=['Won', 'Draft', 'To be approved', 'Lost', 'Not lost for AM','Wonderful', 'Revised', 'Offered', 'Offerable']
    item_options=['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR']
    country_options=[28.,  25.,  30.,  32.,  38.,  78.,  27.,  77., 113.,  79.,  26.,39.,  40.,  84.,  80., 107.,  89.]
    application_options=[10., 41., 28., 59., 15.,  4., 38., 56., 42., 26., 27., 19., 20.,66., 29., 22., 40., 25., 67., 79.,3., 99.,  2.,  5., 39., 69.,70., 65., 58., 68.]
    product_options=['611112', '611728', '628112', '628117', '628377', '640400', '640405', '640665', 
                     '611993', '929423819', '1282007633', '1332077137', '164141591', '164336407', 
                     '164337175', '1665572032', '1665572374', '1665584320', '1665584642', '1665584662', 
                     '1668701376', '1668701698', '1668701718', '1668701725', '1670798778', '1671863738', 
                     '1671876026', '1690738206', '1690738219', '1693867550', '1693867563', '1721130331', '1722207579']
    with st.form('form'):
        col1,col2,col3= st.columns([5,2,5])
        with col1:
            status=st.selectbox('Status',status_options,key=1)
            item_type=st.selectbox('Item Type',item_options,key=2)
            country=st.selectbox('Country',sorted(country_options),key=3)
            application=st.selectbox('Application',sorted(application_options),key=4)
            product=st.selectbox('Product',product_options,key=5)
        with col2:
            quantity_tons = st.text_input("Enter Quantity Tons (Min:611728 & Max:1722207579)")
            thickness = st.text_input("Enter thickness (Min:0.18 & Max:400)")
            width = st.text_input("Enter width (Min:1, Max:2990)")
            customer = st.text_input("customer ID (Min:12458, Max:30408185)")
            submit_button = st.form_submit_button(label="PREDICT SELLING PRICE")
            
            
            flag=0 
            pattern = "^(?:\d+|\d*\.\d+)$"
            for i in [quantity_tons,thickness,width,customer]:             
                if re.match(pattern, i):
                    pass
                else:                    
                    flag=1  
                    break
            
        if submit_button and flag==1:
            if len(i)==0:
                st.write("please enter a valid number")
            else:
                st.write("invalid value: ",i)  
        if submit_button and flag==0:
            with open(r"C:\Industrial copper modeling 1\model.pkl","rb") as f:
                model=pickle.load(f)
            with open(r"C:\Industrial copper modeling 1\scaler.pkl","rb") as f:
                scaler_model=pickle.load(f)
            with open(r"C:\Industrial copper modeling 1\a.pkl","rb") as f:
                a_model=pickle.load(f)
            with open(r"C:\Industrial copper modeling 1\b.pkl","rb") as f:
                b_model=pickle.load(f)
            new_sample= np.array([[np.log(float(quantity_tons)),application,np.log(float(thickness)),float(width),country,float(customer),int(product),item_type,status]])
            n_ohe = a_model.transform(new_sample[:, [7]]).toarray()
            n_ohe1 =b_model.transform(new_sample[:, [8]]).toarray()
            new_sample = np.concatenate((new_sample[:, [0, 1, 2, 3, 4, 5, 6]], n_ohe, n_ohe1), axis=1)
            new_sample1 = scaler_model.transform(new_sample)
            n_pred = model.predict(new_sample1)[0]
            st.write('## :green[Predicted selling price:] ', np.exp(n_pred))


with tab2:
    
    with st.form("my_form1"):
        col1,col2,col3=st.columns([5,1,5])
        with col1:
            quantity_tons2 = st.text_input("Enter Quantity Tons (Min:611728 & Max:1722207579)")
            thickness2 = st.text_input("Enter thickness (Min:0.18 & Max:400)")
            width2 = st.text_input("Enter width (Min:1, Max:2990)")
            customer2 = st.text_input("customer ID (Min:12458, Max:30408185)")
            selling2 = st.text_input("Selling Price (Min:1, Max:100001015)") 
        with col2:
            item_type2=st.selectbox('Item Type',item_options,key=21)
            country2=st.selectbox('Country',sorted(country_options),key=31)
            application2=st.selectbox('Application',sorted(application_options),key=41)
            product2=st.selectbox('Product',product_options,key=51)
            submit_button2 = st.form_submit_button(label="PREDICT STATUS")
            
            flag2=0 
            pattern = "^(?:\d+|\d*\.\d+)$"
            for j in [quantity_tons2,thickness2,width2,customer2]:             
                if re.match(pattern, j):
                    pass
                else:                    
                    flag2=1  
                    break
            
        if submit_button2 and flag2==1:
            if len(i)==0:
                st.write("please enter a valid number")
            else:
                st.write("invalid value: ",j)  
        if submit_button2 and flag2==0:
            with open(r"C:\Industrial copper modeling 1\dr.pkl","rb") as f:
                model2=pickle.load(f)
            with open(r"C:\Industrial copper modeling 1\df2scaler.pkl","rb") as f:
                scaler_model2=pickle.load(f)
            with open(r"C:\Industrial copper modeling 1\df2.pkl","rb") as f:
                df2_model=pickle.load(f)
            
            
            new_sample = np.array([[np.log(float(quantity_tons2)), np.log(float(selling2)), application2, np.log(float(thickness2)),float(width2),country2,int(customer2),int(product2),item_type2]])
            n_ohe = df2_model.transform(new_sample[:, [8]]).toarray()
            new_sample = np.concatenate((new_sample[:, [0,1,2, 3, 4, 5, 6,7]], n_ohe), axis=1)
            new_sample = scaler_model2.transform(new_sample)
            new_pred = model2.predict(new_sample)
            if new_pred==1:
                st.write('## :green[The Status is Won] ')
            else:
                st.write('## :red[The status is Lost] ')
st.write( f'<h6 style="color:rgb(0, 153, 153,0.35);">App Created by saravana</h6>', unsafe_allow_html=True )  

        
    
            
            
            
            
    


