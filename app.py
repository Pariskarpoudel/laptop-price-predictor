import streamlit as st 
import pickle 
import numpy as np 

pipe = pickle.load(open('pipe.pkl', 'rb')) 
df = pickle.load(open('df.pkl', 'rb')) 

st.title('Laptop Price Prediction')

company = st.selectbox('Brand', df['Company'].unique())

type = st.selectbox('Type', df['TypeName'].unique())

ram = st.selectbox('RAM', np.sort(df['Ram'].unique()))



weight = st.number_input('Weight')

touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

ips = st.selectbox('IPS', ['No', 'Yes'])

# ppi ko lagi i need screensize(in inches) and resoluiton -> 1920X1080 -> xres and yres -> and so on

screen_size = st.number_input('Screen Size')

resolution = st.selectbox('Resolution', ['1920X1080', '1366X768', '1600X900', '2560X1440', '3840X2160', '3200X1800','2880X1800','2560X1600','2560X1440','2304X1440'])

cpu = st.selectbox('Cpu Brand', df['Cpu brand'].unique())

hdd = st.selectbox('HDD',[0,8,128,256,512,1024,2048])

ssd= st.selectbox('SSD',[0,8,128,256,512,1024])

gpu = st.selectbox('Gpu Brand', df['Gpu brand'].unique())

os = st.selectbox('OS', df['os'].unique())

if st.button('Predict'):
    # if this predict button is clicked, then do the following
   
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    xres = int(resolution.split('X')[0])
    yres = int(resolution.split('X')[1])

    ppi = (xres**2 + yres**2)**0.5/screen_size


    input_query = np.array([company, type, ram, weight, touchscreen, ips, ppi , cpu ,hdd , ssd , gpu , os])
    # xtrainko sample ma company, type, cpu haru categorical nai rahanxan, pipe vitrai preprocess hunxan , so inlai kei garna pardena 
    # but touchscreen ra ips chai 0,1 ma convert garnu parxa
    input_query = input_query.reshape(1,12)

    # you can give a 2D array as test data even if you trained the model using a DataFrame , or u can also convert it to df and give it to pipe 


    # hamle ytrain ma price lai log garera rakhethim to make it normal distribution, so esle pani log walai sano value print gariraxa 
    # so we need to take exponential of the predicted value to get the actual price
    st.title(int(np.exp((pipe.predict(input_query)[0]))))
