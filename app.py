from flask import Flask, render_template, request, redirect, url_for, session
import sqlite3
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from datetime import datetime
from flask import Flask, render_template, request
import os
import cv2
import requests
from joblib import load
from tensorflow.keras.models import load_model

import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__, static_folder='static')

app.secret_key = '5766ghghgg7654dfd7h9hsffh'

UPLOAD_FOLDER = 'static/uploaded_images'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def home():
    return render_template('landing.html')


## Authentication steps

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':

        name = request.form['name']
        address = request.form['address']
        email = request.form['email']
        contact = request.form['contact']
        age = request.form['age']
        password = request.form['password']
        re_password = request.form['re_password']

        # Check if email already exists in the database
        conn = sqlite3.connect(
            'database/Farmer.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM user_details WHERE email = ?", (email,))
        user = cursor.fetchone()
        conn.close()

        if user is not None:
            # If the user already exists, add a flash message and redirect back to the signup page
            session['message'] = 'email already exist. Please go to login page.'

            return redirect(url_for('signup', error='email already exist.'))

        elif password != re_password:

            session['message'] = 'Both password are different.'

            return redirect(url_for('signup', error='password do not match.'))

        else:
        
            conn = sqlite3.connect(
                'database/Farmer.db')
            cursor = conn.cursor()
            cursor.execute("INSERT INTO user_details (name, address, email, contact, age, password, re_password) VALUES (?, ?, ?, ?, ?, ?, ?)",
                           (name, address, email, contact, age, password, re_password))
            conn.commit()
            conn.close()

            return redirect(url_for('login'))

    elif request.args.get('error') is None:
        return render_template('signup.html')

    else:
        error = request.args.get('error')
        return render_template('signup.html', error=error)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':

        email = request.form['email']
        password = request.form['password']

        conn = sqlite3.connect('database/Farmer.db')
        c = conn.cursor()

        c.execute(
            "SELECT * FROM user_details WHERE email = ? AND password = ?", (email, password))
        user = c.fetchone()
        conn.close()

        if user is not None:
            session['email'] = user[1]
            
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='Invalid email or password')
    else:
        return render_template('login.html')




## Page Redirect Steps

@app.route('/index')
def index():

    email = session.get('email')
    print(email)
    conn = sqlite3.connect('database/Farmer.db')
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM user_details WHERE email = ?", (email,))
    user = cursor.fetchone()
    conn.close()

    print(user)


    if 'email' in session:
        return render_template('index.html', current_user=session['email'], user=user[0])
    return redirect(url_for('login'))


@app.route('/contactus', methods=['GET', 'POST'])
def contactus():

    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        subject = request.form['subject']
        message = request.form['message']

        conn = sqlite3.connect(
            'database/Farmer.db')
        cursor = conn.cursor()
        cursor.execute("INSERT INTO user_query (name, email, subject, message) VALUES (?, ?, ?, ?)",
                       (name, email, subject, message))
        conn.commit()
        conn.close()

        message = "We have received your response, Our team will contact you shortly."

        return render_template('contactus.html',  message = message)

    return render_template('contactus.html')

@app.route('/logout')
def logout():
    # Clear session data
    session.clear()
    # Redirect to the login page
    return redirect(url_for('login'))



### prediction Steps

# List of major cities in India
indian_cities = [
    "Jaipur", "Mumbai", "Delhi", "Kolkata", "Chennai", 
    "Bangalore", "Hyderabad", "Pune", "Ahmedabad", "Surat", 
    "Lucknow", "Nagpur", "Indore", "Bhopal", "Patna", 
    "Vadodara", "Ghaziabad", "Ludhiana", "Agra", "Varanasi",
    "Ranchi", "Kanpur", "Nashik", "Coimbatore", "Kochi", 
    "Visakhapatnam", "Thiruvananthapuram", "Amritsar", "Vijayawada", "Guwahati", 
    "Navi Mumbai", "Thane", "Bhubaneswar", "Dehradun", "Bikaner", 
    "Jodhpur", "Rajkot", "Shimla", "Srinagar", "Jammu"
]


@app.route('/weather', methods=['GET', 'POST'])
def weather():
    if request.method == 'POST':

        temperature_celsius = None
        weather_condition = None
        
        city_name = request.form['city']
        date = request.form['date']

        provided_date = datetime.strptime(date, '%Y-%m-%d').date()

        # Get today's date
        today_date = datetime.now().date()

        # Convert today's date to datetime for subtraction
        today_datetime = datetime.combine(today_date, datetime.min.time())

        # Calculate the difference
        date_difference = provided_date - today_datetime.date()

        # Convert the date difference to days
        n_days = date_difference.days
   
        # API endpoint URL
        url = "http://api.weatherapi.com/v1/forecast.json"

        # Validate user input
        print(city_name)
        print("----------------------------------")
        # if city_name.lower() in indian_cities.lower():
        if city_name.lower() in [city.lower() for city in indian_cities]:
            
            try:
                n_days = int(n_days)
                if n_days < 1:
                    raise ValueError("Please enter a positive integer value for the number of days.")
            except ValueError as e:
                print("Invalid input:", e)
            else:
                # Parameters
                params = {
                    "key": "627b10c8fa8b4ff28c190325242904",
                    "q": city_name + ", India",  # Specify the selected city name and country
                    "days": n_days + 1,  # Retrieve forecast data for n_days plus 1 to get the forecast for after n_days
                    "aqi": "no"  # Exclude air quality data if not needed
                }

                # Send GET request to the API
                response = requests.get(url, params=params)

                # Check if request was successful (status code 200)
                if response.status_code == 200:
                    # Parse JSON response
                    data = response.json()
                    
                    # Extract forecast data for after n_days
                    forecast_after_n_days = data['forecast']['forecastday'][n_days]

                    # Extract relevant weather information
                    date = forecast_after_n_days['date']
                    temperature_celsius = forecast_after_n_days['day']['avgtemp_c']
                    weather_condition = forecast_after_n_days['day']['condition']['text']

                    # Print the weather information for after n_days
                    print(f'\nWeather forecast for {city_name} after {n_days} days ({date}):')
                    print(f'Temperature: {temperature_celsius}°C')
                    print(f'Weather Condition: {weather_condition}')

                    message = f"Weather forecast for {city_name} after {n_days} days i.e, {provided_date}: Temperature: {temperature_celsius}°C, Weather Condition: {weather_condition}"


                    return render_template('weather.html', indian_cities=indian_cities, message = message)
                
                else:
                    print('Error:', response.status_code)

                    return render_template('weather.html', message=response.status_code)
        else:
            print("Invalid city selection. Please enter a valid city name.")
            message = "Invalid city selection. Please enter a valid city name."
            return render_template('weather.html', message = message)

            
    return render_template('weather.html')



### Crop Recommendation

label_mapping = {
            0: 'apple', 1: 'banana', 2: 'blackgram', 3: 'chickpea', 4: 'coconut', 5: 'coffee', 
            6: 'cotton', 7: 'grapes', 8: 'jute', 9: 'kidneybeans', 10: 'lentil', 11: 'maize', 
            12: 'mango', 13: 'mothbeans', 14: 'mungbean', 15: 'muskmelon', 16: 'orange', 
            17: 'papaya', 18: 'pigeonpeas', 19: 'pomegranate', 20: 'rice', 21: 'watermelon'
        }


image_urls = {
    'apple': "static/apple.jpg",
    'banana': 'static/banana.jpg',
    'blackgram': 'static/blackgram.webp',
    'chickpea': 'static/chickpea.jpg',
    'coconut': 'static/coconut.jpg',
    'coffee': 'static/coffee.jpg',
    'cotton': 'static/cotton.jpg',
    'grapes': 'static/grapes.jpg',
    'jute': 'static/jute.jpg',
    'kidneybeans': 'static/kidneybeans.jpg',
    'lentil': 'static/lentil.jpg',
    'maize': 'static/maize.jpg',
    'mango': 'static/mango.jpg',
    'mothbeans': 'static/mothbeans.jpg',
    'mungbean': 'static/mungbean.jpg',
    'muskmelon': 'static/muskmelon.jpg',
    'orange': 'static/orange.jpg',
    'papaya': 'static/papaya.avif',
    'pigeonpeas': 'static/pigeonpeas.jpg',
    'pomegranate': 'static/pomegranate.jpg',
    'rice': 'static/rice.jpg',
    'watermelon': 'static/watermelon.jpg'
}
    

@app.route('/crop_recommendation', methods=['GET', 'POST'])
def crop_recommendation():
    '''
    For rendering results on HTML GUI
    '''
    if request.method == 'POST':

        nitrogen = request.form['nitrogen']
        phosphorous = request.form['phosphorous']
        potassium = request.form['potassium']
        temperature = request.form['temperature']
        humidity = request.form['humidity']
        ph = request.form['ph']
        rainfall = request.form['rainfall']
        
        loaded_model = load('../models/model.joblib')

        recommended_crop = loaded_model.predict([[nitrogen, phosphorous, potassium, temperature, humidity, ph, rainfall]])

        predicted_label = label_mapping[recommended_crop[0]]


         # Get the image URL based on the predicted label
        image_url = image_urls.get(predicted_label)

        return render_template('crop_recommendation.html', predicted_label=predicted_label, img=image_url)

    return render_template('crop_recommendation.html')

@app.route('/crop_recommendation_default', methods=['GET', 'POST'])
def crop_recommendation_default():
    '''
    For rendering results on HTML GUI
    '''
    predicted_label = None

    if request.method == 'POST':

        nitrogen = request.form['nitrogen']
        phosphorous = request.form['phosphorous']
        potassium = request.form['potassium']
        temperature = request.form['temperature']
        humidity = request.form['humidity']
        ph = request.form['ph']
        rainfall = request.form['rainfall']
        
        loaded_model = load('../models/model.joblib')

        recommended_crop = loaded_model.predict([[nitrogen, phosphorous, potassium, temperature, humidity, ph, rainfall]])

        predicted_label = label_mapping[recommended_crop[0]]

        # Get the image URL based on the predicted label
        image_url = image_urls.get(predicted_label)

        return render_template('crop_recommendation_default.html', predicted_label=predicted_label, img=image_url)
    return render_template('crop_recommendation_default.html')



label_mapping = {
            0: 'apple', 1: 'banana', 2: 'blackgram', 3: 'chickpea', 4: 'coconut', 5: 'coffee', 
            6: 'cotton', 7: 'grapes', 8: 'jute', 9: 'kidneybeans', 10: 'lentil', 11: 'maize', 
            12: 'mango', 13: 'mothbeans', 14: 'mungbean', 15: 'muskmelon', 16: 'orange', 
            17: 'papaya', 18: 'pigeonpeas', 19: 'pomegranate', 20: 'rice', 21: 'watermelon'
        }


model_urls = {
    'apple': "../models/apple.keras",
    'corn': "../models/corn.keras",
    'peach': "../models/peach.keras",
    'cherry': "../models/cherry.keras",
    'grape': "../models/grape.keras",
    'pepper': "../models/pepper.keras",
    'potato': "../models/potato.keras",
    'rice': "../models/rice.keras",
    'soyabean': "../models/soyabean.keras",
    'sugarcane': "../models/sugarcane.keras",
    'tomato': "../models/tomato.keras",
    'wheat': "../models/wheat.keras" 
}

img_size = 224

def preprocess_image(img, img_size):
    # Resize the image
    img = cv2.resize(img, (img_size, img_size))
    # Convert to grayscale if needed
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Normalize pixel values
    img = gray.astype('float32') / 255.0
    # Add batch and channel dimensions
    img = np.expand_dims(img, axis=(0, -1))
    return img

@app.route('/leaf', methods=['GET', 'POST'])
def leaf():
    '''
    For rendering results on HTML GUI
    '''

    categories = None
    label_dict = None
    loaded_model = None

    if request.method == 'POST':

        leaf = request.form['leaf']
        file = request.files['image']

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'img.jpg')
        file.save(file_path)

        if leaf == 'apple':
            categories = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy']
            
        elif leaf == 'sugarcane':
            categories = ['Healthy', 'Mosaic', 'RedRot', 'Rust', 'Yellow']

        elif leaf == 'cherry':
            categories = ['Cherry_(including_sour)___healthy', 'Cherry_(including_sour)___Powdery_mildew']

        elif leaf == 'corn':
            categories = ['Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___healthy', 'Corn_(maize)___Northern_Leaf_Blight']
        
        elif leaf == 'grape':
            categories = ['Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___healthy', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)']
        
        elif leaf == 'peach':
            categories = ['Peach___Bacterial_spot', 'Peach___healthy']
        
        elif leaf == 'pepper':
            categories = ['Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy']

        elif leaf == 'potato':
            categories = ['Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight']

        elif leaf == 'rice':
            categories = ['Bacterialblight', 'Blast', 'Brownspot', 'Tungro']

        elif leaf == 'soyabean':
            categories = ['Caterpillar', 'Diabrotica speciosa', 'Healthy']

        elif leaf == 'strawberry':
            categories = ['Strawberry___healthy', 'Strawberry___Leaf_scorch']

        elif leaf == 'tomato':
            categories = ['Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___healthy', 'Tomato___Late_blight']

        elif leaf == 'wheat':
            categories = ['Wheat___Brown_Rust', 'Wheat___Healthy', 'Wheat___Yellow_Rust']
        

        loaded_model = load_model(model_urls.get(leaf))
        label_dict = {i: category for i, category in enumerate(categories)}

         # Read the input image (grayscale)
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        # Convert grayscale image to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Resize the image to img_size x img_size
        resized_img = cv2.resize(img_rgb, (img_size, img_size))

        # Normalize pixel values to range [0, 1]
        normalized_img = resized_img / 255.0

        # Reshape input for model prediction
        input_img = normalized_img.reshape(-1, img_size, img_size, 3)  # Reshape input for model prediction

        # Make a prediction
        prediction = loaded_model.predict(input_img)

        # Get the predicted class label
        predicted_class_index = np.argmax(prediction)
        predicted_label = label_dict[predicted_class_index]

        return render_template('leaf.html', predicted_label=predicted_label, img=file_path)

    return render_template('leaf.html')


crop_labels = {
'Maize' : 0,
'Sugarcane': 1,
'Cotton': 2,
'Tobacco': 3,
'Paddy': 4,
'Barley': 5,
'Wheat': 6,
'Millets': 7,
'Oil seeds': 8,
'Pulses': 9,
'Ground Nuts': 10
}

fertilizers_labels = ['Urea','DAP','14-35-14','28-28','17-17-17', '20-20','10-26-26']


@app.route('/fertilizer', methods=['GET', 'POST'])
def fertilizer():
    '''
    For rendering results on HTML GUI
    '''
    if request.method == 'POST':

        Temperature = request.form['temperature']
        Humidity = request.form['humidity']
        Moisture = request.form['moisture']
        Crop = request.form['crop']
        Nitrogen = request.form['nitrogen']
        Potasium = request.form['potasium']
        Phosphorous = request.form['phosphorous']
        
        loaded_model = load('../models/fertilizer.joblib')


        Crop_Type = crop_labels[Crop]

        def recommend_fertilizer(Temperature, Humidity, Moisture, Crop_Type, Nitrogen, Potasium, Phosphorous):
            
            prediction = loaded_model.predict([[Temperature, Humidity, Moisture, Crop_Type, Nitrogen, Potasium, Phosphorous]])
            return prediction[0]

        # Get the recommended crop
        recommended_fertilizer = recommend_fertilizer(Temperature, Humidity, Moisture, Crop_Type, Nitrogen, Potasium, Phosphorous)
        
        return render_template('fertilizer.html', predicted_label=fertilizers_labels[recommended_fertilizer])

    return render_template('fertilizer.html')

if __name__ == "__main__":
    app.run(debug=True)
