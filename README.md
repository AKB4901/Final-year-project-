# One-Stop Website for Farmers

## Overview
This project provides a comprehensive solution for farmers by offering the following features:
- **Crop Recommendation:** Suggests the best crop to grow based on specific soil and weather parameters.
- **Fertilizer Recommendation:** Recommends the appropriate fertilizer based on the crop and soil conditions.
- **Plant Leaf Diseases Detection:** Identifies diseases in plant leaves and suggests potential treatments.
- **Weather Forecast:** Provides weather forecasts for the next 10 days to help farmers plan their activities.

## Features

### Crop Recommendation
- **Algorithm:** Random Forest
- **Input Parameters:** Nitrogen, Phosphorous, Potassium, Temperature, Humidity, pH, Rainfall
- **Output:** Ideal crop recommendation

### Fertilizer Recommendation
- **Algorithm:** Random Forest
- **Input Parameters:** Crop type, soil nutrients, soil pH, and other relevant parameters
- **Output:** Recommended fertilizer

### Plant Leaf Diseases Detection
- **Algorithm:** Convolutional Neural Network (CNN) using VGG16 architecture
- **Input:** Image of the plant leaf
- **Output:** Disease diagnosis and suggested treatment

### Weather Forecast
- **API Used:** [Specify the Weather API used]
- **Output:** 10-day weather forecast

## Installation

### Prerequisites
- Python 3.8+
- Flask
- TensorFlow
- scikit-learn
- OpenCV
- requests

### Clone the Repository
```bash
git clone https://github.com/yourusername/one-stop-website-for-farmers.git
cd one-stop-website-for-farmers


**Install Dependencies**
```bash
pip install -r requirements.txt

### Usage
Run the Application
python app.py
Project Structure
.
├── app.py                # Main application file
├── requirements.txt      # Python dependencies
├── static
│   ├── css
│   └── images
├── templates
│   ├── index.html        # Home page
│   ├── crop_recommendation.html
│   ├── fertilizer_recommendation.html
│   ├── disease_detection.html
│   └── weather_forecast.html
└── models
    ├── crop_recommendation_model.pkl
    ├── fertilizer_recommendation_model.pkl
    └── disease_detection_model.h5
Model Training
Crop Recommendation Model
1.	Dataset: [Link to dataset]
2.	Algorithm: Random Forest
3.	Training Script: train_crop_recommendation.py
Fertilizer Recommendation Model
1.	Dataset: [Link to dataset]
2.	Algorithm: Random Forest
3.	Training Script: train_fertilizer_recommendation.py
Plant Leaf Diseases Detection Model
1.	Dataset: [Link to dataset]
2.	Algorithm: CNN with VGG16 architecture
3.	Training Script: train_disease_detection.py
Datasets
•	Crop Recommendation: Kaggle - Crop Recommendation Dataset
•	Fertilizer Recommendation: Kaggle - Fertilizer Prediction Dataset
•	Plant Diseases: Kaggle - Plant Diseases Dataset
Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or new features.
License
This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments
•	[Specify any resources, tools, or individuals that helped in the project]

