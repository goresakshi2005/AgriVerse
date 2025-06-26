import os
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
import requests
import ee
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.tavily import TavilyTools
from phi.tools.pubmed import PubmedTools

# ---------- Initialize Environment ----------
load_dotenv()

# Initialize geocoder
geolocator = Nominatim(user_agent="plant_health_monitor")

# Configure Gemini Models
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
image_model = genai.GenerativeModel('gemini-1.5-flash')
research_model = Gemini(api_key=os.getenv("GOOGLE_API_KEY"))

# Setup Research Tools
tools = [TavilyTools(api_key=os.getenv("TAVILY_API_KEY")), PubmedTools()]
research_agent = Agent(tools=tools, model=research_model)

# ---------- Crop Health Functions ----------
def get_coordinates(location_name):
    """Convert location name to coordinates and create a bounding box"""
    try:
        location = geolocator.geocode(location_name)
        if not location:
            raise ValueError("Location not found")
        
        # Create a ~20km × 20km area around the point
        lat, lon = location.latitude, location.longitude
        delta = 0.05  # ~5km in degrees
        return ee.Geometry.Rectangle([
            lon - delta, lat - delta, 
            lon + delta, lat + delta
        ]), lat, lon
    except (GeocoderTimedOut, GeocoderUnavailable) as e:
        print(f"⚠️ Geocoding service unavailable. Using default coordinates.")
        # Return default coordinates (center of Ahmednagar) if geocoding fails
        return ee.Geometry.Rectangle([74.47, 19.07, 74.57, 19.17]), 19.12, 74.52
    except Exception as e:
        print(f"⚠️ Geocoding error: {str(e)}")
        raise

def initialize_earth_engine():
    """Initialize Google Earth Engine"""
    try:
        ee.Initialize(project='hackathon-457607')
    except Exception as e:
        try:
            ee.Authenticate(auth_mode="notebook")
            ee.Initialize(project='hackathon-457607')
        except Exception as auth_error:
            print(f"❌ Earth Engine authentication failed: {str(auth_error)}")
            return False
    return True

def get_weather(location_name, lat, lon):
    """Get weather data for the specified location"""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return {
            "error": "API key not found in .env file",
            "data": None
        }

    # Try with exact location name first, then fall back to coordinates
    for query in [location_name, f"{lat},{lon}"]:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={query}&appid={api_key}&units=metric"
        
        try:
            response = requests.get(url, timeout=5)
            data = response.json()

            if response.status_code == 401:
                return {
                    "error": "Invalid API Key (Unauthorized)",
                    "data": None
                }

            if data.get("cod") == 200:
                return {
                    "error": None,
                    "data": {
                        "temperature": data["main"]["temp"],
                        "humidity": data["main"]["humidity"],
                        "rainfall": data.get("rain", {}).get("1h", 0),
                        "cloudiness": data.get("clouds", {}).get("all", 0),
                        "weather_desc": data["weather"][0]["description"] if data.get("weather") else "N/A"
                    }
                }
            elif query == f"{lat},{lon}":  # If both attempts failed
                return {
                    "error": f"Weather API error: {data.get('message', 'Unknown error')}",
                    "data": None
                }

        except Exception as e:
            if query == f"{lat},{lon}":  # If both attempts failed
                return {
                    "error": f"Weather request failed: {str(e)}",
                    "data": None
                }

def get_soil_data(aoi):
    """Get soil moisture and pH data from Earth Engine"""
    datasets = [
        {
            "name": "Soil Moisture",
            "collection": "NASA/SMAP/SPL4SMGP/007",
            "date_range": ('2023-01-01', '2023-01-05'),  # Using static date for demo
            "band": 'sm_surface',
            "reducer": 'mean',
            "scale": 1000
        },
        {
            "name": "Soil pH",
            "collection": "OpenLandMap/SOL/SOL_PH-H2O_USDA-4C1A2A_M/v02",
            "date_range": None,
            "band": 'b0',
            "scale": 500,
            "divide_by": 10  # New field to specify we need to divide pH values by 10
        }
    ]
    
    results = {}
    
    for ds in datasets:
        try:
            if ds['date_range']:
                col = ee.ImageCollection(ds['collection']) \
                      .filterDate(ds['date_range'][0], ds['date_range'][1]) \
                      .select(ds['band'])
                
                if ds.get('reducer') == 'mean':
                    img = col.mean().clip(aoi)
                elif ds.get('reducer') == 'sum':
                    img = col.sum().clip(aoi)
                else:
                    img = col.first().clip(aoi)
            else:
                img = ee.Image(ds['collection']).select([ds['band']]).clip(aoi)
            
            stats = img.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=aoi,
                scale=ds['scale'],
                maxPixels=1e9,
                bestEffort=True
            )
            
            stats_info = stats.getInfo()
            if stats_info:
                value = list(stats_info.values())[0]
                
                # Convert soil moisture from m³/m³ to percentage
                if ds['name'] == "Soil Moisture":
                    results["soil_moisture"] = round(value * 100, 2)
                # For pH, divide by 10 if the dataset specifies it
                elif ds['name'] == "Soil pH":
                    ph_value = value / ds.get('divide_by', 1)  # Divide by 10 if specified
                    results["soil_ph"] = round(ph_value, 2)
            
        except Exception as e:
            print(f"⚠️ Failed to process {ds['name']}: {str(e)}")
            if ds['name'] == "Soil Moisture":
                results["soil_moisture"] = None
            else:
                results["soil_ph"] = None
    
    return results

def get_weather_soil_data(location_name):
    """Get combined weather and soil data"""
    # Initialize Earth Engine
    if not initialize_earth_engine():
        return None
    
    # Get coordinates
    try:
        aoi, lat, lon = get_coordinates(location_name)
    except Exception as e:
        print(f"❌ Failed to get coordinates for location: {str(e)}")
        return None
    
    # Get weather data
    weather_result = get_weather(location_name, lat, lon)
    if weather_result["error"]:
        print(f"⚠️ Weather data unavailable: {weather_result['error']}")
        weather_data = None
    else:
        weather_data = weather_result["data"]
    
    # Get soil data
    soil_data = get_soil_data(aoi)
    
    # Combine data
    combined_data = {}
    if weather_data:
        combined_data.update(weather_data)
    if soil_data:
        combined_data.update(soil_data)
    
    return combined_data if combined_data else None

# ---------- Disease Detection Functions ----------
def analyze_plant_image(img_path):
    """Analyze plant image and extract symptoms"""
    try:
        img = Image.open(img_path)
    except FileNotFoundError:
        print(f"Error: Could not find image at {img_path}")
        return None

    prompt = """
    Analyze this plant image and provide ONLY the following details in a clear, bullet-point format:
    - **Plant Name** (if identifiable)
    - **Growth Stage** (e.g., seedling, vegetative, flowering, fruiting)
    - **Visible Symptoms/Issues** (e.g., curling, spots, discoloration)
    - **Possible Causes** (e.g., fungal, bacterial, nutrient deficiency)

    Do NOT include treatment recommendations or unrelated explanations.
    """
    
    try:
        response = image_model.generate_content([prompt, img])
        return response.text
    except Exception as e:
        print(f"Error generating response: {e}")
        return None

def research_disease(symptoms, weather_soil_data=None):
    """Get disease information based on symptoms and environmental conditions"""
    env_context = ""
    if weather_soil_data:
        env_context = f"""
        🌤️ Environmental Conditions:
        - Temperature: {weather_soil_data.get('temperature', 'N/A')}°C
        - Humidity: {weather_soil_data.get('humidity', 'N/A')}%
        - Rainfall: {weather_soil_data.get('rainfall', 0)}mm
        - Cloudiness: {weather_soil_data.get('cloudiness', 'N/A')}%
        
        🌱 Soil Conditions:
        - Moisture: {weather_soil_data.get('soil_moisture', 'N/A')}%
        - pH: {weather_soil_data.get('soil_ph', 'N/A')}
        """
    
    prompt = f"""
    You are an expert agronomist with 20+ years of experience in organic farming and plant pathology. 
    Analyze this complete agricultural scenario & use Weather and Soil data to provide a comprehensive plant health report.

    🌿 PLANT PROFILE:
    {symptoms}

    🌍 ENVIRONMENTAL CONTEXT:
    {env_context if env_context else "⚠️ No environmental data available"}

    📝 YOUR COMPREHENSIVE ANALYSIS GUIDE:

    1. CURRENT STAGE DIAGNOSIS:
    - Confirm exact growth stage (seedling/vegetative/flowering/fruiting)
    - List ALL observable symptoms with possible causes (rank by probability)
    - Highlight any immediate threats requiring urgent action

    2. ORGANIC TREATMENT PLAN:
    A. For diagnosed issues:
    - Recommend 3 homemade remedies (with preparation instructions)
    Example: "Neem oil spray: Mix 5ml neem + 3ml soap in 1L water, spray every 3 days"
    - Suggest commercial organic products (with local availability considerations)

    B. Soil enhancement:
    - Organic amendments needed (compost/vermicompost/FYM ratios)
    - Biofertilizers recommendation (Azospirillum/PSB dosage)
    - pH correction methods (if needed)

    3. STAGE-SPECIFIC CARE GUIDE:
    - Current stage maintenance checklist:
    * Ideal watering schedule
    * Pruning requirements
    * Support structures needed
    - Preparation for next growth stage:
    * Nutrients to emphasize
    * Common mistakes to avoid
    * Expected timeline

    4. PREVENTIVE MEASURES:
    - Companion planting suggestions
    - Natural pest deterrent plants
    - Weekly monitoring indicators

    5. FERTILIZATION SCHEDULE:
    - Organic fertilizer calendar (with quantities)
    - Foliar spray recommendations
    - Critical nutrient timings

    6. RISK MANAGEMENT:
    - Weather contingency plans
    - Early warning signs to watch for
    - Emergency protocols for disease outbreaks

    📌 DELIVERY REQUIREMENTS:
    - For all above recommendations, keep weather and soil data like temperature, humidity, rainfall, cloudiness, soil ph & moisture in mind
    - Use simple language (8th grade level), easy to understand, avoid jargon
    - Provide clear, actionable steps
    - Provide measurements in local units (kg, liters, etc.)
    - Include cost-effective solutions
    - Specify application frequency/dosage precisely
    - Mention safety precautions for all treatments
    - Don't use bullets/stars) and use emojis
    
    Example format:
    -------------------------------------------
    🌱 CURRENT STAGE: Flowering (Week 3)

    ⚠️ URGENT ISSUES: 
    - Aphid infestation (heavy)
    - Early blight symptoms (moderate)

    🧪 HOMEMADE TREATMENT:
    1. Garlic-chili spray: 
    - Crush 10 garlic + 5 chilies in 1L water
    - Strain and spray every morning for 5 days
    - Cost: ~₹20 per application

    2. ...

    💧 WATERING SCHEDULE:
    - Morning: 2L per plant (6-7AM)
    - Evening: 1L per plant (4-5PM)
    - Adjust if rainfall >10mm

    -------------------------------------------
"""

    print("\n🤖 Gemini AI Agent is analyzing with environmental context...\n")
    response = research_agent.run(prompt)
    return response.content if response else None

# ---------- Main Integrated Function ----------
def generate_plant_health_report(location_name, img_path=None, manual_symptoms=None):
    """Generate comprehensive plant health report"""
    print("\n" + "="*50)
    print(f"🌿 Comprehensive Plant Health Report for: {location_name}")
    print("="*50 + "\n")
    
    # Get environmental data
    weather_soil_data = get_weather_soil_data(location_name)
    
    # Print weather and soil data
    if weather_soil_data:
        print("\n🌦️ Weather Conditions:")
        print(f"- Temperature: {weather_soil_data.get('temperature', 'N/A')}°C")
        print(f"- Humidity: {weather_soil_data.get('humidity', 'N/A')}%")
        print(f"- Rainfall: {weather_soil_data.get('rainfall', 0)}mm")
        print(f"- Cloudiness: {weather_soil_data.get('cloudiness', 'N/A')}%")
        print(f"- Weather Description: {weather_soil_data.get('weather_desc', 'N/A')}")
        
        print("\n🌱 Soil Conditions:")
        print(f"- Moisture: {weather_soil_data.get('soil_moisture', 'N/A')}%")
        print(f"- pH: {weather_soil_data.get('soil_ph', 'N/A')}")
    else:
        print("\n⚠️ Could not retrieve weather and soil data")
    
    # Get plant symptoms either from image or manual input
    if img_path:
        print("\n🔍 Analyzing plant image...")
        image_analysis = analyze_plant_image(img_path)
        
        if image_analysis:
            print("\n📋 Image Analysis Results:")
            print(image_analysis)
            
            # Extract symptoms from analysis for research
            symptoms = input("\n✏️ Please confirm or add to the symptoms from the image (press Enter to use analysis): ").strip()
            if not symptoms:
                symptoms = image_analysis
        else:
            symptoms = input("\n✏️ Please describe the plant's symptoms (e.g., yellow leaves on tomato): ").strip()
    else:
        symptoms = manual_symptoms if manual_symptoms else input("\n✏️ Please describe the plant's symptoms (e.g., yellow leaves on tomato): ").strip()
    
    if not symptoms:
        print("No symptoms provided. Exiting.")
        return
    
    # Generate comprehensive analysis
    analysis = research_disease(symptoms, weather_soil_data)
    
    if analysis:
        print("\n🎯 Comprehensive Plant Health Analysis:\n")
        print(analysis)
    else:
        print("Failed to generate analysis.")

# ---------- Command Line Interface ----------
if __name__ == "__main__":
    print("🌿 Welcome to the Integrated Plant Health Assistant")
    
    # Get location
    location = input("📍 Enter location (e.g., 'Ahmednagar, India'): ").strip()
    if not location:
        location = "Ahmednagar, India"  # Default location
    
    # Get image path or symptoms
    img_path = input("\n🖼️ Enter path to plant image (or press Enter to describe symptoms manually): ").strip()
    
    if img_path:
        generate_plant_health_report(location, img_path=img_path)
    else:
        generate_plant_health_report(location)