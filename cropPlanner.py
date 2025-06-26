import os
from dotenv import load_dotenv
import google.generativeai as genai
import ee
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.tavily import TavilyTools
import requests
from datetime import datetime, timedelta
import pandas as pd

# ---------- Initialize Environment ----------
load_dotenv()

# Initialize geocoder
geolocator = Nominatim(user_agent="crop_planner")

# Configure Gemini Models
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
planning_model = Gemini(api_key=os.getenv("GOOGLE_API_KEY"))

# Setup Research Tools
tools = [TavilyTools(api_key=os.getenv("TAVILY_API_KEY"))]
planning_agent = Agent(tools=tools, model=planning_model)

# ---------- Helper Functions (Reused from plant health code) ----------
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
        },
        {
            "name": "Soil Texture",
            "collection": "OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02",
            "date_range": None,
            "band": 'b0',
            "scale": 500
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
                # For soil texture, map values to texture classes
                elif ds['name'] == "Soil Texture":
                    texture_map = {
                        1: "Clay",
                        2: "Silty clay",
                        3: "Sandy clay",
                        4: "Clay loam",
                        5: "Silty clay loam",
                        6: "Sandy clay loam",
                        7: "Loam",
                        8: "Silty loam",
                        9: "Sandy loam",
                        10: "Silt",
                        11: "Loamy sand",
                        12: "Sand"
                    }
                    results["soil_texture"] = texture_map.get(round(value), "Unknown")
            
        except Exception as e:
            print(f"⚠️ Failed to process {ds['name']}: {str(e)}")
            if ds['name'] == "Soil Moisture":
                results["soil_moisture"] = None
            elif ds['name'] == "Soil pH":
                results["soil_ph"] = None
            elif ds['name'] == "Soil Texture":
                results["soil_texture"] = None
    
    return results

def get_weather_forecast(lat, lon):
    """Get 5-day weather forecast"""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return {"error": "API key not found", "data": None}
    
    url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    
    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        
        if response.status_code == 200:
            # Process forecast data
            forecast = []
            for item in data['list']:
                forecast.append({
                    "datetime": item['dt_txt'],
                    "temp": item['main']['temp'],
                    "humidity": item['main']['humidity'],
                    "rain": item.get('rain', {}).get('3h', 0),
                    "description": item['weather'][0]['description']
                })
            return {"error": None, "data": forecast}
        else:
            return {"error": data.get('message', 'Unknown error'), "data": None}
    except Exception as e:
        return {"error": str(e), "data": None}

# ---------- Crop Planning Functions ----------
def get_historical_ndvi(aoi, years=3):
    """Get historical NDVI trends for the area"""
    try:
        # Get current year and previous years
        current_year = datetime.now().year
        years_range = range(current_year - years, current_year)
        
        ndvi_data = {}
        
        for year in years_range:
            # Define date range for the growing season (April to October)
            start_date = f"{year}-04-01"
            end_date = f"{year}-10-31"
            
            # Load Sentinel-2 imagery (updated collection)
            sentinel = ee.ImageCollection('COPERNICUS/S2_HARMONIZED') \
                .filterBounds(aoi) \
                .filterDate(start_date, end_date) \
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
            
            # Calculate NDVI with updated band names
            def add_ndvi(image):
                ndvi = image.normalizedDifference(['B8A', 'B4']).rename('NDVI')
                return image.addBands(ndvi)
            
            sentinel_ndvi = sentinel.map(add_ndvi)
            
            # Get mean NDVI for the season
            mean_ndvi = sentinel_ndvi.select('NDVI').mean().clip(aoi)
            
            # Calculate statistics
            stats = mean_ndvi.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=aoi,
                scale=10,
                maxPixels=1e9
            )
            
            stats_info = stats.getInfo()
            if stats_info:
                ndvi_data[year] = round(stats_info['NDVI'], 3)
        
        return ndvi_data if ndvi_data else None
    
    except Exception as e:
        print(f"⚠️ Failed to get historical NDVI: {str(e)}")
        return None

def get_crop_calendar(location_name, crop_type=None):
    """Get optimal planting and harvesting dates for crops in the region"""
    aoi, lat, lon = get_coordinates(location_name)
    
    # Get climate zone information
    try:
        climate_zone_img = ee.Image("USDOS/CDL/2018").clip(aoi)
        climate_zone = climate_zone_img.reduceRegion(
            reducer=ee.Reducer.mode(),
            geometry=aoi,
            scale=30
        ).getInfo()
        
        # This is simplified - actual climate zone would require more complex analysis
        climate_zone = "Tropical" if lat < 23.5 else "Temperate"
    except:
        climate_zone = "Unknown"
    
    # Get soil data
    soil_data = get_soil_data(aoi)
    
    # Get historical NDVI trends
    ndvi_trends = get_historical_ndvi(aoi)
    
    # Get weather forecast
    forecast = get_weather_forecast(lat, lon)
    
    # Prepare context for AI
    context = f"""
    Location: {location_name} (Lat: {lat}, Lon: {lon})
    Climate Zone: {climate_zone}
    
    Soil Conditions:
    - pH: {soil_data.get('soil_ph', 'N/A')}
    - Texture: {soil_data.get('soil_texture', 'N/A')}
    - Moisture: {soil_data.get('soil_moisture', 'N/A')}%
    
    Historical Vegetation Trends (NDVI):
    {ndvi_trends if ndvi_trends else "No historical data available"}
    
    Current Weather:
    - Temperature: {forecast['data'][0]['temp'] if forecast and forecast['data'] else 'N/A'}°C
    - Humidity: {forecast['data'][0]['humidity'] if forecast and forecast['data'] else 'N/A'}%
    - Conditions: {forecast['data'][0]['description'] if forecast and forecast['data'] else 'N/A'}
    """
    
    # Generate crop calendar
    prompt = f"""
    You are an expert agronomist with 20+ years of experience in crop planning. 
    Create a detailed crop calendar for {location_name} based on the following information:
    
    {context}
    
    For {f"crop: {crop_type}" if crop_type else "common crops in this region"}, provide:
    
    1. OPTIMAL PLANTING WINDOWS:
    - List 3-5 recommended crops with their ideal planting dates
    - Include both rainy season and dry season options
    - Specify exact date ranges (e.g., "June 15 - July 10")
    
    2. CROP-SPECIFIC GUIDELINES:
    For each recommended crop:
    - Soil preparation requirements
    - Seed rate per acre
    - Spacing recommendations
    - Fertilizer schedule (organic options preferred)
    - Irrigation requirements
    - Expected harvest timeline
    
    3. INTERCROPPING SUGGESTIONS:
    - Beneficial crop combinations
    - Planting patterns (row ratios)
    - Timing coordination
    
    4. RISK MANAGEMENT:
    - Weather-related risks for each planting window
    - Contingency plans
    - Pest/disease prevention measures
    
    5. COST ANALYSIS:
    - Estimated input costs per acre
    - Expected yield ranges
    - Market price trends
    
    6. LABOR CALENDAR:
    - Peak labor requirements
    - Key activities by month
    
    Format the output clearly with sections for each crop, using simple language and local units of measurement.
    
    DELIVERY REQUIREMENTS:
    - For all above planning, keep weather and soil data like temperature, humidity, rainfall, cloudiness, soil ph & moisture in mind
    - Use simple language (8th grade level), easy to understand, avoid jargon
    - Provide clear, actionable steps
    - Provide measurements in local units (kg, liters, etc.)
    - Include cost-effective solutions
    - Specify application frequency/dosage precisely
    - Mention safety precautions for all treatments
    - Format the following data properly with '-' symbol for subpoints. insert bullet point for only main heading and for subheading use star. Keep the text clear and professional. use emojis
    - separate sections with clear headings
    - Maintain clear and concise sentence structure.
    - remove markdown and align on the left side (subpoints also)
    
    """
    
    response = planning_agent.run(prompt)
    return response.content if response else None

def generate_irrigation_schedule(soil_data, weather_forecast, crop_type):
    """Generate customized irrigation schedule"""
    prompt = f"""
    Create a detailed irrigation schedule for {crop_type} based on:
    - Soil type: {soil_data.get('soil_texture', 'Unknown')}
    - Soil moisture: {soil_data.get('soil_moisture', 'N/A')}%
    - pH: {soil_data.get('soil_ph', 'N/A')}
    - Upcoming weather: {weather_forecast['data'][0]['description'] if weather_forecast and weather_forecast['data'] else 'N/A'}
    
    Include:
    1. Initial soil preparation watering requirements
    2. Weekly irrigation schedule (amount and frequency)
    3. Adjustments for rainfall
    4. Monitoring indicators (how to check if watering is adequate)
    5. Water conservation techniques
    
    DELIVERY REQUIREMENTS:
    - For all above planning, keep weather and soil data like temperature, humidity, rainfall, cloudiness, soil ph & moisture in mind
    - Use simple language (8th grade level), easy to understand, avoid jargon
    - Provide clear, actionable steps
    - Provide measurements in local units (kg, liters, etc.)
    - Include cost-effective solutions
    - Specify application frequency/dosage precisely
    - Mention safety precautions for all treatments
    - Format the following data properly with '-' symbol for subpoints. insert bullet point for only main heading and for subheading use star. Keep the text clear and professional. use emojis       
    - remove markdown and align on the left side (subpoints also)
    - separate sections with clear headings
    - Maintain clear and concise sentence structure.
    
    """
    
    response = planning_agent.run(prompt)
    return response.content if response else None

def generate_crop_rotation_plan(location_name, current_crops, years=3):
    """Generate multi-year crop rotation plan"""
    aoi, lat, lon = get_coordinates(location_name)
    soil_data = get_soil_data(aoi)
    
    prompt = f"""
    Create a {years}-year crop rotation plan for {location_name} with current crops: {', '.join(current_crops)}.
    
    Soil conditions:
    - pH: {soil_data.get('soil_ph', 'N/A')}
    - Texture: {soil_data.get('soil_texture', 'N/A')}
    
    The plan should:
    1. Improve soil health over time
    2. Break pest/disease cycles
    3. Maintain farm income
    4. Include cover crops where beneficial
    5. Provide planting and harvest dates for each crop
    
    Format as a clear yearly table with explanations for each rotation choice.
    
    DELIVERY REQUIREMENTS:
    - For all above planning, keep weather and soil data like temperature, humidity, rainfall, cloudiness, soil ph & moisture in mind
    - Use simple language (8th grade level), easy to understand, avoid jargon
    - Provide clear, actionable steps
    - Provide measurements in local units (kg, liters, etc.)
    - Include cost-effective solutions
    - Specify application frequency/dosage precisely
    - Mention safety precautions for all treatments
    - Format the following data properly with '-' symbol for subpoints. insert bullet point for only main heading and for subheading use star. Keep the text clear and professional. use emojis    - separate sections with clear headings
    - Maintain clear and concise sentence structure.
    - remove markdown and align on the left side (subpoints also)
    
    """
    
    response = planning_agent.run(prompt)
    return response.content if response else None

# ---------- Main Function ----------
def generate_crop_plan(location_name, crop_type=None):
    """Generate comprehensive crop plan for location"""
    print("\n" + "="*50)
    print(f"🌱 Comprehensive Crop Plan for: {location_name}")
    if crop_type:
        print(f"🌾 Focus Crop: {crop_type}")
    print("="*50 + "\n")
    
    # Initialize Earth Engine
    if not initialize_earth_engine():
        return None
    
    # Get coordinates
    try:
        aoi, lat, lon = get_coordinates(location_name)
    except Exception as e:
        print(f"❌ Failed to get coordinates for location: {str(e)}")
        return None
    
    # Get soil data
    print("\n🌱 Analyzing soil conditions...")
    soil_data = get_soil_data(aoi)
    if soil_data:
        print(f"- pH: {soil_data.get('soil_ph', 'N/A')}")
        print(f"- Texture: {soil_data.get('soil_texture', 'N/A')}")
        print(f"- Moisture: {soil_data.get('soil_moisture', 'N/A')}%")
    else:
        print("⚠️ Could not retrieve soil data")
    
    # Get weather forecast
    print("\n🌦️ Checking weather forecast...")
    forecast = get_weather_forecast(lat, lon)
    if forecast and forecast['data']:
        print(f"- Next 5 days: {forecast['data'][0]['description']}")
        print(f"- Temperature: {forecast['data'][0]['temp']}°C")
        print(f"- Humidity: {forecast['data'][0]['humidity']}%")
    else:
        print("⚠️ Could not retrieve weather forecast")
    
    # Get crop calendar
    print("\n📅 Generating crop calendar...")
    calendar = get_crop_calendar(location_name, crop_type)
    if calendar:
        print("\n" + calendar)
    else:
        print("⚠️ Failed to generate crop calendar")
    
    # If specific crop provided, generate irrigation schedule
    if crop_type:
        print(f"\n💧 Generating irrigation schedule for {crop_type}...")
        irrigation = generate_irrigation_schedule(soil_data, forecast, crop_type)
        if irrigation:
            print("\n" + irrigation)
        else:
            print("⚠️ Failed to generate irrigation schedule")
    
    # Generate rotation plan if requested
    if input("\n🔄 Would you like a crop rotation plan? (y/n): ").lower() == 'y':
        current_crops = input("Enter current crops (comma separated): ").split(',')
        rotation_plan = generate_crop_rotation_plan(location_name, current_crops)
        if rotation_plan:
            print("\n" + rotation_plan)
        else:
            print("⚠️ Failed to generate rotation plan")

# ---------- Command Line Interface ----------
if __name__ == "__main__":
    print("🌾 Welcome to the Smart Crop Planning System")
    
    # Get location
    location = input("📍 Enter location (e.g., 'Ahmednagar, India'): ").strip()
    if not location:
        location = "Ahmednagar, India"  # Default location
    
    # Get crop type (optional)
    crop_type = input("\n🌱 Enter specific crop type (or press Enter for general recommendations): ").strip()
    
    generate_crop_plan(location, crop_type if crop_type else None)