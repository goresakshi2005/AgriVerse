import os
from typing import Dict, Optional, Tuple, List
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.tavily import TavilyTools
from datetime import datetime
from PIL import Image
import requests
from io import BytesIO
import google.generativeai as genai
from dotenv import load_dotenv
import re
from collections import defaultdict

# Load environment variables
load_dotenv()

class AgriculturalCommodityPriceFetcher:
    """Class to fetch agricultural commodity prices"""
    
    SYSTEM_PROMPT = """You are a highly specialized agricultural price analyst for Indian markets.
Your goal is to provide the most accurate price in INR per **KILOGRAM (kg)** from multiple official sources.

Follow these steps precisely:
1. Search for the CURRENT/latest wholesale/mandi price for the commodity from at least 2-4 official sources (eNAM, Agmarknet, APMC data, government reports).
2. For each source, extract:
   - The price (likely in INR per Quintal (100 kg))
   - The date of the price data
   - The source name
3. **For each source, convert the price to INR per kg** (divide quintal price by 100).
4. Identify the most recent date among all sources.
5. Provide your final response in the following strict format:

**FOR SINGLE PRICES FROM MULTIPLE SOURCES:**
PRICE_PER_KG: [price1_in_kg] | SOURCE: [source1] | DATE: [date1]
PRICE_PER_KG: [price2_in_kg] | SOURCE: [source2] | DATE: [date2]
...
LATEST_DATE: [most_recent_date]

**FOR PRICE RANGES:**
MIN_PRICE_KG: [min_price_in_kg] | MAX_PRICE_KG: [max_price_in_kg] | SOURCE: [source] | DATE: [date]
...
LATEST_DATE: [most_recent_date]

If after a thorough search, no current data is found from official sources, return the single phrase: 'Not available'.
"""

    QUALITY_PRICE_FACTORS = {'A': 1.15, 'B': 1.0, 'C': 0.85}
    
    def __init__(self):
        self.api_keys_configured = self._check_api_keys()
        if self.api_keys_configured:
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        
    def _check_api_keys(self) -> bool:
        required_keys = {"TAVILY_API_KEY": os.getenv("TAVILY_API_KEY"), "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY")}
        missing_keys = [k for k, v in required_keys.items() if not v]
        if missing_keys:
            print(f"Warning: Missing API keys: {', '.join(missing_keys)}. Some services may not work.")
            return False
        return True
    
    def _parse_price_response(self, response_text: str) -> Dict:
        """Parses the response with multiple sources and dates."""
        data = {
            "prices": [],
            "ranges": [],
            "latest_date": None,
            "sources": set()
        }
        
        if "not available" in response_text.lower():
            return data
        
        try:
            entries = [entry.strip() for entry in response_text.split('\n') if entry.strip()]
            
            for entry in entries:
                if entry.startswith("LATEST_DATE:"):
                    data["latest_date"] = entry.split(":", 1)[1].strip()
                    continue
                
                parts = [p.strip() for p in entry.split('|')]
                entry_data = {}
                for part in parts:
                    if ':' in part:
                        key, value = part.split(':', 1)
                        entry_data[key.strip()] = value.strip()
                
                if "PRICE_PER_KG" in entry_data:
                    data["prices"].append(entry_data)
                    data["sources"].add(entry_data.get("SOURCE", "Unknown"))
                elif "MIN_PRICE_KG" in entry_data and "MAX_PRICE_KG" in entry_data:
                    data["ranges"].append(entry_data)
                    data["sources"].add(entry_data.get("SOURCE", "Unknown"))
            
            return data
            
        except Exception as e:
            print(f"Error parsing price response: {e}")
            return data

    def fetch_price(self, commodity_name: str) -> float:
        """Fetch the average market price per quintal (100 kg) for the commodity"""
        if not self.api_keys_configured: 
            print("Error: API keys not configured properly. Using default price of ₹2000 per quintal.")
            return 2000  # Default price if API keys not configured
            
        try:
            price_agent = Agent(
                model=Gemini(model="gemini-1.5-flash"),
                system_prompt=self.SYSTEM_PROMPT,
                tools=[TavilyTools()],
            )
            
            query = f"Current wholesale prices for {commodity_name} in India from 2-4 official sources with dates"
            response = price_agent.run(query)
            response_text = str(response.content) if hasattr(response, 'content') else str(response)

            # Fallback mechanism if initial search fails
            if "not available" in response_text.lower():
                print("\nInitial search was too specific. Trying a broader query...")
                alt_query = f"Latest mandi prices for {commodity_name} in India from any reliable sources"
                alt_response = price_agent.run(alt_query)
                alt_text = str(alt_response.content) if hasattr(alt_response, 'content') else str(alt_response)
                if "not available" not in alt_text.lower():
                    response_text = alt_text

            parsed_data = self._parse_price_response(response_text)
            
            if not parsed_data["prices"] and not parsed_data["ranges"]:
                print(f"Price information for '{commodity_name}' is not available from official sources. Using default price of ₹2000 per quintal.")
                return 2000  # Default price if no data found

            # Process individual prices from multiple sources
            if parsed_data["prices"]:
                prices = []
                for price_data in parsed_data["prices"]:
                    try:
                        price_kg = float(re.sub(r"[^\d.]", "", price_data["PRICE_PER_KG"]))
                        prices.append(price_kg)
                    except (ValueError, KeyError):
                        continue
                
                if prices:
                    avg_price_kg = sum(prices) / len(prices)
                    return avg_price_kg * 100  # Convert back to quintal price

            # Process price ranges
            elif parsed_data["ranges"]:
                ranges = []
                for range_data in parsed_data["ranges"]:
                    try:
                        min_p = float(re.sub(r"[^\d.]", "", range_data["MIN_PRICE_KG"]))
                        max_p = float(re.sub(r"[^\d.]", "", range_data["MAX_PRICE_KG"]))
                        ranges.append((min_p + max_p) / 2)  # Take average of range
                    except (ValueError, KeyError):
                        continue
                
                if ranges:
                    avg_price_kg = sum(ranges) / len(ranges)
                    return avg_price_kg * 100  # Convert back to quintal price

            print(f"Could not parse price data for '{commodity_name}'. Using default price of ₹2000 per quintal.")
            return 2000  # Fallback default price
            
        except Exception as e:
            print(f"Error during price fetching: {str(e)}. Using default price of ₹2000 per quintal.")
            return 2000  # Default price on error

def calculate_seed_cost(seed_rate_kg, price_per_kg, area):
    return seed_rate_kg * price_per_kg * area

def calculate_fertilizer_cost(cost_per_acre, area):
    return cost_per_acre * area

def calculate_pesticide_cost(total_pesticide_cost):
    return total_pesticide_cost

def calculate_water_cost(cost_per_irrigation, num_irrigations):
    return cost_per_irrigation * num_irrigations

def calculate_electricity_diesel_cost(monthly_cost, duration_months):
    return monthly_cost * duration_months

def calculate_labor_cost(daily_wage, num_workers, num_days):
    return daily_wage * num_workers * num_days

def calculate_equipment_rent(rent_cost):
    return rent_cost

def calculate_land_rent(land_rent_per_season):
    return land_rent_per_season

def calculate_transport_cost(transport_per_quintal, expected_yield):
    return transport_per_quintal * expected_yield

def calculate_other_costs(other_costs):
    return other_costs

def production_cost_calculator():
    print("\n🌾 Realistic Farm Production Cost Calculator")
    
    # Initialize price fetcher
    price_fetcher = AgriculturalCommodityPriceFetcher()
    
    crop = input("Enter crop name: ")
    area = float(input("Land area (in acres): "))
    expected_yield = float(input("Expected total yield (quintals): "))
    
    print(f"\nFetching current market price for {crop}...")
    selling_price_per_quintal = price_fetcher.fetch_price(crop)
    selling_price = selling_price_per_quintal  # Already in ₹/quintal
    
    print(f"\nCurrent market price for {crop}: ₹{selling_price:.2f} per quintal")

    # Inputs with logic
    seed_rate = float(input("\nSeed rate (kg/acre): "))
    seed_price = float(input("Seed price (₹/kg): "))
    seed_cost = calculate_seed_cost(seed_rate, seed_price, area)

    fert_cost_per_acre = float(input("Fertilizer cost per acre: "))
    fertilizer_cost = calculate_fertilizer_cost(fert_cost_per_acre, area)

    pesticide_cost = float(input("Total pesticide cost (₹): "))
    pesticide_cost = calculate_pesticide_cost(pesticide_cost)

    irrigation_cost = float(input("Cost per irrigation (₹): "))
    num_irrigations = int(input("Number of irrigations: "))
    water_cost = calculate_water_cost(irrigation_cost, num_irrigations)

    diesel_cost = float(input("Monthly electricity/diesel cost (₹): "))
    duration_months = int(input("Duration in months: "))
    electricity_cost = calculate_electricity_diesel_cost(diesel_cost, duration_months)

    wage = float(input("Daily wage per worker (₹): "))
    workers = int(input("Number of workers: "))
    days = int(input("Total labor days: "))
    labor_cost = calculate_labor_cost(wage, workers, days)

    equipment_rent = float(input("Total equipment rent (tractor, tools, etc.): "))
    land_rent = float(input("Land lease/rent for the season: "))
    transport_per_quintal = float(input("Transport cost per quintal: "))
    transport_cost = calculate_transport_cost(transport_per_quintal, expected_yield)
    other_cost = float(input("Other/Miscellaneous costs (insurance, interest, etc.): "))

    # Total Cost
    total_cost = sum([
        seed_cost, fertilizer_cost, pesticide_cost,
        water_cost, electricity_cost, labor_cost,
        equipment_rent, land_rent, transport_cost, other_cost
    ])

    break_even_price = total_cost / expected_yield
    total_income = selling_price * expected_yield
    profit = total_income - total_cost

    # Summary
    print("\n📋 Summary for Crop:", crop)
    print(f"Total Production Cost: ₹{total_cost:.2f}")
    print(f"Expected Yield: {expected_yield:.2f} quintals")
    print(f"Break-even Selling Price: ₹{break_even_price:.2f} per quintal")
    print(f"Current Market Price: ₹{selling_price:.2f} per quintal")
    print(f"Expected Income: ₹{total_income:.2f}")

    if profit > 0:
        print(f"✅ Profit: ₹{profit:.2f}")
    elif profit < 0:
        print(f"❌ Loss: ₹{abs(profit):.2f}")
    else:
        print("⚠️ No Profit, No Loss")

# Run
if __name__ == "__main__":
    production_cost_calculator()