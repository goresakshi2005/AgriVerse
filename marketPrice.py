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
    """
    An improved class to fetch agricultural commodity prices with multiple sources,
    latest date tracking, and robust fallback mechanisms.
    """
    
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

    QUALITY_PROMPT = """You are an expert agricultural commodity quality assessor.
First, reason step-by-step internally about the provided image. Then, provide your final assessment in the strict format below.

Analyze the image for:
1.  **Grade (A, B, C):**
    *   **Grade A (Excellent):** Highly uniform in size and color. No visible defects, blemishes, or damage. Looks fresh and clean. Suitable for premium retail.
    *   **Grade B (Good/Average):** Mostly uniform with minor variations. May have small cosmetic blemishes or slight inconsistencies. Suitable for general markets.
    *   **Grade C (Fair/Poor):** Significant variation in size/color. Noticeable defects, damage, or signs of aging/spoilage. Best for processing or immediate discounted sale.
2.  **Moisture Content:** Estimate as Low, Medium, or High based on visual cues (e.g., wilting, shininess, dryness).
3.  **Foreign Matter:** Estimate the percentage of non-commodity material (dirt, stones, stems).
4.  **Damage Details:** Instead of 'yes/no', DESCRIBE any visible damage (e.g., "Minor bruising on 2-3 items", "No visible damage", "Signs of sprouting on one onion").

**Provide your output in this exact format, with no extra text:**
Grade: [A/B/C]
Moisture: [Low/Medium/High]
Foreign Matter: [Low <5%/Medium 5-10%/High >10%]
Damage Details: [Description of any damage, or "None"]
Overall Assessment: [A brief one-sentence summary of the quality]
"""
    
    QUALITY_PRICE_FACTORS = {'A': 1.15, 'B': 1.0, 'C': 0.85}
    
    COMMON_COMMODITIES = [
        "Onion", "Potato", "Tomato", "Wheat", "Paddy (Rice)", "Maize", "Soybean", "Cotton", 
        "Sugarcane", "Gram (Chana)", "Turmeric", "Ginger", "Garlic", "Coriander", "Mustard",
        "Apple", "Banana", "Mango", "Grapes", "Orange"
    ]
    
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
    
    def analyze_commodity_image(self, image_path: str) -> Dict[str, str]:
        """Analyze commodity image using Google's Generative AI with the improved prompt."""
        try:
            img = self._load_image(image_path)
            if isinstance(img, dict) and "error" in img: return img
            
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content([self.QUALITY_PROMPT, img])
            return self._parse_quality_response(response.text)
            
        except Exception as e:
            return {"error": f"Image analysis failed: {str(e)}"}
    
    def _load_image(self, image_path: str):
        try:
            if image_path.startswith(('http://', 'https://')):
                response = requests.get(image_path, timeout=10)
                response.raise_for_status()
                img = Image.open(BytesIO(response.content))
            else:
                if not os.path.exists(image_path): return {"error": f"Image file not found: {image_path}"}
                img = Image.open(image_path)
            return img.convert("RGB")
        except Exception as e:
            return {"error": f"Image loading failed: {str(e)}"}
    
    def _parse_quality_response(self, response_text: str) -> Dict[str, str]:
        quality_data = {}
        fields = ["Grade", "Moisture", "Foreign Matter", "Damage Details", "Overall Assessment"]
        for line in response_text.strip().split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                if key in fields:
                    quality_data[key] = value.strip()
        
        if "Grade" not in quality_data:
            return {"error": "Failed to parse quality analysis response from the AI."}
        return quality_data

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
            # Split into individual price entries
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

    def fetch_price(self, commodity_name: str, quality_data: Optional[Dict[str, str]] = None) -> str:
        if not self.api_keys_configured: return "Error: API keys not configured properly"
            
        try:
            price_agent = Agent(
                model=Gemini(model="gemini-1.5-flash"),
                system_prompt=self.SYSTEM_PROMPT,
                tools=[TavilyTools()],
            )
            
            # Initial query for multiple sources
            query = f"Current wholesale prices for {commodity_name} in India from 2-4 official sources with dates"
            quality_grade = quality_data.get('Grade') if quality_data else None
            if quality_grade: query += f" for quality similar to Grade {quality_grade}"
            
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
                return f"Price information for '{commodity_name}' is not available from official sources at this time."

            final_price_info = []
            quality_grade = quality_data.get('Grade') if quality_data else None
            
            # Process individual prices from multiple sources
            if parsed_data["prices"]:
                prices = []
                for price_data in parsed_data["prices"]:
                    try:
                        price_kg = float(re.sub(r"[^\d.]", "", price_data["PRICE_PER_KG"]))
                        if quality_grade:
                            price_kg *= self.QUALITY_PRICE_FACTORS.get(quality_grade, 1.0)
                        prices.append((price_kg, price_data))
                    except (ValueError, KeyError):
                        continue
                
                if prices:
                    avg_price = sum(p[0] for p in prices) / len(prices)
                    final_price_info.append(f"{avg_price:.2f} INR/kg (Average)")
                    if quality_grade:
                        final_price_info[-1] += f" (Adjusted for Grade {quality_grade})"
                    
                    for i, (price, data) in enumerate(prices[:3], 1):  # Show top 3 sources
                        source = data.get("SOURCE", "Unknown")
                        date = data.get("DATE", "N/A")
                        final_price_info.append(f"  Source {i}: {price:.2f} INR/kg | {source} | {date}")

            # Process price ranges
            elif parsed_data["ranges"]:
                ranges = []
                for range_data in parsed_data["ranges"]:
                    try:
                        min_p = float(re.sub(r"[^\d.]", "", range_data["MIN_PRICE_KG"]))
                        max_p = float(re.sub(r"[^\d.]", "", range_data["MAX_PRICE_KG"]))
                        if quality_grade:
                            min_p *= self.QUALITY_PRICE_FACTORS.get(quality_grade, 1.0)
                            max_p *= self.QUALITY_PRICE_FACTORS.get(quality_grade, 1.0)
                        ranges.append((min_p, max_p, range_data))
                    except (ValueError, KeyError):
                        continue
                
                if ranges:
                    avg_min = sum(r[0] for r in ranges) / len(ranges)
                    avg_max = sum(r[1] for r in ranges) / len(ranges)
                    final_price_info.append(f"{avg_min:.2f} - {avg_max:.2f} INR/kg (Range)")
                    if quality_grade:
                        final_price_info[-1] += f" (Adjusted for Grade {quality_grade})"
                    
                    for i, (min_p, max_p, data) in enumerate(ranges[:3], 1):  # Show top 3 sources
                        source = data.get("SOURCE", "Unknown")
                        date = data.get("DATE", "N/A")
                        final_price_info.append(f"  Source {i}: {min_p:.2f}-{max_p:.2f} INR/kg | {source} | {date}")

            # Add latest date information
            if parsed_data["latest_date"]:
                final_price_info.append(f"Latest Data Date: {parsed_data['latest_date']}")
            elif parsed_data["prices"] or parsed_data["ranges"]:
                dates = []
                for price in parsed_data["prices"]:
                    if "DATE" in price:
                        dates.append(price["DATE"])
                for range_data in parsed_data["ranges"]:
                    if "DATE" in range_data:
                        dates.append(range_data["DATE"])
                if dates:
                    latest_date = max(dates, key=lambda d: datetime.strptime(d, "%Y-%m-%d") if "-" in d else d)
                    final_price_info.append(f"Latest Data Date: {latest_date}")

            return "\n".join(final_price_info)
            
        except Exception as e:
            return f"Error during price fetching: {str(e)}"

    def format_results(self, price_results: str, quality_data: Optional[Dict[str, str]] = None) -> str:
        """Formats the price and quality results into a clean, readable output."""
        output = []
        
        if "Error:" in price_results or "not available" in price_results:
            output.append(f"⚠️ {price_results}")
            if not quality_data:
                output.append("\nℹ️ Note: For a quality-adjusted price, please provide a commodity image.")
            return "\n".join(output)

        output.append("--- PRICE ESTIMATE ---")
        
        # Split the price results into lines
        price_lines = [line.strip() for line in price_results.split('\n') if line.strip()]
        
        # The first line is the main price estimate
        main_price_line = price_lines[0]
        if "Average" in main_price_line:
            output.append(f"💰 Average Price: {main_price_line.split('(')[0].strip()}")
        elif "Range" in main_price_line:
            output.append(f"📊 Price Range: {main_price_line.split('(')[0].strip()}")
        
        if quality_data:
            output.append(f"   (Quality Adjustment: Grade {quality_data.get('Grade', 'N/A')})")
        
        # Add the individual source prices
        for line in price_lines[1:]:
            if line.startswith("Source"):
                output.append(f"   - {line}")
            elif line.startswith("Latest Data Date"):
                output.append(f"\n📅 {line}")

        if not quality_data:
            output.append("\nℹ️ Note: Provide an image for a more precise, quality-adjusted price estimate.")
        
        if quality_data and "error" not in quality_data:
            output.append("\n--- AI QUALITY ASSESSMENT ---")
            for key, value in quality_data.items():
                output.append(f"✅ {key}: {value}")
        
        return "\n".join(output)

def main():
    print("--- Indian Agricultural Commodity Price Checker (v2.2) ---")
    print("Now with multiple sources and date tracking.")
    
    fetcher = AgriculturalCommodityPriceFetcher()
    
    if not fetcher.api_keys_configured:
        return
        
    print("\nCommon Commodities:", ", ".join(fetcher.COMMON_COMMODITIES[:10]) + ", etc.")
    
    commodity_name = input("\nEnter commodity name: ").strip()
    if not commodity_name:
        print("Commodity name cannot be empty.")
        return
    
    image_input = input("\nOptional: Path/URL to commodity image (or press Enter to skip): ").strip()
    quality_data = {}
    
    if image_input:
        print("\n🔬 Analyzing commodity image...")
        quality_data = fetcher.analyze_commodity_image(image_input)
        
        if "error" in quality_data:
            print(f"⚠️ Error: {quality_data['error']}")
            print("Proceeding without quality data.")
            quality_data = {}
        else:
            print("✔️ Analysis complete.")

    print(f"\nSearching for current market price of {commodity_name}...")
    price_string = fetcher.fetch_price(commodity_name, quality_data)
    
    print("\n" + "="*50)
    print(f"      RESULTS FOR: {commodity_name.upper()}")
    print("="*50)
    final_output = fetcher.format_results(price_string, quality_data)
    print(final_output)
    print("="*50)

if __name__ == "__main__":
    main()
