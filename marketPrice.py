import os
from typing import Dict, Optional, Tuple
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

# Load environment variables
load_dotenv()

class AgriculturalCommodityPriceFetcher:
    """
    An improved class to fetch agricultural commodity prices with more reliable parsing,
    enhanced generative AI image analysis, and a robust fallback mechanism.
    """
    
    SYSTEM_PROMPT = """You are a highly specialized agricultural price analyst for Indian markets.
Your goal is to provide the most accurate price in INR per **KILOGRAM (kg)**.

Follow these steps precisely:
1. Search for the CURRENT wholesale/mandi price for the commodity from official sources only (eNAM, Agmarknet, APMC data, government reports).
2. The prices you find will likely be in **INR per Quintal (100 kg)**. This is the standard unit in Indian mandis.
3. **You MUST identify the original price and its unit (e.g., 2500 INR per Quintal).**
4. **You MUST convert this price to INR per kg.** For a price per quintal, you will divide by 100.
5. Provide your final response in the following strict format. Do NOT add any other text or explanations.

**FOR A SINGLE PRICE:**
PRICE_PER_KG: [price_in_kg] | ORIGINAL_PRICE: [original_price] per Quintal | SOURCE: [source_name] | DATE: [date]

**FOR A PRICE RANGE:**
MIN_PRICE_KG: [min_price_in_kg] | MAX_PRICE_KG: [max_price_in_kg] | ORIGINAL_RANGE: [min_original]-[max_original] per Quintal | SOURCE: [source_name] | DATE: [date]

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

    def _parse_price_response(self, response_text: str) -> Optional[Dict[str, str]]:
        """Parses the new, structured response from the price agent."""
        data = {}
        try:
            parts = [p.strip() for p in response_text.split('|')]
            for part in parts:
                if ':' in part:
                    key, value = part.split(':', 1)
                    data[key.strip()] = value.strip()
            return data
        except Exception:
            return None

    def fetch_price(self, commodity_name: str, quality_data: Optional[Dict[str, str]] = None) -> str:
        if not self.api_keys_configured: return "Error: API keys not configured properly"
            
        try:
            price_agent = Agent(
                model=Gemini(model="gemini-1.5-flash"),
                system_prompt=self.SYSTEM_PROMPT,
                tools=[TavilyTools()],
            )
            
            # Initial, more specific query
            query = f"Current wholesale price for {commodity_name} in India from official sources"
            quality_grade = quality_data.get('Grade') if quality_data else None
            if quality_grade: query += f" for quality similar to Grade {quality_grade}"
            
            response = price_agent.run(query)
            response_text = str(response.content) if hasattr(response, 'content') else str(response)

            # --- IMPROVEMENT: FALLBACK MECHANISM ---
            # If the strict initial search fails, try a broader query to get a general price range.
            if "not available" in response_text.lower():
                print("\nInitial search was too specific. Trying a broader query for a price range...")
                alt_query = f"Latest mandi price range for {commodity_name} in India from Agmarknet or eNAM"
                alt_response = price_agent.run(alt_query)
                alt_text = str(alt_response.content) if hasattr(alt_response, 'content') else str(alt_response)

                # Use the fallback result only if it's successful
                if "not available" not in alt_text.lower():
                    response_text = alt_text
            
            # Check again after the potential fallback
            if "not available" in response_text.lower():
                return f"Price information for '{commodity_name}' is not available from official sources at this time."

            parsed_price = self._parse_price_response(response_text)
            
            if not parsed_price:
                return f"Error: Could not parse price information from AI response: '{response_text}'"

            final_price_info = []

            # Adjust price based on quality grade if available
            if quality_grade:
                base_price = 0.0
                try:
                    if "PRICE_PER_KG" in parsed_price:
                        base_price = float(re.sub(r"[^\d.]", "", parsed_price["PRICE_PER_KG"]))
                    elif "MIN_PRICE_KG" in parsed_price and "MAX_PRICE_KG" in parsed_price:
                        min_p = float(re.sub(r"[^\d.]", "", parsed_price["MIN_PRICE_KG"]))
                        max_p = float(re.sub(r"[^\d.]", "", parsed_price["MAX_PRICE_KG"]))
                        base_price = (min_p + max_p) / 2
                except (ValueError, KeyError):
                    return f"Error: Could not extract a valid base price from response: {parsed_price}"

                if base_price > 0:
                    adjustment_factor = self.QUALITY_PRICE_FACTORS.get(quality_grade, 1.0)
                    adjusted_price = base_price * adjustment_factor
                    
                    final_price_info.append(f"{adjusted_price:.2f} INR/kg (Adjusted for Grade {quality_grade})")
                    final_price_info.append(f"Source: {parsed_price.get('SOURCE', 'N/A')}")
                    final_price_info.append(f"Date: {parsed_price.get('DATE', 'N/A')}")
                    if "ORIGINAL_PRICE" in parsed_price:
                        final_price_info.append(f"Market Price (raw): {parsed_price['ORIGINAL_PRICE']}")
                    elif "ORIGINAL_RANGE" in parsed_price:
                         final_price_info.append(f"Market Range (raw): {parsed_price['ORIGINAL_RANGE']}")

            # If no quality grade, return the direct price/range from the agent
            else:
                if "PRICE_PER_KG" in parsed_price:
                    final_price_info.append(f"{parsed_price.get('PRICE_PER_KG', 'N/A')} INR/kg")
                elif "MIN_PRICE_KG" in parsed_price and "MAX_PRICE_KG" in parsed_price:
                     final_price_info.append(f"{parsed_price.get('MIN_PRICE_KG', 'N/A')} - {parsed_price.get('MAX_PRICE_KG', 'N/A')} INR/kg")
                final_price_info.append(f"Source: {parsed_price.get('SOURCE', 'N/A')}")
                final_price_info.append(f"Date: {parsed_price.get('DATE', 'N/A')}")
                
            return " | ".join(final_price_info)
            
        except Exception as e:
            return f"Error during price fetching: {str(e)}"

    def format_results(self, price_results: str, quality_data: Optional[Dict[str, str]] = None) -> str:
        """Formats the price and quality results into a clean, readable output."""
        output = []
        
        # --- IMPROVEMENT: Better handling of error/not-available messages ---
        if "Error:" in price_results or "not available" in price_results:
            output.append(f"⚠️ {price_results}")
            if not quality_data:
                output.append("\nℹ️ Note: For a quality-adjusted price, please provide a commodity image.")
            return "\n".join(output)

        output.append("--- PRICE ESTIMATE ---")
        price_parts = [p.strip() for p in price_results.split('|')]
        
        price_line = price_parts[0]
        if "Adjusted" in price_line:
            output.append(f"💰 Adjusted Price: {price_line.split('(')[0].strip()}")
            output.append(f"   (Based on Grade {quality_data.get('Grade', 'N/A')})")
        else:
             output.append(f"📊 Market Price/Range: {price_line}")

        for part in price_parts[1:]:
            output.append(f"   - {part}")

        if not quality_data:
             output.append("\nℹ️ Note: Provide an image for a more precise, quality-adjusted price estimate.")
        
        if quality_data and "error" not in quality_data:
            output.append("\n--- AI QUALITY ASSESSMENT ---")
            for key, value in quality_data.items():
                output.append(f"✅ {key}: {value}")
        
        return "\n".join(output)

def main():
    print("--- Indian Agricultural Commodity Price Checker (v2.1) ---")
    print("Now with robust fallback for fetching price ranges.")
    
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