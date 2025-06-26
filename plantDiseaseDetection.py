import os
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.tavily import TavilyTools
from phi.tools.pubmed import PubmedTools

# ---------- Step 1: Load API Keys ----------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# ---------- Step 2: Configure Gemini Models ----------
genai.configure(api_key=GOOGLE_API_KEY)
image_model = genai.GenerativeModel('gemini-1.5-flash')
research_model = Gemini(api_key=GOOGLE_API_KEY)

# ---------- Step 3: Setup Research Tools ----------
tools = [TavilyTools(api_key=TAVILY_API_KEY), PubmedTools()]
research_agent = Agent(tools=tools, model=research_model)

# ---------- Step 4: Image Analysis Function ----------
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

# ---------- Step 5: Disease Research Function ----------
def research_disease(symptoms):
    """Get disease information based on symptoms"""
    prompt = f"""
    You are a plant disease expert.

    The plant shows the following symptoms: "{symptoms}"

    🎯 Your task:
    1. Search trusted agricultural sites and scientific sources (PubMed, Tavily, etc.).
    2. Based on the symptoms, identify and return **3 most likely diseases**. Explore fungal, bacterial, viral, and physiological causes — not just common ones.

    Return only the following structured output:

    📌 **Most Likely Disease:**
    - <Disease 1>: (Why it's likely based on specific symptom pattern)
    - <Disease 2>: (Specific indication such as pathogen type or climate-related factor)
    - <Disease 3>: (Physiological or pest-induced if relevant)

    ✅ **Preventive Measures:**
    - <Step 1>: (Include specific tips like resistant varieties, planting depth, spacing, or crop rotation)
    - <Step 2>: (Soil amendment or nutrition advice with fertilizer name or ratio)
    - <Step 3>: (Watering, mulching, or environment control strategy)

    💊 **Treatment Actions:**
    - <Treatment 1>: (Name of fungicide/insecticide/antibiotic; application interval)
    - <Treatment 2>: (Organic or natural option — e.g., neem oil, baking soda, etc.)
    - <Treatment 3>: (Physical method — e.g., pruning, soil solarization, removing infected plants)

    📎 Output must:
    - Be **concise**, practical, and relevant
    - Include **specific names** (disease, fertilizer, product)
    - Avoid generalities or repeated suggestions
    - Skip intro, summaries, or conclusions
    - don't include step1, step2, step3, treatment1, treatment2, treatment3
    """

    print("\n🤖 Gemini AI Agent is analyzing using Tavily + PubMed...\n")
    response = research_agent.run(prompt)
    return response.content if response else None

# ---------- Step 6: Main Function ----------
def main():
    print("🌿 Welcome to the Plant Disease Assistant")
    
    # Get image path from user
    img_path = input("🖼️ Enter path to plant image (or press Enter to describe symptoms manually): ").strip()
    
    if img_path:
        # Analyze image
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
        symptoms = input("\n✏️ Please describe the plant's symptoms (e.g., yellow leaves on tomato): ").strip()
    
    if symptoms:
        # Research disease based on symptoms
        research_results = research_disease(symptoms)
        if research_results:
            print("\n🎯 Diagnosis & Recommendations:\n")
            print(research_results)
        else:
            print("Failed to get disease information.")
    else:
        print("No symptoms provided. Exiting.")

if __name__ == "__main__":
    main()