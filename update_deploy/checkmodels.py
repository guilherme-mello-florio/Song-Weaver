import google.generativeai as genai
import os

# --- IMPORTANT ---
# Make sure you have your GOOGLE_API_KEY set as an environment variable
# before running this script.
try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    print("Please ensure your GOOGLE_API_KEY environment variable is set correctly.")
    exit()

print("--- Available Gemini Models ---")
print("Models that support the 'generateContent' method:\n")

try:
    for model in genai.list_models():
        # The 'generateContent' method is used for standard text generation.
        if 'generateContent' in model.supported_generation_methods:
            print(f"- {model.name}")
except Exception as e:
    print(f"Could not retrieve model list. Error: {e}")

print("\n---------------------------------")
print("Recommendation: Use one of the models listed above in your app.py file.")
print("The model 'gemini-1.0-pro' is usually a safe and powerful choice.")
