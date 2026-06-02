import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv

def fetch_from_api_and_build_snapshot(api_key, output_file="../data/offline_snapshot.csv", target_size=25000):
    base_url = "https://api.nal.usda.gov/fdc/v1/foods/list"
    
    # Core nutrients we care about for the diet planner
    target_nutrients = {
        'Protein': 'Protein', 
        'Total lipid (fat)': 'Total lipid (fat)', 
        'Carbohydrate, by difference': 'Carbohydrate, by difference', 
        'Energy': 'Energy', 
        'Fiber, total dietary': 'Fiber, total dietary'
    }

    all_foods = []
    page_size = 200
    pages_needed = (target_size // page_size) + 1
    
    print(f"Fetching {target_size} items from USDA API across {pages_needed} pages...")

    for page in range(1, pages_needed + 1):
        params = {
            "api_key": api_key,
            "dataType": ["Foundation", "SR Legacy", "Branded"],
            "pageSize": page_size,
            "pageNumber": page
        }
        
        response = requests.get(base_url, params=params)
        
        if response.status_code == 429:
            print("Rate limit hit. Sleeping for 60 seconds...")
            time.sleep(60)
            continue
        elif response.status_code != 200:
            print(f"API Error on page {page}: {response.status_code}")
            break
            
        data = response.json()
        if not data:
            break
            
        for item in data:
            food_entry = {
                "fdc_id": item.get("fdcId"),
                "description": item.get("description").lower(), # Lowercase for easier keyword matching later
                "data_type": item.get("dataType")
            }
            
            # Extract specific nutrients
            nutrients = item.get("foodNutrients", [])
            for nut in nutrients:
                nut_name = nut.get("name")
                if nut_name in target_nutrients:
                    food_entry[nut_name] = nut.get("amount")
            
            all_foods.append(food_entry)
            
            # Stop exactly at target_size
            if len(all_foods) >= target_size:
                break
                
        print(f"Fetched page {page}/{pages_needed} | Total items so far: {len(all_foods)}")
        
        if len(all_foods) >= target_size:
            break

    # Convert to DataFrame
    print("Processing and cleaning data...")
    df = pd.DataFrame(all_foods)
    
    # Drop rows missing the core macronutrients
    df.dropna(subset=['Energy', 'Protein', 'Total lipid (fat)', 'Carbohydrate, by difference'], inplace=True)
    
    # Ensure the data directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save the snapshot
    df.to_csv(output_file, index=False)
    print(f"Success! Offline snapshot saved with {len(df)} items at {output_file}.")

if __name__ == "__main__":
    # Load environment variables from the .env file
    load_dotenv()
    
    # Retrieve the key securely
    USDA_API_KEY = os.getenv("USDA_API_KEY")
    
    if not USDA_API_KEY:
        print("Error: USDA_API_KEY not found in .env file.")
    else:
        fetch_from_api_and_build_snapshot(api_key=USDA_API_KEY, target_size=25000)
