import requests
import pandas as pd
import sys
import numpy as np

#configuration
API_KEY='ed4e95523f37e786997897ea590877da9af075497b1fc2fe958b41687dba13d1'
SEARCH_QUERY="Data Scientist OR Data Engineer OR Engineer"
LOCATION="United States"

def fetch_job_listings():
    url = "https://serpapi.com/search"
    params = {
        "engine": "google_jobs",
        "q": SEARCH_QUERY,
        "location": LOCATION,
        "hl": "en",       # 语言设置为英文
        "api_key": API_KEY
    }
    try:
        print(f"Fetching job listings for query: {SEARCH_QUERY} in {LOCATION}")
        response = requests.get(url, params=params)
        data=response.json()

        jobs=data.get("jobs_results", [])
        job_listings=[]
        for job in jobs:
            job_info={
                "title": job.get("title"),
                "company": job.get("company_name"),
                
            }
            job_listings.append(job_info)
        return pd.DataFrame(job_listings)
    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame()  
if __name__ == "__main__":
    df_jobs = fetch_job_listings()
    print(df_jobs.head())

