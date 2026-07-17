
import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()  

DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_USER = os.getenv("DB_USER", "postgres")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "rf_telecom_db")

if not DB_PASSWORD:
    raise ValueError("DB_PASSWORD not found. Create a .env file with DB_PASSWORD=your_password")

engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

print("Loading dim_tower...")
dim_tower = pd.read_csv('data/processed/dim_tower.csv')
dim_tower.to_sql('dim_tower', engine, if_exists='replace', index=False)
print(f"  Loaded {len(dim_tower)} rows")

print("Loading dim_operator...")
dim_operator = pd.read_csv('data/processed/dim_operator.csv')
dim_operator.to_sql('dim_operator', engine, if_exists='replace', index=False)
print(f"  Loaded {len(dim_operator)} rows")

print("Loading dim_date...")
dim_date = pd.read_csv('data/processed/dim_date.csv')
dim_date.to_sql('dim_date', engine, if_exists='replace', index=False)
print(f"  Loaded {len(dim_date)} rows")

print("Loading dim_environment...")
dim_environment = pd.read_csv('data/processed/dim_environment.csv')
dim_environment.to_sql('dim_environment', engine, if_exists='replace', index=False)
print(f"  Loaded {len(dim_environment)} rows")

print("Loading fact_rf_readings...")
fact_rf_readings = pd.read_csv('data/processed/fact_rf_readings.csv')
fact_rf_readings.to_sql('fact_rf_readings', engine, if_exists='replace', index=False)
print(f"  Loaded {len(fact_rf_readings)} rows")

print("\nAll tables loaded into PostgreSQL successfully!")