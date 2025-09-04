#!/usr/bin/env python3
"""
Data Preparation Script for Multi-Container Data Pipeline
This script generates synthetic data and stores it in PostgreSQL database.
"""

import pandas as pd
import numpy as np
import psycopg2
import time
import os
from sqlalchemy import create_engine, text

def wait_for_database():
    """Wait for the database to be ready"""
    max_attempts = 30
    attempt = 0
    
    while attempt < max_attempts:
        try:
            # Try to connect to the database
            conn = psycopg2.connect(
                host=os.getenv('DB_HOST', 'database'),
                database=os.getenv('POSTGRES_DB', 'dsproject'),
                user=os.getenv('POSTGRES_USER', 'postgres'),
                password=os.getenv('POSTGRES_PASSWORD', 'postgres'),
                port=os.getenv('DB_PORT', '5432')
            )
            conn.close()
            print("✅ Database connection successful!")
            return True
        except psycopg2.OperationalError as e:
            attempt += 1
            print(f"⏳ Waiting for database... (attempt {attempt}/{max_attempts})")
            time.sleep(2)
    
    print("❌ Failed to connect to database after maximum attempts")
    return False

def create_table(engine):
    """Create the linear_data table if it doesn't exist"""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS linear_data (
        id SERIAL PRIMARY KEY,
        x DOUBLE PRECISION NOT NULL,
        y DOUBLE PRECISION NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    
    try:
        with engine.connect() as conn:
            conn.execute(text(create_table_sql))
            conn.commit()
        print("✅ Table 'linear_data' created/verified successfully")
    except Exception as e:
        print(f"❌ Error creating table: {e}")
        raise

def generate_data():
    """Generate synthetic linear data with noise"""
    print("📊 Generating synthetic data...")
    
    # Generate data: y = 2x + 1 + noise
    np.random.seed(42)
    x_values = np.linspace(0, 20, 100)
    y_values = 2 * x_values + 1 + np.random.normal(0, 0.5, 100)
    
    # Create DataFrame
    df = pd.DataFrame({
        'x': x_values,
        'y': y_values
    })
    
    print(f"✅ Generated {len(df)} data points")
    print(f"📈 Data range: x=[{df['x'].min():.2f}, {df['x'].max():.2f}], y=[{df['y'].min():.2f}, {df['y'].max():.2f}]")
    
    return df

def store_data(df, engine):
    """Store the data in PostgreSQL database"""
    print("💾 Storing data in database...")
    
    try:
        # Store data in database
        df.to_sql('linear_data', engine, if_exists='replace', index=False)
        
        # Verify data was stored
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM linear_data"))
            count = result.fetchone()[0]
        
        print(f"✅ Successfully stored {count} records in database")
        
        # Show sample data
        sample_query = "SELECT * FROM linear_data ORDER BY x LIMIT 5"
        with engine.connect() as conn:
            sample_data = pd.read_sql(sample_query, conn)
        
        print("📋 Sample data:")
        print(sample_data.to_string(index=False))
        
    except Exception as e:
        print(f"❌ Error storing data: {e}")
        raise

def main():
    """Main function to prepare and store data"""
    print("🚀 Starting data preparation pipeline...")
    
    # Wait for database to be ready
    if not wait_for_database():
        return
    
    # Database connection parameters
    db_url = f"postgresql://{os.getenv('POSTGRES_USER', 'postgres')}:{os.getenv('POSTGRES_PASSWORD', 'postgres')}@{os.getenv('DB_HOST', 'database')}:{os.getenv('DB_PORT', '5432')}/{os.getenv('POSTGRES_DB', 'dsproject')}"
    
    try:
        # Create database engine
        engine = create_engine(db_url)
        
        # Create table
        create_table(engine)
        
        # Generate data
        df = generate_data()
        
        # Store data
        store_data(df, engine)
        
        print("🎉 Data preparation pipeline completed successfully!")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        raise

if __name__ == '__main__':
    main()
