# Data Directories

## Structure

- `raw/`: Contains raw, unprocessed data files
- `processed/`: Contains cleaned and preprocessed data files

## Usage

Place your raw salary data CSV files in the `raw/` directory.
Processed data will be automatically saved to the `processed/` directory after running preprocessing scripts.

## Data Format

Expected columns in the raw data:
- `years_experience`: Number of years of professional experience
- `education_level`: Education level (e.g., Bachelor, Master, PhD)
- `job_title`: Job title or level (e.g., Junior, Mid-Level, Senior, Lead)
- `location`: Geographic location
- `company_size`: Size of the company (e.g., Small, Medium, Large)
- `industry`: Industry sector
- `salary`: Target variable - annual salary in USD
