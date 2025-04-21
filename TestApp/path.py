# This file is created to store the path used in the projects.
import logging
import os

# Define the base directory for your project
BASE_DIR    = r"C:\Users\clint\OneDrive\coding\Django_backend"  # Use the raw string (r) to handle backslashes

# Define paths relative to the base directory
DATASET_DIR = os.path.join(BASE_DIR, 'dataset_test')
MODELS_DIR  = os.path.join(BASE_DIR, 'models')
OUTPUT_DIR  = os.path.join(BASE_DIR, 'output')
LOGS_DIR    = os.path.join(BASE_DIR, 'logs')

# logger 
logger      = logging.getLogger('django')

# Host 
localhost   = "http://127.0.0.1:8000/api/classify/"

