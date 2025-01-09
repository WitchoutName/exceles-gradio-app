from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Define Python variables based on .env file
BAG_UPLOAD_DIR = os.getenv('BAG_UPLOAD_DIR')
DATA_BASE_DIR = os.getenv('DATA_BASE_DIR')
DEBUG = os.getenv('DEBUG') == 'True'