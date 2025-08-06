# Logger configuration for the Thesis experiments
import logging

# Configure logging to only output to console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)