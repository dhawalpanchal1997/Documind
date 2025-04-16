from datetime import datetime
from database.vector_store import VectorStore
#from utils.synthesizer import Synthesizer
#from timescale_vector import client

# Initialize VectorStore
vec = VectorStore()
# --------------------------------------------------------------
# Basic search 
# --------------------------------------------------------------

def search_documents(query):
    """Perform a basic search using the VectorStore."""
    results = vec.search(query, limit=3)
    return results

