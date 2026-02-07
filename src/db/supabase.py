from supabase import create_client
from src.config.db_config import SUPABASE_URL, SUPABASE_SERVICE_KEY

def connect_to_db():
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise RuntimeError("Supabase config variables not set")
    supabase = create_client(
        SUPABASE_URL,
        SUPABASE_SERVICE_KEY
    )

    return supabase

supabase = connect_to_db
