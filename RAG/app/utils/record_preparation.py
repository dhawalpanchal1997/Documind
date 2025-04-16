from typing import List, Tuple
from timescale_vector.client import uuid_from_time
from datetime import datetime
import pandas as pd

def prepare_record(processed_chunks: List[Tuple[int, str]], source: str, vec) -> pd.DataFrame:
    """Prepare records for insertion into the database."""
    records = []
    for chunk_id, chunk in processed_chunks:
        record = {
            "id": str(uuid_from_time(datetime.now())),
            "metadata": {
                "source": source,
                "chunk_id": chunk_id,
                "chunk_size": len(chunk),
                "created_at": datetime.now().isoformat(),
            },
            "content": chunk,
            "embedding": vec.get_embedding(chunk),  # Use the passed VectorStore instance
        }
        records.append(record)

    return pd.DataFrame.from_records(records)
