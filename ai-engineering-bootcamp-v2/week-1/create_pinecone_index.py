"""One-time setup: create the Pinecone index this app expects.

Run once from this folder (after PINECONE_API_KEY and PINECONE_INDEX_NAME are in .env):
    .venv\\Scripts\\python create_pinecone_index.py

Why a script instead of the console form: the SDK's dimension=/metric= arguments create
an index with a single reserved dense field (`_values`), which is exactly what
vector_store.py's upsert(values=...) / query(vector=...) calls expect. Safe to re-run.
"""

from pinecone import Pinecone, ServerlessSpec

from vector_store import EMBEDDING_DIM, EMBEDDING_MODEL, get_settings, pinecone_health

settings = get_settings()
pc = Pinecone(api_key=settings.api_key)

if pc.has_index(settings.index_name):
    print(f"Index '{settings.index_name}' already exists - not creating it again.")
else:
    print(f"Creating '{settings.index_name}' (dim {EMBEDDING_DIM}, cosine, for {EMBEDDING_MODEL})...")
    pc.create_index(
        name=settings.index_name,
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),  # Free Starter plan region.
        dimension=EMBEDDING_DIM,
        metric="cosine",
    )  # Waits until the index is ready.
    print("Created.")

print(pinecone_health())
