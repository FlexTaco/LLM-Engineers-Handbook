import torch
from pymongo import MongoClient
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from transformers import AutoModel, AutoTokenizer

# change to your mongo_uri and qdrant_host, etc
MONGO_CONNECTION = ""
DATABASE_NAME = "vector_store"
RAW_DOCS_COLLECTION = "incoming_documents"
PROCESSED_DOCS_COLLECTION = "completed_documents"

QDRANT_SERVER = "localhost"
QDRANT_PORT_NUMBER = 8001
VECTOR_COLLECTION_NAME = "document_embeddings"

EMBEDDING_MODEL = "bert-base-uncased"

# connections
mongo_connection = MongoClient(MONGO_CONNECTION)
mongo_database = mongo_connection[DATABASE_NAME]
incoming_documents = mongo_database[RAW_DOCS_COLLECTION]
completed_documents = mongo_database[PROCESSED_DOCS_COLLECTION]
qdrant_database = QdrantClient(host=QDRANT_SERVER, port=QDRANT_PORT_NUMBER)

tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
embedding_model = AutoModel.from_pretrained(EMBEDDING_MODEL)


def create_vector_embedding(input_text):
    tokenized_input = tokenizer(
        input_text, return_tensors="pt", truncation=True, padding=True, max_length=512
    )
    with torch.no_grad():
        embedding_output = (
            embedding_model(**tokenized_input)
            .last_hidden_state.mean(dim=1)
            .squeeze()
            .numpy()
        )
    return embedding_output


def process_and_store_documents():
    qdrant_database.recreate_collection(
        collection_name=VECTOR_COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

    unprocessed_docs = list(incoming_documents.find({"processed": {"$ne": True}}))
    if not unprocessed_docs:
        print("No unprocessed documents found.")
        return

    for document in unprocessed_docs:
        content = document.get("content") or document.get("link", "No Content Provided")
        embedding_vector = create_vector_embedding(content)
        vector_point = PointStruct(
            id=document["_id"],
            vector=embedding_vector,
            payload={"origin": document["origin"]},
        )
        qdrant_database.upsert(
            collection_name=VECTOR_COLLECTION_NAME, points=[vector_point]
        )

        processed_entry = {
            "_id": document["_id"],
            "content": content,
            "vector_id": str(document["_id"]),
            "origin": document["origin"],
            "processed": True,
        }
        completed_documents.insert_one(processed_entry)

    print(f"Successfully processed and stored {len(unprocessed_docs)} documents.")


if __name__ == "__main__":
    process_and_store_documents()
