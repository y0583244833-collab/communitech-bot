import os
from pathlib import Path
import time
from dotenv import load_dotenv
import chromadb
from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

load_dotenv()

DATA_DIR = Path("data/communitech_docs")
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "communitech"

# מודל ה-embedding
Settings.embed_model = GoogleGenAIEmbedding(
    model_name="gemini-embedding-001",
    api_key=os.getenv("GOOGLE_API_KEY"),
)

# 1. טעינת המסמכים (קריאה מפורשת ב-UTF-8 כדי למנוע בעיות עברית ב-Windows)
docs = []
for path in sorted(DATA_DIR.rglob("*.md")):
    docs.append(Document(
        text=path.read_text(encoding="utf-8"),
        metadata={"file_name": path.name},
    ))
print(f"נטענו {len(docs)} מסמכים")

# 2. חלוקה לקטעים לפי כותרות Markdown
parser = MarkdownNodeParser()
nodes = parser.get_nodes_from_documents(docs)
print(f"נוצרו {len(nodes)} קטעים")

# הצצה לקטע לדוגמה, כדי לוודא שהחלוקה הגיונית
print("\n--- קטע לדוגמה ---")
print(nodes[10].text)
print("------------------\n")

# 3. שמירה ב-ChromaDB על הדיסק (מוחקים אוסף קודם כדי לא לשכפל קטעים)
client = chromadb.PersistentClient(path=CHROMA_DIR)
try:
    client.delete_collection(COLLECTION_NAME)
except Exception:
    pass
collection = client.get_or_create_collection(COLLECTION_NAME)

vector_store = ChromaVectorStore(chroma_collection=collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# 4. יצירת הווקטורים 
# 4. יצירת הווקטורים בקבוצות, בגלל מגבלה של 100 בקשות לדקה בשכבה החינמית
BATCH_SIZE = 80
index = VectorStoreIndex([], storage_context=storage_context)

for i in range(0, len(nodes), BATCH_SIZE):
    batch = nodes[i:i + BATCH_SIZE]
    print(f"שולחת קטעים {i + 1}-{i + len(batch)} מתוך {len(nodes)}...")
    index.insert_nodes(batch)
    if i + BATCH_SIZE < len(nodes):
        print("ממתינה דקה בגלל מגבלת הקצב...")
        time.sleep(61)
print(f"\nהאינדקס נשמר. מספר הווקטורים ב-Chroma: {collection.count()}")