import os
from dotenv import load_dotenv
import chromadb
from llama_index.core import VectorStoreIndex, Settings, PromptTemplate
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.vector_stores.chroma import ChromaVectorStore

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

# ===== הגדרות (כאן נמצאות ההחלטות שלך) =====
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "communitech"
EMBED_MODEL = "gemini-embedding-001"
# LLM_MODEL = "gemini-3.5-flash"
LLM_MODEL = "gemini-3.5-flash-lite"
TOP_K = 5                    # החלטה 6: כמה קטעים לשלוף
USE_QUERY_EXPANSION = True   # החלטה 8: הרחבת שאלה
NUM_QUERIES = 3              # השאלה המקורית + 2 ניסוחים חלופיים

# חייב להיות אותו מודל embedding שבנה את האינדקס!
Settings.embed_model = GoogleGenAIEmbedding(model_name=EMBED_MODEL, api_key=API_KEY)
Settings.llm = GoogleGenAI(model=LLM_MODEL, api_key=API_KEY, temperature=0.1)

# ההוראות למודל: לענות רק מהמידע שנשלף
# QA_PROMPT = PromptTemplate(
#     "אתה נציג שירות של מרכז החוגים הקהילתי CommuniTech.\n"
#     "ענה על שאלת הלקוח אך ורק על סמך המידע שבהקשר שלמטה.\n"
#     "כללים:\n"
#     "- אם התשובה לא מופיעה בהקשר, כתוב בדיוק: \"אין לי ידע על זה. יש לך שאלה נוספת?\" ואל תנחש.\n"
#     "- אל תשתמש בידע כללי מחוץ להקשר.\n"
#     "- בשאלה על הרשמה לשני חוגים באותו יום: חשב את שעות ההתחלה והסיום של כל קבוצה, "
#     "ובדוק לפי כלל החפיפה (מותרת חפיפה של עד 5 דקות בלבד). פרט את החישוב בקצרה.\n"
#     "- כשאתה מתייחס לקבוצה, תאר אותה לפי הרמה, היום והשעה (למשל: 'קבוצת המתחילות ביום שלישי ב-16:00').\n"
#     "- אם הלקוח מזכיר רמה שיש לה כמה קבוצות, או לא מציין קבוצה בכלל, פרט את כל הקבוצות המתאימות עם הימים והשעות שלהן.\n"
#     "- ענה בעברית, בצורה ברורה וקצרה.\n\n"
#     "הקשר:\n---------------------\n{context_str}\n---------------------\n"
#     "שאלה: {query_str}\n"
#     "תשובה: "
# )
QA_PROMPT = PromptTemplate(
    "אתה נציג שירות של מרכז החוגים הקהילתי CommuniTech. ענה בטון נעים, אדיב ועניני, "
    "כמו נציגה מקצועית ונחמדה. אל תחמיא ללקוח, אל תגזים בהתלהבות ואל תשתמש בסימני קריאה מרובים. "
    "אל תפתח בברכות כמו 'שלום' או 'ברוכים הבאים'. גש ישר לתשובה.\n"
    "כללי ההרשמה הקבועים של המרכז:\n"
    "1. ניתן להירשם לקבוצה אחת בלבד מכל חוג. אי אפשר להירשם לשתי קבוצות של אותו חוג "
    "(למשל גם למתחילות וגם למתקדמות), גם אם השעות שלהן לא חופפות.\n"
    "2. אסור להירשם לשני חוגים שונים באותו יום אם השעות שלהם חופפות. מותרת חפיפה של עד 5 דקות בלבד.\n\n"

    "הנחיות לתשובה:\n"
    "- בשאלה על הרשמה לשתי קבוצות: בדוק קודם את כלל 1. רק אם מדובר בשני חוגים שונים, בדוק את כלל 2.\n"
    "- אל תציג חישובים של שעות. אם אפשר להירשם, ענה בקצרה ובחום, למשל: "
    "\"כן, אפשר להירשם לשניהם. תיהנו משני החוגים!\"\n"
    "- אם אי אפשר להירשם, ענה בקצרה והסבר במשפט אחד, למשל: "
    "\"לא, אפשר להירשם לקבוצה אחת בלבד מכל חוג.\" או \"לא, השעות של שני החוגים חופפות.\"\n"
    "- כשהתשובה כוללת כמה חוגים, סדר אותה לפי חוג: שם החוג בהדגשה, ומתחתיו רשימה של הקבוצות שלו "
    "(רמה, יום ושעות, גילאים). אחר כך החוג הבא.\n"
    "- כשאתה מתייחס לקבוצה, תאר אותה לפי הרמה, היום והשעה.\n"
    "- אם הלקוח מזכיר רמה שיש לה כמה קבוצות, או לא מציין קבוצה בכלל, פרט את כל הקבוצות המתאימות.\n"
    "- אם המידע בהקשר חלקי, ענה על מה שידוע וציין מה לא צוין. למשל, אם המחיר החודשי לא צוין "
    "אבל יש תשלום נוסף, אמור שהמחיר החודשי לא צוין, וציין את התשלום הנוסף.\n"
    "- רק אם אין בהקשר שום מידע שקשור לשאלה, כתוב בדיוק: \"אין לי ידע על זה. יש לך שאלה נוספת?\" ואל תנחש.\n"
    "- אל תשתמש בידע כללי מחוץ להקשר.\n"
    "- ענה בעברית, בצורה ברורה וקצרה.\n\n"

    "הקשר:\n---------------------\n{context_str}\n---------------------\n"
    "שאלה: {query_str}\n"
    "תשובה: "
)

# ההוראות להרחבת השאלה (בעברית, אחרת המודל ינסח באנגלית)
QUERY_GEN_PROMPT = (
    "אתה עוזר שמנסח שאילתות חיפוש עבור מאגר מידע של מרכז חוגים.\n"
    "נסח {num_queries} ניסוחים חלופיים בעברית לשאלה הבאה, כל ניסוח בשורה נפרדת.\n"
    "שמור על כל פרט מהשאלה המקורית: שמות חוגים, ימים, שעות, גילאים ומספרים. אל תוסיף מידע חדש.\n"
    "שאלה: {query}\n"
    "ניסוחים:\n"
)


def build_query_engine():
    # טעינת האינדקס הקיים מ-Chroma (לא בונים מחדש!)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    index = VectorStoreIndex.from_vector_store(vector_store)

    # החלטה 9: חיפוש וקטורי לפי דמיון
    base_retriever = index.as_retriever(similarity_top_k=TOP_K)

    if USE_QUERY_EXPANSION:
        retriever = QueryFusionRetriever(
            [base_retriever],
            llm=Settings.llm,
            similarity_top_k=TOP_K,
            num_queries=NUM_QUERIES,
            mode="reciprocal_rerank",
            use_async=False,
            query_gen_prompt=QUERY_GEN_PROMPT,
            verbose=True,  # מדפיס את הניסוחים שנוצרו
        )
    else:
        retriever = base_retriever

    return RetrieverQueryEngine.from_args(retriever, text_qa_template=QA_PROMPT)


def ask(engine, question):
    response = engine.query(question)
    sources = [
        {"file": n.node.metadata.get("file_name"), "score": n.score, "text": n.node.text}
        for n in response.source_nodes
    ]
    return str(response), sources


# בדיקה מהירה בטרמינל, לפני שבונים את הממשק
if __name__ == "__main__":
    engine = build_query_engine()
    while True:
        question = input("\nשאלה (או 'יציאה'): ")
        if question.strip() == "יציאה":
            break
        answer, sources = ask(engine, question)
        print(f"\nתשובה:\n{answer}\n\nמקורות:")
        for s in sources:
            print(f"  - {s['file']} (ציון: {s['score']:.3f})")