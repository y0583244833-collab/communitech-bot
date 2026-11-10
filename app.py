import streamlit as st
from rag import build_query_engine, ask

st.set_page_config(page_title="CommuniTech - עוזר החוגים", page_icon="🎨")

# יישור לימין עבור עברית
st.markdown("""
<style>
.stApp, .stChatMessage, .stMarkdown, .stCaption, textarea {
    direction: rtl;
    text-align: right;
}
</style>
""", unsafe_allow_html=True)

st.title("🎨 עוזר החוגים של CommuniTech")
st.caption("שאלו אותי על החוגים, המחירים, הימים והשעות, ונוהל ההרשמה במרכז.")


# טעינת המנוע פעם אחת בלבד (ולא מחדש בכל שאלה)
@st.cache_resource
def get_engine():
    return build_query_engine()


engine = get_engine()

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "שלום וברוכים הבאים למרכז החוגים CommuniTech! 👋\n\n"
                   "אפשר לשאול אותי על החוגים, המחירים, הימים והשעות, ונוהל ההרשמה. במה אפשר לעזור?"
    }]

def show_sources(sources):
    with st.expander("📄 מקורות התשובה"):
        for s in sources:
            st.markdown(f"**{s['file']}** (ציון: {s['score']:.3f})")
            st.text(s["text"][:300])


# הצגת ההודעות הקודמות
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            show_sources(msg["sources"])

# שאלה חדשה
question = st.chat_input("כתבו שאלה...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("מחפש במאגר..."):
            try:
                answer, sources = ask(engine, question)
            except Exception as e:
                answer, sources = f"אירעה שגיאה: {e}", []

        # כשאין תשובה במאגר, אין טעם להציג מקורות לא רלוונטיים
        if "אין לי מידע" in answer:
            sources = []

        st.markdown(answer)
        if sources:
            show_sources(sources)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})