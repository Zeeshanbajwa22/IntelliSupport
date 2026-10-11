from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vs = Chroma(persist_directory="user_data/eval_user", embedding_function=emb)

q = "Which university is this project submitted to?"
for i, (doc, score) in enumerate(vs.similarity_search_with_score(q, k=8), 1):
    has_iqra = "IQRA" in doc.page_content.upper()
    print(f"{i}. page {doc.metadata.get('page_label')} | score {score:.3f} | contains IQRA: {has_iqra}")