"""
RAG pipeline: load the team-description PDF (one team per page),
embed each page, and support similarity search over it.
"""

import re
import numpy as np
import faiss
from langchain_community.document_loaders import PyPDFLoader

from config import client, EMBEDDING_MODEL


def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()  # one Document per page = one Document per team

    for doc in documents:
        first_line = doc.page_content.strip().split("\n")[0]
        doc.metadata["team"] = first_line.strip()
        doc.metadata["char_count"] = len(doc.page_content)

    return documents


def check_pages(documents, max_expected_chars=2500):
    # Sanity check: flag any page that looks like it might contain more than
    # one team, or got split awkwardly during PDF export.
    warnings_found = []
    for doc in documents:
        if doc.metadata["char_count"] > max_expected_chars:
            msg = (f"Page for '{doc.metadata['team']}' has "
                   f"{doc.metadata['char_count']} chars — check for split/merge issues")
            warnings_found.append(msg)
    return documents, warnings_found


def clean_text(text):
    text_lower = text.lower()
    text_normalized_tabs = re.sub(r'(\t)+', '', text_lower)
    text_clean = re.sub(r'\s+', ' ', text_normalized_tabs)
    return text_clean.strip()


def prepare_docs(documents):
    # Keep both a clean version (for embedding) and the original (for display).
    for doc in documents:
        doc.metadata["display_text"] = doc.page_content
        doc.page_content = clean_text(doc.page_content)
    return documents


def create_embeddings(documents):
    embeddings = []
    for doc in documents:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=doc.page_content
        )
        embeddings.append(response.data[0].embedding)
    return np.array(embeddings)


def build_vector_db(documents, embeddings):
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(embeddings.astype('float32'))
    return index, documents


def search(query, index, documents, k=3):
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=clean_text(query)
    )
    query_vector = np.array([response.data[0].embedding]).astype('float32')

    distances, indices = index.search(query_vector, k)

    results = []
    for idx, dist in zip(indices[0], distances[0]):
        doc = documents[idx]
        results.append({
            "team": doc.metadata["team"],
            "text": doc.metadata["display_text"],
            "distance": float(dist)
        })
    return results


def run_pipeline(pdf_path):
    documents = load_pdf(pdf_path)
    documents, page_warnings = check_pages(documents)
    documents = prepare_docs(documents)
    embeddings = create_embeddings(documents)
    index, documents = build_vector_db(documents, embeddings)
    return index, documents, page_warnings
