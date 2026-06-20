# loading data and building the search index
import requests
from minsearch import Index
from gitsource import GithubRepositoryDataReader, chunk_documents



def load_documents():
    reader = GithubRepositoryDataReader(
        repo_owner="DataTalksClub",
        repo_name="llm-zoomcamp",
        commit_id="8c1834d",
        allowed_extensions={"md"},
        filename_filter=lambda path: "/lessons/" in path,
    )
    files = reader.read()

    documents = []

    for file in files:
        doc = file.parse()
        documents.append(doc)
        
    return documents
def build_index(documents):
    
    chunk = chunk_documents(
        documents, 
        size=2000,
        step=1000
    )
    print(len(documents))
    print(len(chunk))    
    index = Index(
        text_fields=["content"],
        keyword_fields=["filename"]
    )
    index.fit(chunk)
    # result = index.search(
    #     "How does the agentic loop keep calling the model until it stops?",
    #     num_results=5,
    #     boost_dict={"contest":2}
        
    # )
    return index


build_index(load_documents())
