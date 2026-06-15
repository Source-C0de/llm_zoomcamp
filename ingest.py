# loading data and building the search index


import requests
from minsearch import index


def laod_faq_data():
    docs_url = "https://datatalks.club/faq/json/courses.json"
    response = requests.get(docs_url)
    courses_raw = response.json()
    
    
    documents - []
    url_prefix = ""
    
    for course in courses_raw:
        cours_ulr
    
    
    
def build_index(documents):
    index = Index(
        text_fields = ["question","sections", "answer"],
        keywords_fields = ["course"]
    )
    
    index.fit(documents)
    return index