from gitsource import GithubRepositoryDataReader
from gitsource import chunk_documents

reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)

documents = [file.parse() for file in reader.read()]
chunk = chunk_documents(
    documents,
    size=2000,
    step=1000
)

if __name__ == "__main__":
    scores = X.dot(v)

    print(scores)