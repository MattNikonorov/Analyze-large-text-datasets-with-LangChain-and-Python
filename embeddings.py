from langchain.embeddings import OpenAIEmbeddings
from openai.embeddings_utils import cosine_similarity
import os
import pandas
import time

os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_KEY"
embeddings_model = OpenAIEmbeddings()

f = open("texts/Beyond Good and Evil.txt", "r")
phi_text = str(f.read())

chapters = phi_text.split("CHAPTER")

emb_list = []

for i in range(len(chapters)):
    embs = embeddings_model.embed_query(chapters[i])
    emb_list.append(embs)
    # time.sleep(20) if you face rate limits

embedded_question = embeddings_model.embed_query("What are the flaws of philosophers?")
similarities = [] 
tags = []
for i2 in range(len(emb_list)):
    similarities.append(cosine_similarity(emb_list[i2], embedded_question))
    tags.append(f"CHAPTER {i2}")

print(tags[similarities.index(max(similarities))])
