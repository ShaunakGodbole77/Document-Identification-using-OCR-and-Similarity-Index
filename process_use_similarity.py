from sklearn.metrics.pairwise import cosine_similarity

import tensorflow as tf
import tensorflow_hub as hub



def process_use_similarity(text,num):
	filename = "./models/Test"
	base_document = text
	f = open(f"{num}.txt","r")
	documents = f.read()
	documents.replace("\n"," ")

	documents = documents.split(".")
	#print(documents)
	while "" in documents:
		documents.remove("")
	while " " in documents:
		documents.remove(" ")
	while("" in documents):
		documents.remove("")

	model = hub.load(filename)

	base_embeddings = model([base_document])


	embeddings = model(documents)

	scores = cosine_similarity(base_embeddings, embeddings).flatten()

	highest_score = 0
	highest_score_index = 0
	for i, score in enumerate(scores):
		if highest_score < score:
			highest_score = score
			highest_score_index = i

	most_similar_document = documents[highest_score_index]
	#print("Most similar document by USE with the score:", most_similar_document, highest_score)
	return most_similar_document,highest_score

