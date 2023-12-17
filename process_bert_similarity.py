import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk import sent_tokenize
import nltk

from sentence_transformers import SentenceTransformer



def process_bert_similarity(text='',num=1):
	# This will download and load the pretrained model offered by UKPLab.

	base_document = text
	#print("base document",base_document)
	f = open(f"{num}.txt","r")
	
	documents = f.read()
	#print(documents)
	documents.replace("\n"," ")
	documents = documents.split(".")
	empty = []
	for i in documents:
		if "\n" in i:
			i = i.replace("\n","")
			empty.append(i)
	while "" in empty:
		empty.remove("")
	while " " in empty:
		empty.remove(" ")
	#print(empty)
	
	#print(documents)
	model = SentenceTransformer('bert-base-nli-mean-tokens')

	# Although it is not explicitly stated in the official document of sentence transformer, the original BERT is meant for a shorter sentence. We will feed the model by sentences instead of the whole documents.
	sentences = sent_tokenize(base_document)
	base_embeddings_sentences = model.encode(sentences)
	base_embeddings = np.mean(np.array(base_embeddings_sentences,dtype=object), axis=0)

	vectors = []
	for i, document in enumerate(empty):
		#print(True)

		sentences = sent_tokenize(document)
		embeddings_sentences = model.encode(sentences)
		embeddings = np.mean(np.array(embeddings_sentences), axis=0)
		#print(embeddings)

		vectors.append(embeddings)
		#print(vectors)

		
	#print(False)
	scores = cosine_similarity([base_embeddings], vectors).flatten()

	highest_score = 0
	highest_score_index = 0
	for i, score in enumerate(scores):
		if highest_score < score:
			highest_score = score
			highest_score_index = i

	most_similar_document = documents[highest_score_index]
	#print("Most similar document by BERT with the score:", most_similar_document, highest_score)
	return most_similar_document, highest_score