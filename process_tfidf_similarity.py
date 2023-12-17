from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



def process_tfidf_similarity(text,num):
	vectorizer = TfidfVectorizer()
	#f = open("model_ans.txt","r")
	base_document = text
	#f.close()
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

	# To make uniformed vectors, both documents need to be combined first.
	documents.insert(0, base_document)
	embeddings = vectorizer.fit_transform(documents)

	cosine_similarities = cosine_similarity(embeddings[0:1], embeddings[1:]).flatten()

	highest_score = 0
	highest_score_index = 0
	for i, score in enumerate(cosine_similarities):
		if highest_score < score:
			highest_score = score
			highest_score_index = i


	most_similar_document = documents[highest_score_index]

	#print("Most similar document by TF-IDF with the score:", most_similar_document, highest_score)
	return most_similar_document,highest_score
