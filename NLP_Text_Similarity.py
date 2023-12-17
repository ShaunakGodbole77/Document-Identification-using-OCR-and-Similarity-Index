import spacy

nlp = spacy.load("en_core_web_lg")

def text_matching(filepath,num):
    answer1 = filepath
    f = open(f"{num}.txt","r")
    answer2 = f.read()

    answer1 = nlp(answer1)
    answer2 = nlp(answer2)

    answer1.similarity(answer2)

    answer1 = " ".join([token.lemma_ for token in answer1])
    answer2 = " ".join([token.lemma_ for token in answer2])

    percentage = nlp(answer1).similarity(nlp(answer2))
    #print(percentage)
    return percentage