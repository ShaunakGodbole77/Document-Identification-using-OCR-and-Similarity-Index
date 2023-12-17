import ocr_tesseract
import NLP_Text_Similarity
import process_bert_similarity
import process_jaccard_similarity
import process_tfidf_similarity
import process_use_similarity
import os 

for i in range(1,4):
    for j in range(1,7):
        total_text = ""
        filepath = os.listdir(f"test_paper_{i}\\{j}")
        for k in filepath:
            text = ocr_tesseract.image_to_text(f"test_paper_{i}\\{j}\\{k}")
            total_text = total_text + str(text)

        NLP = NLP_Text_Similarity.text_matching(total_text,i)
        bertdoc,bert = process_bert_similarity.process_bert_similarity(total_text,i)
        jacdoc,jaccard = process_jaccard_similarity.process_jaccard_similarity(total_text,i)
        tfidfdoc, tfidf = process_tfidf_similarity.process_tfidf_similarity(total_text,i)
        usedoc, use = process_use_similarity.process_use_similarity(total_text,i)
        print("**********************************************************************************************************\n")
        print(f"Results for Paper {i} for Response {j} are: \n")
        print("NLP Similarity is: ", NLP)
        print("Bert's Similarity Index is: ", bert)
        print("Jaccard's Similarity Index is: ", jaccard)
        print("TFIDF Similarity Index is: ", tfidf)
        print("USE/Cosine Similarity is: ", use,"\n")
        print("**********************************************************************************************************\n")
