import spacy 
nlp = spacy.load("en_core_web_md")

word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")
print(word1.similarity(word2))
print(word3.similarity(word2))
print(word3.similarity(word1))
print()
print()

tokens = nlp('cat apple monkey banana ')

for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))
print()
print()

sentence_to_compare = "Why is my cat on the car"


sentences = ["where did my dog go", "Hello, there is my car", "I\'ve lost my car in my car", 
             "I\'d like my boat back", "I will name my dog Diana"]

model_sentence = nlp(sentence_to_compare)

for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence + " - ", similarity)

'''I find it interesting that the model can determine the animals and the link between different objects 
to those animals. ie the banana had a higher similarity rate to the monkey than the cat

The similarity values were much lower on sm vs md - The sm is a more lightweight model and md is a more intermediate model'''