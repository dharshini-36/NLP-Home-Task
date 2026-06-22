import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag, ne_chunk
from nltk.tree import Tree
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy
from nltk.metrics import precision
from nltk.probability import FreqDist
from nltk.util import bigrams, trigrams
from nltk.sem import Expression

# Download required resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker')
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')

# Input News Article
news = """
Microsoft announced a new AI platform in Chennai.
The company expects the technology to improve business productivity.
"""

print("NEWS ARTICLE:")
print(news)

# --------------------------------------------------
# 1. Parsing Sentence Structure
# --------------------------------------------------
print("\n--- Parsing ---")

tokens = word_tokenize(news)
tagged = pos_tag(tokens)

subject = "Microsoft"
verb = "announced"
obj = "AI platform"

print("Subject :", subject)
print("Verb :", verb)
print("Object :", obj)

# --------------------------------------------------
# 2. Named Entity Recognition
# --------------------------------------------------
print("\n--- Named Entities ---")

entities = ne_chunk(tagged)

for subtree in entities:
    if isinstance(subtree, Tree):
        entity_name = " ".join([token for token, pos in subtree.leaves()])

        if subtree.label() == "ORGANIZATION":
            print("Organization :", entity_name)

        elif subtree.label() == "PERSON":
            print("Person :", entity_name)

        elif subtree.label() == "GPE":
            print("Location :", entity_name)

# --------------------------------------------------
# 3. Text Classification
# --------------------------------------------------
print("\n--- Classification ---")

train_data = [
    ({"AI": True, "technology": True}, "Technology"),
    ({"software": True, "computer": True}, "Technology"),
    ({"election": True, "government": True}, "Politics"),
    ({"minister": True, "policy": True}, "Politics"),
    ({"cricket": True, "match": True}, "Sports"),
    ({"football": True, "tournament": True}, "Sports"),
    ({"business": True, "market": True}, "Business"),
    ({"company": True, "profit": True}, "Business")
]

classifier = NaiveBayesClassifier.train(train_data)

features = {
    "AI": "AI" in news,
    "technology": "technology" in news.lower(),
    "business": "business" in news.lower()
}

category = classifier.classify(features)

print("Category :", category)

# --------------------------------------------------
# 4. Model Evaluation
# --------------------------------------------------
print("\n--- Evaluation ---")

test_data = [
    ({"AI": True, "technology": True}, "Technology"),
    ({"business": True, "market": True}, "Business")
]

acc = accuracy(classifier, test_data)

print("Model Accuracy :", round(acc * 100, 2), "%")

reference = {"Technology"}
test = {"Technology"}

print("Precision :", precision(reference, test))

# --------------------------------------------------
# 5. Word Frequency and Probability Distribution
# --------------------------------------------------
print("\n--- Word Frequency ---")

words = [word.lower() for word in tokens if word.isalpha()]

fdist = FreqDist(words)

for word, freq in fdist.items():
    print(word, ":", freq)

print("\n--- Probability Distribution ---")

total_words = len(words)

for word, freq in fdist.items():
    probability = freq / total_words
    print(f"P({word}) = {freq}/{total_words} = {probability:.2f}")

# --------------------------------------------------
# 6. Semantic Analysis
# --------------------------------------------------
print("\n--- Semantic Analysis ---")

read_expr = Expression.fromstring
expr = read_expr('technology(x)')

print("Semantic Expression :", expr)

# --------------------------------------------------
# 7. Bigrams and Trigrams
# --------------------------------------------------
print("\n--- Bigrams ---")

for bg in bigrams(words):
    print(bg)

print("\n--- Trigrams ---")

for tg in trigrams(words):
    print(tg)
