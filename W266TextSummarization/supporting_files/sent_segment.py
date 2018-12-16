# Natural Language Toolkit: code_classification_based_segmenter
# code is from http://www.nltk.org/book/ch06.html#code-classification-based-segmenter
import nltk
sents = nltk.corpus.treebank_raw.sents()
tokens = []
boundaries = set()
offset = 0
for sent in sents:
    tokens.extend(sent)
    offset += len(sent)
    boundaries.add(offset-1)


def punct_features(tokens, i):
    if i+1 >= len(tokens):
        next_cap = False
    else:
        next_cap = tokens[i+1][0].isupper()
    if i-1 <0:
        prev_lower = False
        prev_one_char = False
    else:
        prev_lower = tokens[i-1].lower()
        prev_one_char = len(tokens[i-1]) == 1
       
    return {'next-word-capitalized': next_cap,
    'prev-word': prev_lower,
    'punct': tokens[i],
    'prev-word-is-one-char': prev_one_char}
featuresets = [(punct_features(tokens, i), (i in boundaries)) for i in range(1, len(tokens)-1) if tokens[i] in '.?!;']

size = int(len(featuresets) * 0.1)
train_set, test_set = featuresets[size:], featuresets[:size]
classifier = nltk.NaiveBayesClassifier.train(train_set)
# nltk.classify.accuracy(classifier, test_set)

def segment_sentences(words):
    start = 0
    sents = []
    for i, word in enumerate(words):
        if word in '.?!;' and classifier.classify(punct_features(words, i)) == True:
            sents.append(words[start:i+1])
            start = i+1
    if start < len(words):
        sents.append(words[start:])
    return sents

