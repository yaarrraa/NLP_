import nltk
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import movie_reviews

#Classifying Text
tell_words = ['amazing', 'incredible', 'perfect', 'nice', 'fine', 'hilarious', 'excellent', 'awesome',  'good',
              'wonderful', 'powerful', 'impressive', 'fantastic', 'funny', 'awful', 'horror', 'inspiring', 'memorable',
              'mystery', 'successful','great', 'best', 'funny', 'interesting', 'bad', 'least', 'worst', 'long',
              'wrong', 'stupid', 'boring', 'predictable', 'terrible', 'scary', ]

# Отримання списку файлів з позитивними та негативними рецензіями
pos_fileids = movie_reviews.fileids('pos')
neg_fileids = movie_reviews.fileids('neg')
# Функція для отримання абзаців з рецензії
def get_paragraphs(fileids):
    paragraphs = []
    for fileid in fileids:
        # Отримуємо текст рецензії
        text = movie_reviews.raw(fileid)
        # Розбиваємо текст на речення
        sentences = sent_tokenize(text)
        # Групуємо речення у відповідні абзаци
        paragraphs.append(sentences)
    return paragraphs

# Отримання абзаців з позитивних та негативних рецензій
pos_paragraphs = get_paragraphs(pos_fileids)
neg_paragraphs = get_paragraphs(neg_fileids)
# print(pos_paragraphs[:5])
# print(len(pos_paragraphs))
# print(len(neg_paragraphs))

def flatten(paragraph):
    output = set([])
    for item in paragraph:
       if isinstance(item, (list, tuple)):
           output.update(item)
       else:
           output.add(item)
    return output

pos_flat = []
for paragraph in pos_paragraphs:
    pos_flat.append(flatten(paragraph))

neg_flat = []
for paragraph in neg_paragraphs:
    neg_flat.append(flatten(paragraph))

labeled_data = []
for paragraph in pos_flat:
    labeled_data.append((paragraph, 'positive'))

for paragraph in neg_flat:
    labeled_data.append((paragraph, 'negative'))

from random import shuffle
shuffle(labeled_data)


def define_features(paragraph):
    features = {}
    for tell_word in tell_words:
        features[tell_word] = tell_word in paragraph
    return features

feature_data = []
for labeled_paragraph in labeled_data:
    paragraph, label = labeled_paragraph
    feature_data.append((define_features(paragraph), label,))

train_data = feature_data[:1000]
test_data = feature_data[1000:]

decision_tree = nltk.NaiveBayesClassifier.train(train_data)
print(decision_tree.classify(train_data[0][0]))

decision_tree.show_most_informative_features()
print(nltk.classify.accuracy(decision_tree, test_data))
