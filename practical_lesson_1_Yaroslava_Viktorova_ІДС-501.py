import nltk
from nltk.corpus import gutenberg
import string
nltk.download('stopwords')
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from nltk.text import Text
nltk.download('gutenberg')
nltk.download('punkt')



# print(list(gutenberg.fileids()))
# emma_text = gutenberg.raw('austen-emma.txt')
# print(emma_text[:500])

# по словам
hamlet_w = gutenberg.words('shakespeare-hamlet.txt')
print(hamlet_w )

# по реченням
hamlet_s = gutenberg.sents('shakespeare-hamlet.txt')
print(hamlet_s)

# по параграфам
hamlet_p = gutenberg.paras('shakespeare-hamlet.txt')
print(hamlet_p)

# Frequency Distributions
# слова, які найчастіше зустрічаються в тексті (разом з пунктуацією)
hamlet_dict = nltk.FreqDist(hamlet_w) #словник слово/скільки раз трапляється
print(hamlet_dict.most_common(15))

print(string.punctuation)

# відфільтруємо текст (прибираємо пунктуацію)
hamlet_without_punct = [] #текст без пунктуації
for word in hamlet_w:
    if word not in string.punctuation:
        hamlet_without_punct.append(word)

print(len(hamlet_w))  # довжина тексту з пунктуацію
print(len(hamlet_without_punct))  # довжина тексту без пунктуацію

hamlet_dict_no_punct = nltk.FreqDist(hamlet_without_punct) #словник без пунктуації слово/скільки раз трапляється
print(hamlet_dict_no_punct.most_common(10))

english_stopwords = stopwords.words('english')
print(english_stopwords[:10])

hamlet_without_stop_word = []
for word in hamlet_without_punct:
    if word.lower() not in english_stopwords:
        hamlet_without_stop_word.append(word)
print(len(hamlet_without_punct))
print(len(hamlet_without_stop_word))

hamlet_dict_no_stop_word = nltk.FreqDist(hamlet_without_stop_word)
print(hamlet_dict_no_stop_word.most_common(10))

print(hamlet_dict_no_stop_word.tabulate(25))

#графік частота слів, які найчастіше зустрічаються
fig = plt.figure()
plt.gcf().subplots_adjust(bottom=0.30) # to avoid x-ticks cut-off

hamlet_dict_no_stop_word.plot(10)
plt.tight_layout()

#Text Objects
hamlet_t = Text(hamlet_w)
hamlet_t.concordance('Hamlet', lines=5)
hamlet_t.collocations(num=5) #слова які найчастіше зустрічаються разом

fig_1 = plt.figure()
plt.gcf().subplots_adjust(bottom=0.30) # to avoid x-ticks cut-off

hamlet_t.dispersion_plot(['Lord', 'King', 'Hamlet'])
plt.show()
