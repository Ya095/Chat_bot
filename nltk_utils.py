import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np
# nltk.download('punkt')


def tokenize(sentence):
    """
        Создание метода токенезации, который получает предложение и возвращает массив слов
    """
    return nltk.word_tokenize(sentence)


stemmer = PorterStemmer()
def stem(word):
    """
        Создание метода stemming с приведением к нижнему регистру. Приведение к начальной форме слова
        stemming = find the root form of the word
        пример:
        words = ["organize", "organizes", "organizing"]
        words = [stem(w) for w in words]
        -> ["organ", "organ", "organ"]
    """
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, all_words):
    """
        Возвращает список bag_of_words:
        1 для каждого известного слова, существующего в предложении, иначе - 0
        пример:
        sentence = ["hello", "how", "are", "you"]
        words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
        bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0

    return bag
