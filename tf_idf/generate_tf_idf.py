import math
import time
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import collections


# File format: Text TAB Label

# Returns the number of documents that contain the given word.
def getDocFreq(wordDocFrequency, listDocumentName, word):
    counter = 0
    for doc in listDocumentName:
        wordDoc = doc + "#" + word
        if wordDocFrequency.has_key(wordDoc):
            counter += 1

    return counter


# Returns the Idf(t, D).
def getIDF(listDocumentName, wordDocFrequency, listWord):
    idf = collections.OrderedDict()
    n = len(listDocumentName)

    for word in listWord:
        idf[word] = float(math.log10(float(n) / float(getDocFreq(wordDocFrequency, listDocumentName, word))))
    return idf


# Returns the TfIdf(t, D).
def calcTfIdf(listDocumentName, wordDocFrequency, listWord, docFreq):
    tfIdf = collections.OrderedDict()
    idf = getIDF(listDocumentName, wordDocFrequency, listWord)

    for doc in listDocumentName:
        for word in listWord:
            wordDoc = doc + "#" + word
            if wordDocFrequency.has_key(wordDoc):
                tfIdf[wordDoc] = float(float(idf[word]) * float(wordDocFrequency[wordDoc]) / float (docFreq[doc]))
                # tfIdf[wordDoc] = float(float(idf[word]) * float(wordDocFrequency[wordDoc]))
            else:
                tfIdf[wordDoc] = 0.0
    return tfIdf


listDocumentName = []

# Stores the count of a given word in a given document.
wordDocFrequency = collections.OrderedDict()

# List of words
listWord = []

# Stores the number of words in each document.
docFreq = collections.OrderedDict()
wordFreq = collections.OrderedDict()

wordnet_lemmatizer = WordNetLemmatizer()


def generate_util(inputFile):
    with open(inputFile, "r+") as readFile:

        i = 0
        print "Generating TfIdf .... \n "
        start_time = time.time()
        # Calculates the TfIdf for each word across documents.

        stop_list = set(stopwords.words('english'))
        new_stop_list = list(stop_list - {'who', 'what', 'when', 'how', 'where', 'why'})

        for row in readFile:
            row = row.lower()
            l = row.split("\t")[0].strip()
            label = row.split("\t")[1]
            label = label.strip('\n').strip(' ')

            # words = []
            words = l.split(" ")

            # Updates the list of documents(goals).
            if not label in listDocumentName:
                listDocumentName.append(label)

            # Updates the word frequency
            for w in words:
                w.strip('\n')
                w = w.lower()
                w1 = wordnet_lemmatizer.lemmatize(w, 'v')
                w2 = wordnet_lemmatizer.lemmatize(w, 'n')

                if (w != w1):
                    w = w1
                elif (w != w2):
                    w = w2

                if w in new_stop_list or w == "n't" or w == "'s" or w == "'re" or w == 'u' or w == 'ur':
                    # print w
                    continue

                if wordFreq.has_key(w):
                    wordFreq[w] = wordFreq[w] + 1
                else:
                    wordFreq[w] = 1

                if not w in listWord:
                    listWord.append(w)

                # Updates the count of a given word in a given document(goal).
                wordDoc = label + "#" + w

                if wordDocFrequency.has_key(wordDoc):
                    count = wordDocFrequency[wordDoc]
                    wordDocFrequency[wordDoc] = count + 1
                else:
                    wordDocFrequency[wordDoc] = 1

                # Updates the word count for each document(goal).
                if docFreq.has_key(label):
                    docFreq[label] += 1
                else:
                    docFreq[label] = 1
            i += 1

        tfIdf = calcTfIdf(listDocumentName, wordDocFrequency, listWord, docFreq)
        elapsed_time = time.time() - start_time
        print "TfIdf Generation Time : " + str(elapsed_time) + "s\n"

        goal_to_words_map = collections.OrderedDict()
        for a in tfIdf:
            goal = a.split("#")[0]
            word = a.split("#")[1]
            if goal_to_words_map.has_key(goal):
                goal_to_words_map[goal].append((tfIdf[a], word))
            else:
                goal_to_words_map[goal] = [(tfIdf[a], word)]

        topw = open(inputFile + '_tf_idf_map', 'w')
        goal_map_file = open(inputFile + '_tf_idf_goal_map', 'w')

        tf_map = collections.defaultdict(set)

        for goal in goal_to_words_map:
            goal_to_words_map[goal] = sorted(goal_to_words_map[goal])
            topw.write(str(goal) + " => ")

            goal_map_file.write(str(goal) + '\t')

            word_string = set()
            goal_string = ''
            for word in reversed(goal_to_words_map[goal][-4:]):
                topw.write("(" + str(word[1]) + "," + str(word[0]) + ")" + ",")
                word_string.add(word[1])
                goal_string += word[1] + ' '
            for key in tf_map:
                if (len(set(key) - set(word_string)) == 0):
                    print word_string

            if (not tf_map.has_key(tuple(word_string))):
                tf_map[tuple(word_string)] = 1

            goal_map_file.write(goal_string.strip() + '\n')

            topw.write("\n\n")
        topw.flush()
        topw.close()
        goal_map_file.close()
        return goal_to_words_map


generate_util('input_file_path')
