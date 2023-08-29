import nltk
import re
from string import punctuation
import math
import random
import sys
import pickle

'''
    Language model constructed by Yanzheng Wu and Linghe Wang
    Our program runs on the ASCII text. we employ the data clean methods by removing punctuations, numbers, 
    dollar signs, whitespace and line breaks. For the program running with the test flag, we build our language models 
    in trigrams with Katz Backoff. For the program running without the test flag, we build our language models in 
    bigrams. 80 percents of training text and 20 percent develop text are randomly selected. For the smoothing method
    we use Good-Turing discounting. We improved results by building models in trigram and Katz Backoff. 
    result from bigrams:
    Results on dev set:
    austen       47  / 100 correct
    dickens      39  / 100 correct
    tolstoy      48  / 100 correct
    wilde        27  / 100 correct
    
    result from trigrams:
    austen       60  / 100 correct
    dickens      35  / 100 correct
    tolstoy      53  / 100 correct
    wilde        26  / 100 correct
'''


def clean(s):
    text_no_punc = re.sub(r'[{}]+'.format(punctuation), "", s)
    text_no_num = re.sub("[0-9]+", ' ', text_no_punc)
    text_no_dollar = re.sub(r'\$\w*', '', text_no_num)
    text_no_whitespace = re.sub(r'\s\s+', ' ', text_no_dollar)
    text_clean = re.sub('\n', ' ', text_no_whitespace)
    return text_clean.lower()


def smooth(dic1, dic2, preword):
    count_den = 304000000
    count_num = 0
    for value in dic1.values():
        if value == 1:
            count_num += 1
        else:
            continue
    return (((0 + 1) * count_num) / count_den) / dic2[preword]


def biagram_prob(train_file):
    dict_hd = open("words")
    dict_words = dict_hd.read()
    result_dic = {"austen": 0, "dickens": 0, "tolstoy": 0, "wilde": 0}
    test_list = []
    author_list = []
    tokens = []
    a = 0
    print("training... (this may take a while)")
    fh = open(train_file)
    texts = fh.read()
    # texts = "austen.txt\ndickens.txt\ntolstoy.txt\nwilde.txt"
    texts = texts.rstrip("\n")
    for text in texts.split("\n"):
        author_list.append(text.rstrip(".txt"))
        in_text = open(text, 'r')
        content = in_text.read()
        text_test = ""
        text_train = ""
        content = nltk.tokenize.sent_tokenize(content)
        for line in content:
            r = random.random()
            if r < 0.8:
                text_train = text_train + line
            else:
                text_test = line + text_test
        text_train = clean(text_train)
        tokens = nltk.tokenize.word_tokenize(text_train)
        test_sentences = nltk.tokenize.sent_tokenize(text_test)
        test_list.append(test_sentences)

        for i in range(len(tokens)):
            if tokens[i] in dict_words:
                continue
            else:
                tokens[i] = "UNK"

        word_tuples = []
        for i in range(len(tokens) - 1):
            if (tokens[i], tokens[i + 1]) in word_tuples:
                continue
            else:
                word_tuples.append((tokens[i], tokens[i + 1]))

        phrase_count = {}
        for i in range(len(tokens) - 1):
            if (tokens[i], tokens[i + 1]) in phrase_count:
                phrase_count[(tokens[i], tokens[i + 1])] += 1
            else:
                phrase_count[(tokens[i], tokens[i + 1])] = 1

        word_count = {}
        for i in range(len(tokens) - 1):
            if tokens[i] in word_count:
                word_count[tokens[i]] += 1
            else:
                word_count[tokens[i]] = 1

        phase_prob = {}
        for i in word_tuples:
            phase_prob[i] = phrase_count[i] / word_count[i[0]]

        f1 = open(author_list[a] + ".pkl", 'wb')
        pickle.dump(phase_prob, f1, True)
        pickle.dump(word_count, f1, True)
        pickle.dump(phrase_count, f1, True)
        f1.close()
        a += 1
    j = 0
    while j < 4:
        for sentence in test_list[j][0:101]:
            results = []
            sentence = clean(sentence)
            k = 0
            while k < 4:
                f2 = open(author_list[k] + ".pkl", 'rb')
                two_word_prob = pickle.load(f2)
                one_word_count = pickle.load(f2)
                two_word_count = pickle.load(f2)
                test_tokens = nltk.word_tokenize(sentence)
                for token in test_tokens:
                    if (token not in dict_words) or (token not in one_word_count):
                        test_tokens[test_tokens.index(token)] = "UNK"
                prob = math.log(one_word_count[test_tokens[0]] / len(tokens), 2)
                for i in range(len(test_tokens) - 1):
                    if (test_tokens[i], test_tokens[i + 1]) not in two_word_prob:
                        prob += math.log(smooth(two_word_count, one_word_count, test_tokens[i]), 2)
                    else:
                        prob += math.log(two_word_prob[(test_tokens[i], test_tokens[i + 1])], 2)
                results.append(prob)
                k += 1
            f2.close()
            max_prob = max(results)
            author_name_index = results.index(max_prob)
            author_name = author_list[author_name_index]
            if author_name == author_list[j]:
                result_dic[author_name] += 1
            else:
                continue
        j += 1
    print("Results on dev set:")
    print("austen      ", result_dic["austen"], " / 100 correct")
    print("dickens     ", result_dic["dickens"], " / 100 correct")
    print("tolstoy     ", result_dic["tolstoy"], " / 100 correct")
    print("wilde       ", result_dic["wilde"], " / 100 correct")


def trigram_test(train_file, test_file):
    dict_hd = open("words")
    dict_words = dict_hd.read()
    author_list = []
    tokens = []
    a = 0
    print("training... (this may take a while)")
    fh = open(train_file)
    texts = fh.read()
    texts = texts.rstrip("\n")
    for text in texts.split("\n"):
        author_list.append(text.rstrip(".txt"))
        in_text = open(text, 'r')
        content = in_text.read()
        text_train = clean(content)
        tokens = nltk.tokenize.word_tokenize(text_train)

        for i in range(len(tokens)):
            if tokens[i] in dict_words:
                continue
            else:
                tokens[i] = "UNK"
        three_words_tuples = []
        for i in range(len(tokens) - 2):
            if (tokens[i], tokens[i + 1], tokens[i + 2]) in three_words_tuples:
                continue
            else:
                three_words_tuples.append((tokens[i], tokens[i + 1], tokens[i + 2]))

        two_words_tuples = []
        for i in range(len(tokens) - 1):
            if [tokens[i], tokens[i + 1]] in two_words_tuples:
                continue
            else:
                two_words_tuples.append((tokens[i], tokens[i + 1]))

        three_words_count = {}
        for i in range(len(tokens) - 2):
            if (tokens[i], tokens[i + 1], tokens[i + 2]) in three_words_count:
                three_words_count[(tokens[i], tokens[i + 1], tokens[i + 2])] += 1
            else:
                three_words_count[(tokens[i], tokens[i + 1], tokens[i + 2])] = 1

        phrase_count = {}
        for i in range(len(tokens) - 1):
            if (tokens[i], tokens[i + 1]) in phrase_count:
                phrase_count[(tokens[i], tokens[i + 1])] += 1
            else:
                phrase_count[(tokens[i], tokens[i + 1])] = 1

        word_count = {}
        for i in range(len(tokens) - 1):
            if tokens[i] in word_count:
                word_count[tokens[i]] += 1
            else:
                word_count[tokens[i]] = 1

        phase_prob = {}
        for i in two_words_tuples:
            phase_prob[i] = phrase_count[i] / word_count[i[0]]

        three_word_prob = {}
        for i in three_words_tuples:
            three_word_prob[i] = three_words_count[i] / phrase_count[i[0:2]]

        f1 = open(author_list[a] + ".pkl", 'wb')
        pickle.dump(phase_prob, f1, True)
        pickle.dump(word_count, f1, True)
        pickle.dump(three_word_prob, f1, True)
        f1.close()

        a += 1

    fh2 = open(test_file)
    test = fh2.read()
    test_list = nltk.sent_tokenize(test)
    for sentence in test_list:
        results = []
        sentence = clean(sentence)
        k = 0
        while k < 4:
            f2 = open(author_list[k] + ".pkl", 'rb')
            two_word_prob = pickle.load(f2)
            one_word_count = pickle.load(f2)
            three_word_prob = pickle.load(f2)
            test_tokens = nltk.word_tokenize(sentence)
            for token in test_tokens:
                if (token not in dict_words) or (token not in one_word_count):
                    test_tokens[test_tokens.index(token)] = "UNK"
            if len(test_tokens) == 1:
                prob = math.log(one_word_count[test_tokens[0]] / len(tokens), 2)
                results.append(prob)
                k += 1
                continue
            prob = math.log(one_word_count[test_tokens[0]] / len(tokens), 2)
            if (test_tokens[0], test_tokens[1]) in two_word_prob:
                prob += math.log(two_word_prob[(test_tokens[0], test_tokens[1])], 2)
            else:
                prob += math.log(one_word_count[test_tokens[1]] / len(tokens), 2)
            for i in range(len(test_tokens) - 2):
                if (test_tokens[i], test_tokens[i + 1], test_tokens[i + 2]) not in three_word_prob:
                    temp = prob
                    try:
                        prob += math.log(one_word_count[test_tokens[i]] / len(tokens), 2)
                        prob += math.log(two_word_prob[(test_tokens[i], test_tokens[i + 1])], 2)
                        prob += math.log(two_word_prob[(test_tokens[i + 1], test_tokens[i + 2])], 2)
                    except:
                        prob = temp
                        prob += math.log(one_word_count[test_tokens[i]] / len(tokens), 2)
                        prob += math.log(one_word_count[test_tokens[i + 1]] / len(tokens), 2)
                        prob += math.log(one_word_count[test_tokens[i + 2]] / len(tokens), 2)
                else:
                    prob += math.log(three_word_prob[(test_tokens[i], test_tokens[i + 1], test_tokens[i + 2])], 2)
            results.append(prob)
            f2.close()
            k += 1
        max_prob = max(results)
        author_name_index = results.index(max_prob)
        author_name = author_list[author_name_index]
        print(author_name)


def main():
    if len(sys.argv) == 2:
        biagram_prob(sys.argv[1])
    elif len(sys.argv) == 4:
        trigram_test(sys.argv[1], sys.argv[3])
    else:
        print("invalid number of arguments")


if __name__ == '__main__':
    main()
