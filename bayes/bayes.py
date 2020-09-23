import numpy as np
import math


def loadDataSet():
    post_list = [
        ["my", "dog", "has", "flea", "problems", "help", "please"],
        ["maybe", "not", "take", "him", "to", "dog", "park ", "stupid"],
        ["my", "dalmation", "is", "so", "cute", "I", "love", "him"],
        ["stop", "posting", "stupid", "worthless", "garbage"],
        ["mr", "licks", "ate", "my", "steak", "how", "to", "stop", "him"],
        ["quit", "buying", "worthless", "dog", "food", "stupid"],
    ]
    labels = [0, 1, 0, 1, 0, 1]
    return post_list, labels


def createVocabList(dateSet):
    vocab = set()
    for doc in dateSet:
        vocab = vocab | set(doc)
    return list(vocab)


def word2Vec(vocab_list, input_set):
    ret = [0] * len(vocab_list)
    for w in input_set:
        if w in vocab_list:
            ret[vocab_list.index(w)] = 1
        else:
            print("the word:%s is OOV" % w)
    return ret


def trainNB0(train_x, train_y):
    docs_num = len(train_x)
    words_num = len(train_x[0])
    pAbusive = sum(train_y) / float(len(train_y))
    p0num, p1num = np.ones(words_num), np.ones(words_num)  # 改为noe 防止log 0
    p0denom, p1denom = 2.0, 2.0
    for i in range(docs_num):
        if train_y[i] == 1:
            p1num += train_x[i]
            p1denom += np.sum(train_x[i])
        else:
            p0num += train_x[i]
            p0denom += np.sum(train_x[i])

    p1vect = p1num / p1denom
    p0vect = p0num / p0denom
    return np.log(p0vect), np.log(p1vect), pAbusive


def classifyNB(input, p0Vec, p1Vec, pClass1):
    p1 = np.sum(input * p1Vec) + math.log(pClass1)
    p0 = np.sum(input * p0Vec) + math.log(1 - pClass1)
    return 1 if p1 > p0 else 0


def testingNB():
    posts, labels = loadDataSet()
    vocab_list = createVocabList(posts)
    trains = []
    for doc in posts:
        trains.append(word2Vec(vocab_list, doc))
    p0v, p1v, pAb = trainNB0(np.array(trains), np.array(labels))
    test_ents = [["love", "my", "dalmation"], ["stupid", "garbage"]]

    for test_ent in test_ents:
        test_doc = np.array(word2Vec(vocab_list, test_ent))
        print(test_ent, "classified as :", classifyNB(test_doc, p0v, p1v, pAb))


def word2bag(vacab_list, input_set):
    vet = [0] * len(vacab_list)
    for w in input_set:
        if w in vacab_list:
            vet[vacab_list.inde(w)] += 1
        else:
            print("this is a OOV :%s" % w)
    return vet


def text_parse(input_str):
    import re

    tokens = re.split(r"\W*", input_str)
    return [token.lower() for token in tokens if len(token) > 2]


def spam_test():
    docs, labels, full_text = [], [], []
    for i in range(1, 26):
        with open("email/spam/%d.txt" % i) as f:
            input_str = f.read()
        doc = text_parse(input_str)
        full_text.extend(doc)
        docs.append(doc)
        labels.append(1)
        with open("email/ham/%d.txt" % i) as f:
            input_str = f.read()
        doc = text_parse(input_str)
        full_text.extend(doc)
        docs.append(doc)
        labels.append(0)
    vocab_list = createVocabList(docs)
    training_set = range(50)
    test_set = []
    for i in range(10):
        rand_inx = int(np.random.uniform(0, len(training_set)))
        test_set.append(training_set[rand_inx])
        del training_set[rand_inx]
    x_train, y_train = [], []
    for inx in training_set:
        x_train.append(word2Vec(vocab_list, docs[inx]))
        y_train.append(labels[inx])
    p0v, p1v, pSpam = trainNB0(np.array(x_train), np.array(y_train))
    errors = 0
    for inx in test_set:
        test_vec = word2Vec(vocab_list, docs[inx])
        if classifyNB(np.array(test_vec), p0v, p1v, pSpam) != labels[inx]:
            errors += 1
    print("the error rate is :", float(errors) / len(test_set))


if __name__ == "__main__":
    print(testingNB())
