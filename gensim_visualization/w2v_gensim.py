# Python program to generate word vectors using Word2Vec

#      IT WORKS !!! :)

import gensim
from gensim.models import Word2Vec

egypt = True
train = True
show = True
letters = False

create_model = 1
load_model = 0

file_name = "../data/hier_sep.txt"
model_name = "word2vecEGYPT.model"

if train:
    print("will train model")
    sample = open(file_name, "r", encoding="utf-8")
    s = sample.read()
    f = s

    data = []

    # iterate through each sentence in the file
    if egypt:
        for i in s.split('\n'):
            temp = []

            # tokenize the sentence into words
            for j in i.split('_'):
                if j != ' ':
                    temp.append(j)

            data.append(temp)
    elif letters:
        for i in s.split(' '):
            i = i.replace("\n", ' ')
            i = i.replace("?", "")
            i = i.replace(",", "")
            data.append(i)
    else:
        for i in s.split('.'):
            temp = []
            i = i.replace("\n", ' ')
            i = i.replace("?", "")
            i = i.replace(",", "")
            i = i.replace('\t', '\t ')

            # tokenize the sentence into words
            for j in i.split(' '):
                temp.append(j)

            data.append(temp)

    # Create CBOW model
    if create_model:
        print("creating model")
        model1 = Word2Vec(data, min_count=1,
                            vector_size=100, window=5, sg = 0)  # window = window around words
    elif load_model:
        print("loading model")
        model1 = Word2Vec.load(file_name)
    else:
        model1 = 0
        Exception("no not really")

    print(model1.train(data, total_examples=len(model1.wv), epochs=40))
    word_vectors = model1.wv

    model1.save(model_name)
    print("model saved")


if show:
    from sklearn.decomposition import PCA, TruncatedSVD
    import matplotlib.pyplot as plt
    import numpy as np

    model1 = Word2Vec.load(model_name)

    # print(model1.wv.key_to_index)  # prints the slovnicek

    # NORMALIZATION
    X = model1.wv[model1.wv.key_to_index]
    Y = np.zeros(X.shape)
    seznam = list(model1.wv.key_to_index.keys())

    for i in range (len(model1.wv.key_to_index)):
        Y[i] = model1.wv.get_vector(seznam[i], norm=True)
    #X = Y

    #X = model1.wv[model1.wv.key_to_index]
    pca = PCA(n_components=2)
    # pca = TruncatedSVD(n_components=2)
    result = pca.fit_transform(X)
    fig = plt.figure(figsize=(20,10))

    num_v = len(model1.wv)

    colors = np.arange(4572)
    plt.scatter(result[:num_v, 0], result[:num_v, 1], c=colors, cmap='viridis')
    words = list(model1.wv.key_to_index)[:num_v]
    count = 0
    for i, word in enumerate(words):
        # plt.annotate(word, xy=(result[i, 0], result[i, 1]), fontsize=15)
        count += 1
    assert count == len(model1.wv)
    plt.show()

    # model1.wv.similarity('orange', 'pear')