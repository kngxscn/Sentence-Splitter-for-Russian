#!/usr/bin/python3
import collections
import re

import numpy as np
import pandas as pd
from keras.models import model_from_json
from functools import reduce


class Splitter:
    def __init__(self):
        self.model = model_from_json(open('./Models/keras_model.json').read())
        self.model.load_weights('./Models/keras_weights.h5')

        self.model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        self.radius = int(self.model.get_config()['layers'][0]['config']['batch_input_shape'][1] / 2)
        self.length = 2 * self.radius
        self.padding = []
        for i in range(int(self.radius)):
            self.padding.append('^')

        header = ['Char']
        for i in range(64):
            header.append('X' + str(i+1))

        embeddings = pd.read_csv('./Models/char_embeddings.csv', names=header)
        embeddings_dictionary = {}
        
        for i in range(len(embeddings)):
            vec = []
            for j in range (64):
                vec += [embeddings['X' + str(j+1)][i]]
            embeddings_dictionary[embeddings['Char'][i]] = vec
        embeddings_dictionary[' '] = embeddings_dictionary['_']
        embeddings_dictionary['\n '] = embeddings_dictionary['.']

        class Embeddings_Reader(dict):
            def __missing__(self, key):
                return embeddings_dictionary[u'UNK']

        self.embeddings_lookup = Embeddings_Reader(embeddings_dictionary)

    @staticmethod
    def formatting (res):
        res = re.sub(u'\xa0', u' ', res)
        res = re.sub(u'[\n \r\t]*[\n\r][\n \r\t]*', u'\n', res)
        res = re.sub(u'[ \t]+', u' ', res)
        res = re.sub(u'\n+', u'\n', res)
        return res

    @staticmethod
    def stop_split(text):
        dot_list = list(map(lambda x: x + '.', text.split('.')))
        if dot_list[-1] == '.':
            dot_list = dot_list[:-1]
        excl_list = reduce(lambda x, y: x + list(map(lambda z: z + '!', y.split('!')[:-1])) + [y.split('!')[-1]], dot_list,
                           [])
        quest_list = reduce(lambda x, y: x + list(map(lambda z: z + '?', y.split('?')[:-1])) + [y.split('?')[-1]], excl_list,
                            [])
        return quest_list

    @staticmethod
    def add_newline(l, stop_type):
        if stop_type == 0:
            return l + '\n'
        else:
            return l

    @staticmethod
    def remove_spaces (sent):
        sent = re.sub(u'^ +', u'', sent)
        return sent

    def split(self, input_text):
        input_text = input_text

        dw_input_text = self.formatting(input_text)
        dw_input_list = self.stop_split(dw_input_text)
        padded_list = self.padding + dw_input_list + self.padding
        embedded_vectors = collections.deque([])

        for linenum in range(self.radius, len(padded_list) - self.radius):
            left = ''.join(padded_list[linenum - self.radius: linenum + 1])[:-1]
            right = ''.join(padded_list[linenum + 1: linenum + self.radius + 1])
            features = list(left)[-self.radius:] + list(right)[:self.radius]
            feature_vector = list(map(lambda x: self.embeddings_lookup[x], features))
            embedded_vectors.append(feature_vector)

        X_text = np.array(embedded_vectors, dtype='float32')
        predictions = self.model.predict(X_text)
        classes = list(map (lambda x: 0 if x < 0.5 else 1, predictions))

        final_lines = list(map(self.add_newline, dw_input_list, classes))
        final_text = reduce(lambda x, y: x + y, final_lines)
        final_text = '\n'.join(list(map(self.remove_spaces, final_text.splitlines())))
        split_text = final_text.split('\n')


        return split_text
