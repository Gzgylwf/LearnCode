import tensorflow as tf
import numpy as np
import re
import json
import io
import pre_process as pp

def main():
    pass

if __name__ == '__main__':
    x, y = pp.load_data_and_labels("data/rt-polarity.pos", "data/rt-polarity.neg")
    vocabs = pp.get_vocab(x)
    x = pp.padding(x, max_length=59)

    np.random.shuffle(x)

    #for batch in pp.batch_iter(x, 100, 4):
        #print(batch)

    # Save vocab
    '''
    with io.open("data/all.vocab", "w", encoding="utf-8") as f:
        print("Total {} vocabularies...".format(len(vocabs)))
        for word in vocabs.keys():
            f.write("{} {} {}\n".format(word, vocabs[word][0], vocabs[word][1]))
        f.close()
    '''
    