import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    # return probabilities, guesses
    all_Xlengths = test_set.get_all_Xlengths()
    for test_word in all_Xlengths.keys():
        X, lengths = all_Xlengths[test_word]
        test_word_prob = {}
        logL = float('-inf')
        for word, model in models.items():
            try:
                test_word_prob[word] = model.score(X, lengths)
                if test_word_prob[word] > logL:
                    guess_word = word
                    logL = test_word_prob[word]
            except:
                test_word_prob[word] = float('-inf')
        probabilities.append(test_word_prob)
        guesses.append(guess_word)
    return probabilities, guesses

