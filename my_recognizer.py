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
    for word_id in range(test_set.num_items):  # iterate through the test_set
        X, Xlength = test_set.get_item_Xlengths(word_id)  # get the length of the word
        logL_dict = {} # Likelihood dict
        selected_word = None
        best_score = float('-inf')

        # iterate through the models
        for word, model in models.items():
            try:
                # calculate logL of the given word
                logL_dict[word] = model.score(X, Xlength)

            except:

                logL_dict[word] = float('-inf')

            # handles maximum score for given word
            if logL_dict[word] > best_score:
                best_score = logL_dict[word]
                selected_word = word
        # Update probabilities with the Likelihood score for given word's model
        probabilities.append(logL_dict)

        # Update guesses with the selected model for given word
        guesses.append(selected_word)

    return probabilities, guesses
