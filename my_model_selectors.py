import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        selected_model = None
        likelihood_score = float('inf')

        for n in range(self.min_n_components, self.max_n_components + 1):
            # iterate over the states

            try:
                hmm_model = self.base_model(num_states=n)  # obtain and assign model for given state
                LogL = hmm_model.score(self.X, self.lengths)
                # using hmmlearn calculate the log probability for sample
                # X - matrix of word samples, lengths - length of individual sample
                # calculating likelihood of given sample
                LogN = np.log(sum(self.lengths))
                # using np calculate the log probability for sum of the sizes of the samples

                p = np.power(n, 2) + 2 * len(self.sequences) * n - 1
                # calculates K - number of free parameters

                if hmm_model is not None:  # making sure the model is not empty
                    bic = -2 * LogL + p * LogN  # input the variables into the BIC formula
                    if bic < likelihood_score:  # check if calculated BIC is lower than the current estimate
                        likelihood_score = bic
                        selected_model = hmm_model

            except:
                if self.verbose:
                    print("BIC failure on {} with {} states".format(self.this_word, n))

            return selected_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        selected_model = None
        likelihood_score = float('inf')

        for n in range(self.min_n_components, self.max_n_components + 1):
            # iterate over the states

            try:
                hmm_model = self.base_model(num_states=n) # obtain and assign model for given state
                LogL = hmm_model.score(self.X, self.lengths)

                other_word_sum = 0.0  # sum of Likelihood score for other word
                for word in self.words:  # iterate through each word in sequence
                    if word != self.this_word:  # check that word is not the current word
                        oth_word, oth_lengths = self.hwords[word]  # obtain the length sequence for the word
                        other_word_sum += hmm_model.score(oth_word, oth_lengths)
                        # calculate the sum of the likelihood of all words that are not this_word

                    other_word_mean = other_word_sum / (len(self.words) - 1)
                    # calculate mean likelihood of all the words except i - the anti-likelihood
                    dic = LogL - other_word_mean # input the different likelihoods into calculating the DIC
                    if dic < likelihood_score:  # check if calculated DIC is lower than the current estimate
                        likelihood_score = dic
                        selected_model = hmm_model
            except:
                if self.verbose:
                    print("BIC failure on {} with {} states".format(self.this_word, n))

            return selected_model

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        best_score = float('-inf')
        selected_model = None
        scores = []

        for n in range(self.min_n_components, self.max_n_components + 1):
            n_splits_check = min(3, len(self.sequences))  # handles words that don't have the default 3
            hmm_model = self.base_model(n)

            split_method = KFold(random_state=self.random_state, n_splits=n_splits_check) # utilize KFold to split data
            for train_id, test_id in split_method.split(self.sequences):#iterate the partitions of the sequences
                # setup the training dataset
                train_X,train_Xlengths = combine_sequences(train_id, self.sequences)
                # steup the testing dataset
                test_X, test_Xlengths = combine_sequences(test_id, self.sequences)
                try:
                    hmm_model.fit(train_X, train_Xlengths) # update model with training dataset
                    likelihood_score = hmm_model.score(test_X, test_Xlengths) # calculate likilihood score for test dataset
                    scores.append(likelihood_score)
                except:
                    pass
            mean = np.mean(scores)  # calculate the mean of list
            if mean > best_score:  # check if mean i higher than current best_score
                best_score = mean
                selected_model = hmm_model

        return selected_model



