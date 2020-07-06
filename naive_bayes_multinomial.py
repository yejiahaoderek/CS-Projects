'''naive_bayes_multinomial.py
Naive Bayes classifier with Multinomial likelihood for discrete features
Jiahao (Derek) Ye
CS 251 Data Analysis Visualization, Spring 2020
'''
import numpy as np


class NaiveBayes:
    '''Naive Bayes classifier using Multinomial likeilihoods (discrete data belonging to any
     number of classes)'''
    def __init__(self, num_classes):
        '''Naive Bayes constructor sets the number of classes (int: num_classes) this classifier
        will be trained to detect. All other fields initialized to None.'''
        
        self.num_classes = num_classes

        # class_priors: ndarray. shape=(num_classes,).
        #   Probability that a training example belongs to each of the classes
        #   For spam filter: prob training example is spam or ham
        self.class_priors = None
        
        # class_likelihoods: ndarray. shape=(num_classes, num_features).
        #   Probability that each word appears within class c
        self.class_likelihoods = None

    def train(self, data, y):
        '''Train the Naive Bayes classifier so that it records the "statistics" of the training set:
        class priors (i.e. how likely an email is in the training set to be spam or ham?) and the
        class likelihoods (the probability of a word appearing in each class — spam or ham)

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.
        y: ndarray. shape=(num_samps,). Corresponding class of each data sample.

        TODO:
        - Compute the instance variables self.class_priors and self.class_likelihoods needed for
        Bayes Rule. See equations in notebook.
        '''

        # num_samples
        N = data.shape[0]
        # num_features/ words
        M = data.shape[1]

        # initialize self.class_priors
        self.class_priors = []
        # compute class priors
        for i in range(self.num_classes):
            # i = class
            Nc = np.array(np.where(y == i)).shape[1]
            prior = Nc/N
            self.class_priors.append(prior)
        self.class_priors = np.array(self.class_priors)

        # initialize self.class_likelihoods
        self.class_likelihoods = np.zeros((self.num_classes, M))
        # compute class likelihood
        for i in range(self.num_classes):
            # i = class = c
            # idx of samps belong to that class
            idx = np.where(y == i)
            # Ncw   - total count of the word in class c
            Ncw = np.sum(data[idx], axis = 0)
            # Nc    - # of all words belong to class c
            Nc = np.sum(data[idx])
            llh = (Ncw + 1)/(Nc + M)
            self.class_likelihoods[i] = llh
        
        pass

    def predict(self, data):
        '''Combine the class likelihoods and priors to compute the posterior distribution. The
        predicted class for a test sample from `data` is the class that yields the highest posterior
        probability.

        Parameters:
        -----------
        data: ndarray. shape=(num_test_samps, num_features). Data to predict the class of
            Need not be the data used to train the network

        Returns:
        -----------
        ndarray of nonnegative ints. shape=(num_samps,). Predicted class of each test data sample.

        TODO:
        - Process test samples one-by-one.
        - Look up the likelihood (from training) ONLY AT the words that appear > 0 times in the
        current test sample.
        - Take the log and sum these likelihoods together.
        - Solve for posterior for each test sample i (see notebook for equation).
        - Predict the class of each test sample according to the class that produces the largest
        posterior probability.
        '''

        N = data.shape[0]
        M = data.shape[1]
        pred_class = []

        for i in range(N):
            pos_idices = np.array(np.where(data[i] > 0))
            c = np.zeros((self.num_classes,1))
            # store prob of class j on row j of c
            for j in range(self.num_classes):
                # compute log_posteriors for class j
                prob = np.log(self.class_priors[j]) + np.sum(np.log(self.class_likelihoods[j, pos_idices]))    
                c[j] = prob
            # pick the class that has the max_prob
            final_c = int(np.where(np.max(c) == c)[0])
            # add to the final_c
            pred_class.append(final_c)

        return np.array(pred_class)

        pass

    def accuracy(self, y, y_pred):
        '''Computes accuracy based on percent correct: Proportion of predicted class labels `y_pred`
        that match the true values `y`.

        Parameters:
        -----------
        y: ndarray. shape=(num_data_sams,)
            Ground-truth, known class labels for each data sample
        y_pred: ndarray. shape=(num_data_sams,)
            Predicted class labels by the model for each data sample

        Returns:
        -----------
        float. Between 0 and 1. Proportion correct classification.

        NOTE: Can be done without any loops
        '''

        N = y.shape[0]
        num_correct = np.count_nonzero(y-y_pred)
        
        return 1 - num_correct / N


        pass

    def confusion_matrix(self, y, y_pred):
        '''Create a confusion matrix based on the ground truth class labels (`y`) and those predicted
        by the classifier (`y_pred`).

        Parameters:
        -----------
        y: ndarray. shape=(num_data_samps,)
            Ground-truth, known class labels for each data sample
        y_pred: ndarray. shape=(num_data_samps,)
            Predicted class labels by the model for each data sample

        Returns:
        -----------
        ndarray. shape=(num_classes, num_classes).
            Confusion matrix
        '''

        confusion_mat = np.zeros((2,2))
        comp = np.equal(y, y_pred)
        correct_preds = y_pred[np.where(comp == True)]

        confusion_mat[0,0] = correct_preds.shape[0] - np.count_nonzero(correct_preds)
        confusion_mat[0,1] = np.count_nonzero(y_pred) - np.count_nonzero(correct_preds)
        confusion_mat[1,0] = y_pred.shape[0] - np.count_nonzero(y_pred) - confusion_mat[0,0]
        confusion_mat[1,1] = np.count_nonzero(correct_preds)

        return confusion_mat


        pass
