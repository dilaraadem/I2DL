"""Linear Softmax Classifier."""
# pylint: disable=invalid-name
import numpy as np

from .linear_classifier import LinearClassifier


def cross_entropy_loss_naive(W, X, y, reg):
    """
    Cross-entropy loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # pylint: disable=too-many-locals
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient using explicit     #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################
    epsilon = 1e-10
    N = X.shape[0]
    C = W.shape[1]
    D = X.shape[1]
        
    #run the loop for all samples in dataset.
    for i in range(N):
        f = X[i].dot(W) #take dot product, f is scores
        f -= np.max(f) #for numerical stability, make the max value 0
        
        p = np.exp(f)/np.sum(np.exp(f)) #probability score estimates
                
        loss += -np.log(p[y[i]] + epsilon) #calculate loss for each data point
        
        for j in range(D): #calculate gradient descent.
            for k in range(C):
                if k==y[i]: #if the true class is equal to k
                    dW[j, k] += X.T[j, i] * (p[k]-1)
                else:
                    dW[j, k] += X.T[j, i] * p[k]
        
    loss /= float(N) # get average of loss
    loss += reg * np.sum(W*W) # do regularization
        
    dW /= float(N) # get average of dW
    dW += 2* reg * W # do regularization
        
    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


def cross_entropy_loss_vectorized(W, X, y, reg):
    """
    Cross-entropy loss function, vectorized version.

    Inputs and outputs are the same as in cross_entropy_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient without explicit   #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################
    
    epsilon = 1e-10
    
    N = X.shape[0]
    
    f = np.dot(X, W) #scores (N,C)
    f -= np.max(f, axis=1, keepdims=True) #for numerical stability
    expo_probs = np.exp(f)
    sum_probs = np.sum(np.exp(f), axis=1, keepdims=True)
    p = expo_probs / sum_probs #calculate probability estimates
    
    log_probs = -np.log(p[range(N), y] + epsilon)
    loss = np.sum(log_probs) #calculate loss
    
    loss /= float(N) # get average of loss
    loss += reg * np.sum(W*W) # do regularization
    
    i = np.zeros_like(p)
    i[np.arange(N), y] = 1
    dW = X.T.dot(p - i)
    
    dW /= float(N) # get average of dW
    dW += 2 * reg * W # do regularization    

    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


class SoftmaxClassifier(LinearClassifier):
    """The softmax classifier which uses the cross-entropy loss."""

    def loss(self, X_batch, y_batch, reg):
        return cross_entropy_loss_vectorized(self.W, X_batch, y_batch, reg)


def softmax_hyperparameter_tuning(X_train, y_train, X_val, y_val):
    # results is dictionary mapping tuples of the form
    # (learning_rate, regularization_strength) to tuples of the form
    # (training_accuracy, validation_accuracy). The accuracy is simply the
    # fraction of data points that are correctly classified.
    results = {}
    best_val = -1
    best_softmax = None
    all_classifiers = []
    #learning_rates = [0.5e-7, 1e-7, 1.5e-7, 2e-7, 2.5e-7, 3e-7]
    #regularization_strengths = [1e2, 1.5e2, 2e2, 0.5e3, 1e3, 1.5e3, 2e3, 1e4, 2.5e4]
    
    learning_rates = [1e-15, 1e-10, 5e-9, 5e-7, 2e-6, 5e-8, 1e-7]
    regularization_strengths = [0.5e2, 1e2, 0.5e3, 1e3]


    ############################################################################
    # TODO:                                                                    #
    # Write code that chooses the best hyperparameters by tuning on the        #
    # validation set. For each combination of hyperparameters, train a         #
    # classifier on the training set, compute its accuracy on the training and #
    # validation sets, and  store these numbers in the results dictionary.     #
    # In addition, store the best validation accuracy in best_val and the      #
    # Softmax object that achieves this accuracy in best_softmax.              #
    #                                                                          #
    # Hint: You should use a small value for num_iters as you develop your     #
    # validation code so that the classifiers don't take much time to train;   # 
    # once you are confident that your validation code works, you should rerun #
    # the validation code with a larger value for num_iters.                   #
    ############################################################################
    num_it = 1000
    softm = SoftmaxClassifier() #init
    
    for rg in regularization_strengths:
        for lr in learning_rates:
            
            softm.train(X_train, y_train, learning_rate=lr, reg=rg, num_iters=num_it) #train
            
            y_t_pred = softm.predict(X_train) #predict train labels
            train_acc = np.mean(y_t_pred == y_train) #get accuracy for train labels
            
            y_val_pred = softm.predict(X_val) #predict validation
            val_acc = np.mean(y_val_pred == y_val)
            
            results [(lr, rg)] = (train_acc, val_acc) # store accuracies
            if best_val < val_acc: # get the best accuracy
                best_val = val_acc
                best_softmax = softm
                all_classifiers.append(best_softmax)

    ############################################################################
    #                              END OF YOUR CODE                            #
    ############################################################################
        
    # Print out results.
    for (lr, reg) in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg)]
        print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
              lr, reg, train_accuracy, val_accuracy))
        
    print('best validation accuracy achieved during validation: %f' % best_val)

    return best_softmax, results, all_classifiers
