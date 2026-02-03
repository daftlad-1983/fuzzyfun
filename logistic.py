
import numpy as np
from typing import Union

class LogisticReg:
    '''
    A class to perform a logistic regression using gradient descent

    Atributes
    ---------

    weights : np.array
        the model weights
    loss_tracker : list
        a list to track the loss during training
    epoc : list
        a list to track the epochs during training
    l2_mask : np.array
        an array to mask off the intercept weight if l2 reg is being used
    training_mean : np.array
        an array to keep the means of the variables used in training. This is used for normalization
        for training then prediction and also returning q values to monitor drift
    training_std : np.array
        an array to keep the std of the variables used in training. This is used for normalization
        for training then prediction and also returning q values to monitor drift

    Methods
    -------

    fit(self, x: np.array, y: np.array, lr: float =0.1, epochs: int =1000, l2_reg: float = 0) -> dict[str,list]:
        fits the model. Returns the loss and epochs for plotting

    predict(self, x: np.array, return_q: bool = False) -> np.array:
        predicts the class from the x array. Returns the probabilities and also the q_values for monitoring model
        drift.

    '''

    def __init__(self):

        self.weights = None
        self.loss_tracker = None
        self.epoc = None
        self.l2_mask = None
        self.training_mean = None
        self.training_std = None

    def get_sig(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        '''Calculates the logistic function'''
        y = 1 / (1 + np.exp(-1 * np.matmul(x, w)))
        return y

    def calc_loss(self, x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        '''Calculates the loss'''
        sigfn = self.get_sig(x, w)
        return np.mean(-y * np.log(sigfn) - (1 - y) * np.log(1 - sigfn))

    def get_grad(self, x: np.ndarray, y: np.ndarray, l2_reg: float = 0) -> np.ndarray:
        '''
        Calculates  the gradients
        l2_reg is used for l2 regularization. Uses  l2_reg*  0.5 * weights.T @ weights
        which differentiates to l2_reg*weights.
        the l2_mask masks off the intercept so this is not penalized
        '''

        gvec = 1 / x.shape[0] * (
                    np.matmul(x.T, (self.get_sig(x, self.weights) - y)) + l2_reg * self.weights * self.l2_mask)
        return gvec

    def fit(self, x: np.ndarray, y: np.ndarray, lr: float = 0.1, epochs: int = 1000, l2_reg: float = 0) -> dict[str, list]:
        '''
        Info:
            Takes the X array and y array (targets) and fits a logistic regression. This can then be used with
            the predict method
        Parameters:
            x: np.array
            The np array of the explanatory variables. Must be shape [samples, features]
            y: np.array
            The np array of the targets. These must be in {0,1}. Must have shape [samples, 1]
            lr: float
            The learning rate for the gradient decent
            epochs: int
            Number of epochs to go through
            l2_reg: float
            The l2 penalty to put on the weights
        Returns:
            Dictionary of 'loss': list and 'epochs': list
            This is for monitoring the loss over the epochs during training
        Effects:
            Updates self.weights for use in predict
            Updates self.training_mean for use in predict
            Updates self.training_std for use in predict
        '''

        assert x.shape[0] == y.shape[0], f"x and y must be the same shape. Recieved x: {x.shape}, y: {y.shape}"

        self.training_mean = np.mean(x, axis=0)
        self.training_std = np.std(x, axis=0)

        x = (x - self.training_mean) / self.training_std  # Normalizes the data

        x = np.column_stack((x, np.ones(x.shape[0])))  # Adds on a 1 for the intercept

        x_rows = x.shape[0]
        x_cols = x.shape[1]

        self.loss_tracker = []
        self.epoc = []
        self.weights = np.random.uniform(0, 1, x_cols).reshape(x_cols, 1)  # Initialize weights  randomly
        l2_mask = np.ones_like(self.weights)
        l2_mask[-1] = 0
        self.l2_mask = l2_mask  # A mask for l2 reg which stops the intercept being penalized

        for i in range(epochs):
            self.weights += -lr * self.get_grad(x, y, l2_reg)

            loss = self.calc_loss(x, y, self.weights)
            self.loss_tracker.append(loss)
            self.epoc.append(i)

        return {'loss': self.loss_tracker, 'epochs': self.epoc}

    def predict(self, x: np.ndarray, return_q: bool = False,
                binary: bool = True) -> Union[dict[str, np.ndarray], np.ndarray]:
        '''
        Info:
        Makes predictions given the x input features and returns prediction probabilities
        Parameters:
            x: np.array
            The np array of the input parameters
            return_q: bool
            whether to return the qvalues for use in model tracking
            binary: bool
            whether to return binary predictions (True) or probabilities (False)
        Returns:
            dict:
            dictionary of 'predictions': np.array, the probability  of class 1 and
            'q_values': Union[np.array, None], the q values for the predicted data

        Effects:
            None
        '''

        assert self.weights is not None, "No weights found. Please fit a model first"
        assert x.shape[-1] == self.training_mean.shape[-1], f"x must have {self.training_mean.shape[-1]} variables"

        q_values = None

        if return_q:
            q_values = (x - self.training_mean) / self.training_std

        x = (x - self.training_mean) / self.training_std  # Normalizes the data in the same way as training

        x = np.column_stack((x, np.ones(x.shape[0])))  # Adds ones for the intercept

        if binary:
            return (self.get_sig(x, self.weights) >= 0.5).astype(int)
        return {'predictions': self.get_sig(x, self.weights), 'q_values': q_values}

    def dump(self) -> dict:
        ''' Dumps details for re-initialization'''
        details = {
            'weights': self.weights.tolist(),
            'loss_tracker': self.loss_tracker,
            'epoc': self.epoc,
            'l2_mask': self.l2_mask.tolist(),
            'training_mean': self.training_mean.tolist(),
            'training_std': self.training_std.tolist(),
        }
        return details

    def load(self, weights, loss_tracker, epoc, l2_mask, training_mean, training_std):
        '''loads a model based on the provided metrics from **self.dump()'''
        self.weights = np.array(weights)
        self.loss_tracker = loss_tracker
        self.epoc = epoc
        self.l2_mask = np.array(l2_mask)
        self.training_mean = np.array(training_mean)
        self.training_std = np.array(training_std)

