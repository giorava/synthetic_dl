### lambda factor callback
def compute_lambda(p, c, f, lambda_e):
    
    lambda_ep1_prime = (p*f)/(1-f)*c
    lambda_ep1_second = 0.3*lambda_ep1_prime + 0.7*lambda_e
    gamma = lambda_ep1_second/lambda_e

    if gamma>2: 
        return 2*lambda_e
    if gamma<(1/2): 
        return lambda_e/2
    else: 
        return lambda_ep1_second


### early stopping callback
class EarlyStopping():
    def __init__(self, tolerance=15, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def criteria(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True

    def get_early_stop(self):
        return self.early_stop
            