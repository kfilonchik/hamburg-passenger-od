from hmmlearn import hmm
import pandas as pd
from joblib import dump, load
import numpy as np

class DataTraining():

    def __init__(self, preprocessing, df, observations, lengths):
        self.df = df
        self.observations = observations
        self.lengths = lengths
        self.preprocessing = preprocessing

    def setup_hmmodel(self):
        model = hmm.MultinomialHMM(n_components=self.preprocessing.hidden_states(self.df), init_params='')
        model.n_features = len(self.observations[0])
        model.startprob_ = self.preprocessing.calculate_initial_state(self.df)
        
        model.transmat_ = self.preprocessing.calculate_transition_matrix(self.df)
        model.emissionprob_ = self.preprocessing.calculate_emission_probabilities(self.df)
        

        # Check if any emission probabilities are zero or NaN
        print("Emission matrix issues:", np.any(model.emissionprob_ == 0), np.isnan(model.emissionprob_).any())
        print("Stat prob issues:", np.any(model.startprob_  == 0), np.isnan(model.startprob_ ).any())
        print("Transition matrix issues:", np.any(model.transmat_ == 0), np.isnan(model.transmat_).any())
    
        if np.isnan(model.startprob_ ).any():
            print('true')
            model.startprob_  = np.nan_to_num(model.startprob_ )
        if np.isnan(model.emissionprob_).any():
            model.emissionprob_ = np.nan_to_num(model.emissionprob_)
        if np.isnan(model.startprob_).any():
            model.model.startprob_ = np.nan_to_num(model.startprob_)


        print('Number of hidden states:', self.preprocessing.hidden_states(self.df))
        print('Started to train with n_features:',  model.n_features)


        model.fit(self.observations, self.lengths)

        dump(model, 'hmm_model.joblib')

        return model