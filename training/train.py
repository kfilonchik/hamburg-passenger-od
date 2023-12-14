from hmmlearn import hmm
import pandas as pd
from joblib import dump, load
import numpy as np

class DataTraining():

    def __init__(self, preprocessing, df, observations, lengths, trip):
        self.df = df
        self.observations = observations
        self.lengths = lengths
        self.preprocessing = preprocessing
        self.trip = trip

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

        dump(model, 'hmm_model_4.joblib')

        return model
    
    def fit_multinominal_model_parameters(self):
        trans_matrix = self.trip.groupby('TripID').apply(self.preprocessing.calculate_transition_matrix)
        model = hmm.MultinomialHMM(n_components=self.preprocessing.hidden_states(self.df), n_iter=10, init_params='e')
        model.transmat_ = self.preprocessing.get_pl_columns(trans_matrix)
        model.startprob_ = self.preprocessing.calculate_initial_state(self.trip)
        model.fit(self.observations)

        print("Learned emission probs:")
        print(model.emissionprob_)

        print("Learned transition matrix:")
        print(model.transmat_)
        dump(model, 'hmm_model_trained_multinominal_2.joblib')
        score =  model.score(self.observations)
        print(score)

        return model
    
    def fit_gauss_model_parameters(self):

        trans_matrix = self.trip.groupby('TripID').apply(self.preprocessing.calculate_transition_matrix)
        model = hmm.GaussianHMM(n_components=self.preprocessing.hidden_states(self.df), n_iter=300, init_params='mc')#, covariance_type="spherical")
        print('number of hidden states:', self.preprocessing.hidden_states(self.df))
        #print("Learned emission probs:")
        #print(model.emissionprob_)
        print('start_prob', self.preprocessing.calculate_initial_state(self.trip))
        print('mean', self.preprocessing.calculate_means(self.trip))
        print('covariance', self.preprocessing.calculate_cov_matrix(self.trip) )

        model.startprob_ =  self.preprocessing.calculate_initial_state(self.trip)
        model.transmat_ = self.preprocessing.get_pl_columns(trans_matrix)
       # model.means_ = self.preprocessing.calculate_means(self.trip)
        #model.covars_ = self.preprocessing.calculate_cov_matrix(self.trip)

        #if np.isnan(model.startprob_ ).any():
        #    print('true')
        #    model.startprob_  = np.nan_to_num(model.transmat_)

        #if np.isnan(model.transmat_).any():
          #  print('true')
          #  model.model.transmat_ = np.nan_to_num(model.transmat_)


        model.fit(self.observations)
        print("Learned transition matrix:")
        print(model.transmat_)
        print("Learned covariance matrix")
        print(model.covars_)
        dump(model, 'hmm_model_trained_10.joblib')
        score =  model.score(self.observations)
        print(score)

        return model