from __future__ import print_function
import json
import numpy as np


class HMM:

    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
        - pi: (1*num_state) A numpy array of initial probabilities. pi[i] = P(Z_1 = s_i)
        - A: (num_state*num_state) A numpy array of transition probabilities. A[i, j] = P(Z_t = s_j|Z_{t-1} = s_i)
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - obs_dict: A dictionary mapping each observation symbol to its index 
        - state_dict: A dictionary mapping each state to its index
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict

    def forward(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - alpha: (num_state*L) A numpy array where alpha[i, t-1] = P(Z_t = s_i, X_{1:t}=x_{1:t})
                 (note that this is alpha[i, t-1] instead of alpha[i, t])
        """
        S = len(self.pi)
        L = len(Osequence)
        O = self.find_item(Osequence)
        alpha = np.zeros([S, L])
        ######################################################
        # TODO: compute and return the forward messages alpha
        ######################################################
        alpha[:, 0] = self.pi * self.B[:, O[0]]
        for t in range(1, L):
            alpha[:, t] = self.B[:, O[t]] * np.dot(self.A.T, alpha[:, t - 1])
        return alpha

    def backward(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - beta: (num_state*L) A numpy array where beta[i, t-1] = P(X_{t+1:T}=x_{t+1:T} | Z_t = s_i)
                    (note that this is beta[i, t-1] instead of beta[i, t])
        """
        S = len(self.pi)
        L = len(Osequence)
        O = self.find_item(Osequence)
        beta = np.zeros([S, L])
        #######################################################
        # TODO: compute and return the backward messages beta
        #######################################################
        beta[:, L - 1] = np.ones(S)
        for t in range(L - 2, -1, -1):
            beta[:, t] = np.dot(self.A * beta[:, t + 1], self.B[:, O[t+1]])
        return beta

    def sequence_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: A float number of P(X_{1:T}=x_{1:T})
        """

        #####################################################
        # TODO: compute and return prob = P(X_{1:T}=x_{1:T})
        #   using the forward/backward messages
        #####################################################
        alpha_beta = np.dot(self.forward(Osequence).T, self.backward(Osequence))
        return np.trace(alpha_beta)/len(Osequence)

    def posterior_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - gamma: (num_state*L) A numpy array where gamma[i, t-1] = P(Z_t = s_i | X_{1:T}=x_{1:T})
		           (note that this is gamma[i, t-1] instead of gamma[i, t])
        """
        ######################################################################
        # TODO: compute and return gamma using the forward/backward messages
        ######################################################################
        gamma = (self.forward(Osequence) * self.backward(Osequence)) / self.sequence_prob(Osequence)
        return gamma

    def likelihood_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*num_state*(L-1)) A numpy array where prob[i, j, t-1] = 
                    P(Z_t = s_i, Z_{t+1} = s_j | X_{1:T}=x_{1:T})
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, S, L - 1])
        #####################################################################
        # TODO: compute and return prob using the forward/backward messages
        #####################################################################
        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)
        O = self.find_item(Osequence)
        for s in range(S):
            for s_hat in range(S):
                for t in range(L - 1):
                    prob[s][s_hat][t] = (alpha[s][t] *
                                         self.A[s][s_hat] *
                                         self.B[s_hat][O[t + 1]] *
                                         beta[s_hat][t + 1]) / self.sequence_prob(Osequence)
        return prob

    def viterbi(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - path: A List of the most likely hidden states (return actual states instead of their indices;
                    you might find the given function self.find_key useful)
        """
        S = len(self.pi)
        L = len(Osequence)
        delta2 = np.zeros([S, L], dtype="int")
        delta = np.zeros([S, L])
        delta[:, 0] = self.pi * self.B[:, self.obs_dict[Osequence[0]]]

        for t in range(1, L):
            xt = self.obs_dict[Osequence[t]]
            for s in range(S):
                max_delta = -1
                argmax_delta = -1
                for s_dash in range(S):
                    delta_temp = self.A[s_dash][s] * delta[s_dash][t - 1]
                    if delta_temp > max_delta:
                        max_delta = delta_temp
                        argmax_delta = s_dash
                delta[s][t] = self.B[s][xt] * max_delta
                delta2[s][t] = argmax_delta
        z_star = []
        z = np.argmax(delta[:, L - 1])
        z_star.append(z)
        for t in range(L - 1, 0, -1):
            z = delta2[z][t]
            z_star.append(z)
        z_star = z_star[::-1]
        path = [0] * len(z_star)

        for state, observation in self.state_dict.items():
            for i in range(len(z_star)):
                if observation == z_star[i]:
                    path[i] = state
        return path
        # path = []
        # S = len(self.pi)
        # L = len(Osequence)
        # delta = np.zeros([S, L])
        # delta_2 = np.zeros([S, L])
        # delta[:, 0] = self.pi * self.B[:, self.obs_dict[Osequence[0]]]
        # for t in range(1, L):
        #     for s in range(S):
        #         max_delta = -1
        #         argmax_delta = -1
        #         for s_dash in range(S):
        #             temp = self.A[s_dash][s] * delta[s_dash][t - 1]
        #             if temp > max_delta:
        #                 max_delta = temp
        #                 argmax_delta = s_dash
        #         delta[s][t] = self.B[s][self.obs_dict[Osequence[t]]] * max_delta
        #         delta_2[s][t] = argmax_delta
        # z_star = []
        # last_z = np.argmax(delta[:, L-1])
        # z_star.append(last_z)
        # for t in range(L-1, 0, -1):
        #     last_z = delta_2[int(last_z)][t]
        #     z_star.append(last_z)
        # z_star = z_star[::-1]
        # for state, idx in self.state_dict.items():
        #     for i in range(len(z_star)):
        #         if idx == z_star[i]:
        #             path.append(state)
        # return path

    # DO NOT MODIFY CODE BELOW
    def find_key(self, obs_dict, idx):
        for item in obs_dict:
            if obs_dict[item] == idx:
                return item

    def find_item(self, Osequence):
        O = []
        for item in Osequence:
            O.append(self.obs_dict[item])
        return O
