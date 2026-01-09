import math
import random
import numpy as np
import tensorflow as tf
from sb.segment_tree import SumSegmentTree, MinSegmentTree


class ReplayBuffer(object):
    def __init__(self, conf):
        '''
        :input conf :                           (Configuration file)

            :param REPLAY_SIZE :                (int) Max number of transitions to store in the buffer. When the buffer overflows the old memories are dropped
            :param BATCH_SIZE :                 (int) Size of the mini-batch 
            :param nb_state :                   (int) State size (robot state size + 1)
        '''

        self.conf = conf
        self.storage_mat = np.zeros((conf.REPLAY_SIZE, conf.nb_state + 1 + conf.nb_state + conf.nb_state-1 + 1 + 1))
        self.next_idx = 0
        self.full = 0
        self.exp_counter = np.zeros(conf.REPLAY_SIZE)

    def add(self, data):
        ''' Add transitions to the buffer '''
        # obses_t, rewards, obses_t1, dVdxs, dones, terms
        if len(data) + self.next_idx > self.conf.REPLAY_SIZE:
            self.storage_mat[self.next_idx:,:] = data[:self.conf.REPLAY_SIZE-self.next_idx,:]
            self.storage_mat[:self.next_idx+len(data)-self.conf.REPLAY_SIZE,:] = data[self.conf.REPLAY_SIZE-self.next_idx:,:]
            self.full = 1
        else:
            self.storage_mat[self.next_idx:self.next_idx+len(data),:] = data

        self.next_idx = (self.next_idx + len(data)) % self.conf.REPLAY_SIZE

    def sample(self, batch_size=None, n_sample=1):
        ''' Sample a batch of transitions '''
        # Select indexes of the batch elements
        if batch_size is None:
            batch_size = self.conf.BATCH_SIZE

        if self.full:
            max_idx = self.conf.REPLAY_SIZE
        else:
            max_idx = self.next_idx
        
        idxes = np.random.randint(0, max_idx, size=int(batch_size*n_sample))

        obses_t = self.storage_mat[idxes, :self.conf.nb_state]
        rewards = self.storage_mat[idxes, self.conf.nb_state:self.conf.nb_state+1]
        obses_t1 = self.storage_mat[idxes, self.conf.nb_state+1:self.conf.nb_state*2+1]
        dVdxs = self.storage_mat[idxes, self.conf.nb_state*2+1:self.conf.nb_state*3-1+1]
        dones = self.storage_mat[idxes, self.conf.nb_state*3-1+1:self.conf.nb_state*3-1+2]
        terms = self.storage_mat[idxes, self.conf.nb_state*3-1+2:self.conf.nb_state*3-1+3]

        # Priorities not used
        weights = np.ones((int(batch_size*n_sample),1))
        batch_idxes = None

        # Convert the sample in tensor
        obses_t, rewards, obses_t1, dVdxs, dones, weights = self.convert_sample_to_tensor(obses_t, rewards, obses_t1, dVdxs, dones, weights)

        if n_sample != 1:
            obses_t, rewards, obses_t1, dVdxs, dones, terms, weights = self.reshape_tensor(obses_t, rewards, obses_t1, dVdxs, dones, terms, weights, n_sample)

        return obses_t, rewards, obses_t1, dVdxs, dones, terms, weights, batch_idxes
    
    def convert_sample_to_tensor(self, obses_t, rewards, obses_t1, dVdxs, dones, weights):
        ''' Convert batch of transitions into a tensor '''
        obses_t = tf.convert_to_tensor(obses_t, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)                                  
        obses_t1 = tf.convert_to_tensor(obses_t1, dtype=tf.float32)
        dVdxs = tf.convert_to_tensor(dVdxs, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        weights = tf.convert_to_tensor(weights, dtype=tf.float32)
        
        return obses_t, rewards, obses_t1, dVdxs, dones, weights
    
    def reshape_tensor(self, obses_t, rewards, obses_t1, dVdxs, dones, terms, weights, n_sample):
        ''' Reshape tensor '''
        obses_t  = [obses_t[i*self.conf.BATCH_SIZE:(i+1)*self.conf.BATCH_SIZE] for i in range(n_sample)]
        rewards  = [rewards[i*self.conf.BATCH_SIZE:(i+1)*self.conf.BATCH_SIZE] for i in range(n_sample)]
        obses_t1 = [obses_t1[i*self.conf.BATCH_SIZE:(i+1)*self.conf.BATCH_SIZE] for i in range(n_sample)]
        dVdxs    = [dVdxs[i*self.conf.BATCH_SIZE:(i+1)*self.conf.BATCH_SIZE] for i in range(n_sample)]
        dones    = [dones[i*self.conf.BATCH_SIZE:(i+1)*self.conf.BATCH_SIZE] for i in range(n_sample)]
        terms    = [terms[i*self.conf.BATCH_SIZE:(i+1)*self.conf.BATCH_SIZE] for i in range(n_sample)]
        weights  = [weights[i*self.conf.BATCH_SIZE:(i+1)*self.conf.BATCH_SIZE] for i in range(n_sample)]

        return obses_t, rewards, obses_t1, dVdxs, dones, terms, weights