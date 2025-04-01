import jax
import jax.numpy as jnp

class ReplayBuffer_JAX():
    def __init__(self, conf):
        '''
        :input conf :                           (Configuration file)

            :param REPLAY_SIZE :                (int) Max number of transitions to store in the buffer. When the buffer overflows the old memories are dropped
            :param BATCH_SIZE :                 (int) Size of the mini-batch 
            :param nb_state :                   (int) State size (robot state size + 1)
        '''

        self.conf = conf
        self.storage_mat = None
        self.next_idx = 0
        self.full = 0

    def add(self, data):
        ''' Add transitions to the buffer '''
        new_entries = jnp.concatenate(data, axis=1) 
        if self.storage_mat is None:
            self.storage_mat = new_entries
        else:
            self.storage_mat = jnp.concatenate([self.storage_mat, new_entries], axis=0)
        
        if self.storage_mat.shape[0] > self.conf.REPLAY_SIZE:
            self.storage_mat = self.storage_mat[-self.conf.REPLAY_SIZE:]
                
    def sample(self, key, batch_size=None, n_sample=1):
        ''' Sample a batch of transitions '''
        if batch_size is None:
            batch_size = self.conf.BATCH_SIZE

        num_samples = self.storage_mat.shape[0]
        indices = jax.random.choice(key, num_samples, shape=(n_sample,batch_size), replace=True)
        
        return self.storage_mat[indices,:self.conf.nb_state], self.storage_mat[indices,self.conf.nb_state:self.conf.nb_state+1], self.storage_mat[indices,self.conf.nb_state+1:self.conf.nb_state*2+1], self.storage_mat[indices,self.conf.nb_state*2+1:self.conf.nb_state*3+1], self.storage_mat[indices,self.conf.nb_state*3+1:self.conf.nb_state*3+2], jnp.ones((n_sample,batch_size,1)), None