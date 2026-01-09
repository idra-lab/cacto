import jax
import functools
import jax.numpy as jnp
from flax import struct

@struct.dataclass
class ReplayBufferJAX:
    """
    Replay buffer class.

    Attributes:
        storage :   (jnp.darray) Main memory buffer storing all transitions or samples.
        ptr :       (jnp.int) Index of the next insertion (circular buffer behavior).
        size :      (jnp.int) Number of filled slots in the buffer.
        max_size :  (int) Maximum capacity of the buffer.
        asup :      (bool) Whether this buffer is for supervised learning (True) or RL transitions (False).
        nb_state :  (int) Dimensionality of the state vector.
        nb_action : (int) Dimensionality of the action vector.
    """
    storage: jnp.ndarray
    ptr: jnp.int32
    size: jnp.int32
    max_size: int
    nb_state: int = struct.field(pytree_node=False)
    nb_action: int = struct.field(pytree_node=False)
    
def init_buffer(conf) -> ReplayBufferJAX:
    """Initialize a ReplayBufferJAX instance."""
    total_dim = 3 * conf.nb_state + 2 # (x, partial/total_cost_to_go, x_next, V_x, done_flag)

    storage = jnp.zeros((conf.REPLAY_SIZE, total_dim), dtype=jnp.float32)

    return ReplayBufferJAX(
        storage=storage,
        ptr=jnp.int32(0),
        size=jnp.int32(0),
        max_size=conf.REPLAY_SIZE,
        nb_state=conf.nb_state,
        nb_action=conf.nb_action,
    )

def buffer_add(buf: ReplayBufferJAX, data: tuple) -> ReplayBufferJAX:
    """Add new entries to the replay buffer in a circular fashion."""
    new_entries = jnp.concatenate(data, axis=1) 
    n = new_entries.shape[0]

    # Compute insertion indices (circular buffer)
    idxs = (jnp.arange(n) + buf.ptr) % buf.max_size

    # Insert new data
    storage = buf.storage.at[idxs].set(new_entries)

    # Update pointer and buffer size
    ptr = (buf.ptr + n) % buf.max_size               
    size = jnp.minimum(buf.size + n, buf.max_size) 

    return buf.replace(storage=storage, ptr=ptr, size=size)

@functools.partial(jax.jit, static_argnames=("batch_size", "n_sample"))
def buffer_sample(buf: ReplayBufferJAX, key, batch_size: int = None, n_sample: int = 1):
    """Randomly sample transitions from the buffer for RL training."""
    if batch_size is None:
        raise ValueError("batch_size must be provided")
    
    num_samples = buf.size
    s = buf.storage
    ns = buf.nb_state

    # Sample random indices
    idxs = jax.random.randint(key, (n_sample, batch_size), 0, num_samples)

    return (
        s[idxs, :ns],                         # x
        s[idxs, ns:ns+1],                     # partial/total cost to go
        s[idxs, ns+1:ns*2+1],                 # x_next
        s[idxs, ns*2+1:ns*3+1],               # V_x
        s[idxs, ns*3+1:ns*3+2],               # Done flag
        jnp.ones((n_sample, batch_size, 1)),  # Sample weights (currently not used)
    )

