import numpy as np
import time
from operator import itemgetter
import ray
import thinker.util as util

AB_CAN_WRITE, AB_FULL, AB_FINISH = 0, 1, 2


def custom_choice(tran_n, batch_size, p, replace=False):
    if np.any(np.isnan(p)):
        p[np.isnan(p)] = 0.01
        p /= p.sum()

    non_zero_count = np.count_nonzero(p)
    if non_zero_count < batch_size and not replace:
        # Set zero probabilities to 0.01
        zero_indices = np.where(p == 0)[0]
        p[zero_indices] = 0.01
        # Scale the remaining probabilities
        p /= p.sum()

    return np.random.choice(tran_n, batch_size, p=p, replace=replace)


@ray.remote
class ActorBuffer:
    def __init__(self, batch_size=32):
        self.batch_size = batch_size
        self.buffer = []
        self.buffer_state = []
        self.finish = False

    def write(self, data, state):
        # Write data, a named tuple of numpy arrays each with shape (t, b, ...)
        # and state, a tuple of numpy arrays each with shape (*,b, ...)
        self.buffer.append(data)
        self.buffer_state.append(state)

    def available_for_read(self):
        # Return True if the total size of data in the buffer is larger than the batch size
        total_size = sum([data[0].shape[1] for data in self.buffer])
        return total_size >= self.batch_size

    def get_status(self):
        # Return AB_FULL if the number of data inside the buffer is larger than 3 * self.batch_size
        if self.finish:
            return AB_FINISH
        total_size = sum([data[0].shape[1] for data in self.buffer])
        if total_size >= 2 * self.batch_size:
            return AB_FULL
        else:
            return AB_CAN_WRITE

    def read(self):
        # Return a named tuple of numpy arrays, each with shape (t, batch_size, ...)
        # from the first n=batch_size data in the buffer and remove it from the buffer

        if not self.available_for_read():
            return None

        collected_data = []
        collected_state = []
        collected_size = 0
        n_tuple = type(self.buffer[0])
        while collected_size < self.batch_size and self.buffer:
            collected_data.append(self.buffer.pop(0))
            collected_state.append(self.buffer_state.pop(0))
            collected_size += collected_data[-1][0].shape[1]

        # Concatenate the named tuple of numpy arrays along the batch dimension
        output = n_tuple(
            *(
                np.concatenate([data[i] for data in collected_data], axis=1)
                if collected_data[0][i] is not None
                else None
                for i in range(len(collected_data[0]))
            )
        )
        output_state = tuple(
            np.concatenate([state[i] for state in collected_state], axis=1)
            for i in range(len(collected_state[0]))
        )

        # If the collected data size is larger than the batch size, store the extra data back to the buffer
        if collected_size > self.batch_size:
            extra_data = n_tuple(
                *(
                    data[:, -collected_size + self.batch_size :, ...]
                    if data is not None
                    else None
                    for data in output
                )
            )
            self.buffer.insert(0, extra_data)
            extra_state = tuple(
                state[:, -collected_size + self.batch_size :, ...]
                for state in output_state
            )
            self.buffer_state.insert(0, extra_state)

            # Trim the output to have the exact batch size
            output = n_tuple(
                *(
                    data[:, : self.batch_size, ...] if data is not None else None
                    for data in output
                )
            )
            output_state = tuple(
                state[:, : self.batch_size, ...] for state in output_state
            )

        return output, output_state

    def set_finish(self):
        self.finish = True


@ray.remote
class ModelBuffer:
    def __init__(self, flags):
        self.alpha = flags.priority_alpha

        self.t = flags.model_unroll_length
        self.k = flags.model_k_step_return

        self.max_buffer_n = flags.model_buffer_n // self.t + 1  # maximum buffer length
        self.batch_size = flags.model_batch_size  # batch size in returned sample
        self.wram_up_n = (
            flags.model_warm_up_n if not flags.load_checkpoint else flags.model_buffer_n
        )  # number of total transition before returning samples

        self.buffer = []

        self.priorities = None
        self.next_inds = None
        self.cur_inds = [
            None for _ in range(flags.gpu_num_actors + flags.cpu_num_actors)
        ]

        self.base_ind = 0
        self.abs_tran_n = 0
        self.preload_n = 0
        self.clean_m = 0

        self.finish = False

    def write(self, data, rank):
        # data is a named tuple with elements of size (t+2*k-1, n, ...)
        n = data[0].shape[1]

        for m in range(n):
            self.buffer.append(util.tuple_map(data, lambda x: x[:, m]))

        p_shape = self.t * n
        if self.priorities is None:
            self.priorities = np.ones((p_shape), dtype=float)
        else:
            max_priorities = np.full(
                (p_shape), fill_value=self.priorities.max(), dtype=float
            )
            self.priorities = np.concatenate([self.priorities, max_priorities])

        # to record a table for chaining entry
        if self.cur_inds[rank] is not None:
            last_ind = self.cur_inds[rank] - self.base_ind
            mask = last_ind >= 0
            self.next_inds[last_ind[mask]] = (
                len(self.next_inds) + self.base_ind + np.arange(n)
            )[mask]

        if self.next_inds is None:
            self.next_inds = np.full(
                (n),
                fill_value=np.nan,
            )
        else:
            self.next_inds = np.concatenate(
                [self.next_inds, np.full((n), fill_value=np.nan)]
            )
        self.cur_inds[rank] = len(self.next_inds) + self.base_ind - n + np.arange(n)
        self.abs_tran_n += self.t * n

        # clean periordically
        self.clean()

    def read(self, beta):
        if self.priorities is None or self.abs_tran_n < self.wram_up_n:
            return None
        if self.finish:
            return "FINISH"
        return self.prepare(beta)

    def prepare(self, beta):
        buffer_n = len(self.buffer)
        tran_n = len(self.priorities)
        probs = self.priorities**self.alpha
        probs /= probs.sum()
        flat_inds = custom_choice(tran_n, self.batch_size, p=probs, replace=False)

        inds = np.unravel_index(flat_inds, (buffer_n, self.t))

        weights = (tran_n * probs[flat_inds]) ** (-beta)
        weights /= weights.max()

        data = []
        for d in range(len(self.buffer[0])):
            elems = []
            for i in range(self.batch_size):
                elems.append(
                    self.buffer[inds[0][i]][d][
                        inds[1][i] : inds[1][i] + 2 * self.k, np.newaxis
                    ]
                )
            data.append(np.concatenate(elems, axis=1))
        data = type(self.buffer[0])(*data)

        base_ind_pri = self.base_ind * self.t
        abs_flat_inds = flat_inds + base_ind_pri
        return data, weights, abs_flat_inds, self.abs_tran_n - self.preload_n

    def set_finish(self):
        self.finish = True

    def get_processed_n(self):
        if self.finish:
            return "FINISH"
        return self.abs_tran_n - self.preload_n

    def update_priority(self, abs_flat_inds, priorities):
        """Update priority in the buffer; both input
        are np array of shape (update_size,)"""
        base_ind_pri = self.base_ind * self.t

        # abs_flat_inds is an array of shape (model_batch_size,)
        # priorities is an array of shape (model_batch_size, k)
        priorities = priorities.transpose()

        flat_inds = abs_flat_inds - base_ind_pri  # get the relative index
        mask = flat_inds >= 0
        flat_inds = flat_inds[mask]
        priorities = priorities[mask]

        flat_inds = flat_inds[:, np.newaxis] + np.arange(
            self.k
        )  # flat_inds now stores uncarried indexes
        flat_inds_block = flat_inds // self.t  # block index of flat_inds
        carry_mask = ~(flat_inds_block[:, [0]] == flat_inds_block).reshape(-1)
        # if first index block is not the same as the later index block, we need to carry it

        flat_inds = flat_inds.reshape(-1)
        flat_inds_block = flat_inds_block.reshape(-1)
        carry_inds_block = (
            self.next_inds[flat_inds_block[carry_mask] - 1] - self.base_ind
        )  # the correct index block

        flat_inds = flat_inds.astype(float)
        flat_inds[carry_mask] = (
            flat_inds[carry_mask]
            + (-flat_inds_block[carry_mask] + carry_inds_block) * self.t
        )

        priorities = priorities.reshape(-1)
        mask = ~np.isnan(flat_inds) & ~np.isnan(priorities)
        flat_inds = flat_inds[mask].astype(int)
        priorities = priorities[mask]
        self.priorities[flat_inds] = priorities

    def clean(self):
        buffer_n = len(self.buffer)
        if buffer_n > self.max_buffer_n:
            excess_n = buffer_n - self.max_buffer_n
            del self.buffer[:excess_n]
            self.next_inds = self.next_inds[excess_n:]
            self.priorities = self.priorities[excess_n * self.t :]
            self.base_ind += excess_n

    def check_preload(self):
        return (len(self.buffer) >= self.max_buffer_n, len(self.buffer) * self.t)

    def set_preload(self):
        self.preload_n = self.abs_tran_n
        return True


@ray.remote
class GeneralBuffer(object):
    def __init__(self):
        self.data = {}

    def extend_data(self, name, x):
        if name in self.data:
            self.data[name].extend(x)
        else:
            self.data[name] = x
        return self.data[name]

    def update_dict_item(self, name, key, value):
        if not name in self.data:
            self.data[name] = {}
        self.data[name][key] = value
        return True

    def set_data(self, name, x):
        self.data[name] = x
        return True

    def get_data(self, name):
        return self.data[name] if name in self.data else None
