from . import Trajectory
from tensorflow import keras

class GTARLogger(keras.callbacks.Callback):
    """Keras callback to log all weights of a model at a given frequency

    :param filename: Filename to save to
    :param period: Frequency (in epochs or batches) to save
    :param when: String indicating when to save: one of `pre_batch`, `post_batch`, `pre_epoch`, or `post_epoch`
    :param append: If True, append to instead of overwriting the file if it exists already
    :param step_offset: Offset to apply to the epoch or batch index
    """
    def __init__(self, filename, period=1, when='post_epoch', append=True,
                 step_offset=0, *args, **kwargs):
        self.filename = filename
        self.period = period
        self.when = when
        self.append = append
        self.step_offset = step_offset
        self.batches = 0

        assert when in ('pre_batch', 'post_batch', 'pre_epoch', 'post_epoch')
        super().__init__(*args, **kwargs)

    def on_train_begin(self, logs={}):
        mode = 'a' if self.append else 'w'
        self.trajectory = Trajectory(self.filename, mode)
        self.trajectory.save_model(self.model)

    def on_train_end(self, logs={}):
        self.trajectory.close()

    def _save(self, index, required_time):
        if self.when != required_time:
            return

        index = index + self.step_offset

        if index%self.period == 0:
            self.trajectory.save_weights(self.model, str(index))

    def on_batch_begin(self, index, logs={}):
        return self._save(self.batches, 'pre_batch')

    def on_batch_end(self, index, logs={}):
        result = self._save(self.batches, 'post_batch')
        self.batches += 1
        return result

    def on_epoch_begin(self, index, logs={}):
        return self._save(index, 'pre_epoch')

    def on_epoch_end(self, index, logs={}):
        return self._save(index, 'post_epoch')
