import collections
import pickle
import re

import gtar
from tensorflow import keras

class Trajectory:
    """Interface to save and load models from a GTAR trajectory

    .. note:
      Consistent with the GTAR schema, model weights are saved with a
      dynamic index, which is a string and could indicate a "timestep"
      or other time. When accessed via the `load` function, however,
      the frame is a simple integer index, beginning at 0.

    :param filename: File to save or load from
    :param mode: File open mode: 'r' (read-only), 'w' (overwrite), or 'a' (append)

    """

    def __init__(self, filename, mode='r'):
        self.filename = filename
        self.mode = mode
        self.handle = gtar.GTAR(filename, mode)

    def __enter__(self):
        return self

    def __exit__(self, typ, val, trace):
        self.close()

    def __len__(self):
        return len(self.frames)

    def close(self):
        self.handle.close()

    @property
    def frames(self):
        (_, frames) = self.handle.framesWithRecordsNamed('weight')
        return frames

    def get_weights(self, frame=-1):
        """Returns a list of weight arrays for a model stored at the given frame index

        :param frame: integer index of the step to load. Can be negative to count from the end.
        """
        (_, frames) = self.handle.framesWithRecordsNamed('weight')
        frame_index = frames[frame]

        weight_records = collections.defaultdict(dict)
        shape_records = collections.defaultdict(dict)
        weight_pattern = re.compile(r'keras/layer/(?P<layer>\d+)/weight/(?P<weight>\d+)')
        for rec in self.handle.getRecordTypes():
            match = weight_pattern.search(rec.getGroup())
            if not match:
                continue

            layer = int(match.group('layer'))
            weight = int(match.group('weight'))
            if rec.getName() == 'weight':
                weight_records[layer][weight] = rec
            elif rec.getName() == 'shape':
                shape_records[layer][weight] = rec

        all_weights = []
        for (i, records) in sorted(weight_records.items()):
            for weight_index in range(len(records)):
                weight_rec = records[weight_index]
                shape_rec = shape_records[i][weight_index]
                shape = self.handle.getRecord(shape_rec, frame_index)
                weight = self.handle.getRecord(weight_rec, frame_index)
                weight = weight.reshape(shape)
                all_weights.append(weight)

        return all_weights

    def load(self, frame=-1):
        """Loads a model stored at the given frame index

        :param frame: integer index of the step to load. Can be negative to count from the end.
        """
        model_description = self.handle.readStr('keras/model.json')
        assert model_description

        extra_classes = self.handle.readBytes('keras/layer_classes.pkl')
        extra_classes = pickle.loads(extra_classes) if extra_classes else {}

        model = keras.models.model_from_json(model_description, extra_classes)

        all_weights = self.get_weights(frame)

        model.set_weights(all_weights)
        return model

    def save(self, model, frame=None, only_weights=False):
        """Save a model description and/or current state

        :param frame: Frame index (string) to save as. If not given, do not save weights.
        :param only_weights: If True, only save the current model weights, not the model architecture.
        """
        if not only_weights:
            model_json = model.to_json()
            layer_classes = {type(layer).__name__: type(layer) for layer in model.layers}
            layer_classes = pickle.dumps(layer_classes)

            self.handle.writeStr('keras/model.json', model_json)
            self.handle.writeBytes('keras/layer_classes.pkl', layer_classes)
        else:
            assert frame, 'Trying to save only the weights of a model without a frame given'

        dtypes = {'float32': 'f32',
                  'float64': 'f64'}
        if frame:
            for (i, layer) in enumerate(model.layers):
                for (j, weight) in enumerate(layer.get_weights()):
                    dtype_string = dtypes[weight.dtype.name]
                    group = 'keras/layer/{}/weight/{}'.format(i, j)
                    self.handle.writePath('{}/frames/{}/weight.{}.uni'.format(group, frame, dtype_string), weight)
                    self.handle.writePath('{}/shape.u32.uni'.format(group), weight.shape)

    def save_weights(self, model, frame):
        """Save (only) the current model weights.

        :param model: Keras Model object containing weights to save
        :param frame: Frame index (string) to save as
        """
        return self.save(model, frame, only_weights=True)

    def save_model(self, model):
        """Save (only) the current model architecture.

        :param model: Keras Model object to save
        """
        return self.save(model)
