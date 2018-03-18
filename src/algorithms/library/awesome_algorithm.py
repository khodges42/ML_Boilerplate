import numpy as np


class Awesome_Algorithm(object):
    def __init__(self):
        self.model = None
        self.time_window_size = None
        self.metric = None
        self.threshold = 5.0
        self.config = None

    @staticmethod
    def create_model(time_window_size, metric):

        return model

    @staticmethod
    def get_config_file(model_dir_path):
        return model_dir_path + '/' + self.model_name + '-config.npy'

    @staticmethod
    def get_weight_file(model_dir_path):
        return model_dir_path + '/' + self.model_name + '-weights.h5'

    @staticmethod
    def get_architecture_file(model_dir_path):
        return model_dir_path + '/' + self.model_name + '-architecture.json'

    def load_model(self, model_dir_path):
        config_file_path = self.get_config_file(model_dir_path)
        self.config = np.load(config_file_path).item()
        self.metric = self.config['metric']
        self.time_window_size = self.config['time_window_size']
        self.threshold = self.config['threshold']
        self.model = self.create_model(self.time_window_size, self.metric)
        weight_file_path = self.get_weight_file(model_dir_path)
        self.model.load_weights(weight_file_path)

    def fit(self, dataset, model_dir_path, batch_size=None, epochs=None, validation_split=None, metric=None,
            estimated_negative_sample_ratio=None):
        if batch_size is None:
            batch_size = 8
        if epochs is None:
            epochs = 100
        if validation_split is None:
            validation_split = 0.2
        if metric is None:
            metric = 'mean_absolute_error'
        if estimated_negative_sample_ratio is None:
            estimated_negative_sample_ratio = 0.9

        self.time_window_size = dataset.shape[1]
        self.metric = metric

        input_timeseries_dataset = np.expand_dims(dataset, axis=2)

        weight_file_path = Conv1DAutoEncoder.get_weight_file(model_dir_path=model_dir_path)
        architecture_file_path = Conv1DAutoEncoder.get_architecture_file(model_dir_path)
        checkpoint = ModelCheckpoint(weight_file_path)
        self.model = Conv1DAutoEncoder.create_model(self.time_window_size, metric=self.metric)
        open(architecture_file_path, 'w').write(self.model.to_json())
        history = self.model.fit(x=input_timeseries_dataset, y=dataset,
                                 batch_size=batch_size, epochs=epochs,
                                 verbose=Conv1DAutoEncoder.VERBOSE, validation_split=validation_split,
                                 callbacks=[checkpoint]).history
        self.model.save_weights(weight_file_path)

        scores = self.predict(dataset)
        scores.sort()
        cut_point = int(estimated_negative_sample_ratio * len(scores))
        self.threshold = scores[cut_point]

        print('estimated threshold is ' + str(self.threshold))

        self.config = dict()
        self.config['time_window_size'] = self.time_window_size
        self.config['metric'] = self.metric
        self.config['threshold'] = self.threshold
        config_file_path = Conv1DAutoEncoder.get_config_file(model_dir_path=model_dir_path)
        np.save(config_file_path, self.config)

        return history

    def predict(self, df):
        return pred
