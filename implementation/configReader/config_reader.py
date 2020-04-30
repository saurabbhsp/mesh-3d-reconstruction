import inflection
from os.path import join
from helper.io.json import json_file_reader


def get_path_list(config):
    path_dictionary = {}
    for path_key in config['pathConfiguration']:
        path_dictionary[inflection.underscore(path_key)] = config[
                                                'pathConfiguration'][path_key]
    return path_dictionary


def get_category(config):
    return config['modelConfiguration']['category']


def get_model_params(config):
    file_name = config['modelConfiguration']['trainingRegim'] + \
                "_" + get_category(config) + ".json"

    model_params = json_file_reader(join(
                                config['modelConfiguration']['configBasePath'],
                                get_category(config),
                                file_name))
    return model_params


def get_split_params(config):
    split_config = {}
    split_config['train_split'] = config.getfloat('splitConfig',
                                                  'trainSplit')
    split_config['random_seed'] = config.getint('splitConfig', 'randomSeed')
    split_config['shuffle'] = config.getboolean('splitConfig', 'shuffle')
    split_config['replace'] = config.getboolean('splitConfig', 'replace')
    return split_config


def get_model_id(config):
    return get_category(config)+"_" + \
                        config['modelConfiguration']['trainingRegim'] + \
                        "_"+config['trainingConfiguration']['modelId']

def get_depth_model_id(config):
    return get_category(config)+"_" + \
                        config['trainingConfiguration']['depthModelId']


def get_max_steps(config):
    return config.getint('trainingConfiguration', 'maxSteps')


def get_train_config(config):
    train_config = {}
    train_config['batch_size'] = config.getint('trainingConfiguration',
                                               'batchSize')
    train_config['n_ffd_samples'] = config.getint('trainingConfiguration',
                                                  'nFFDSamples')
    view_list = config['trainingConfiguration']['views'].split(',')
    views = None
    if len(view_list) == 1:
        """Single angle view is used for reconstruction process"""
        views = int(view_list[0])
    else:
        views = []
        for _view in view_list:
            views.append(int(_view))
    train_config['views'] = views
    train_config['point_cloud_sub_samples'] = config.getint(
                                        'trainingConfiguration',
                                        'pointCloudSubSamples')
    train_config['slices'] = config.getint('trainingConfiguration', 'slices')
    train_config['ffd_resample_count'] = config.getint('trainingConfiguration',
                                                       'ffdResampleCount')
    train_config['n_ffd_resamples'] = config.getint('trainingConfiguration',
                                                    'nFFDResamples')
    return train_config

def get_run_config(config):
    """Run configuration"""
    run_config = {}
    run_config['save_checkpoints_secs'] = config.getint('checkpointConfiguration',
                                                        'saveCheckpointsSec'
                                                        )
    run_config['keep_checkpoint_max'] = config.getint('checkpointConfiguration',
                                                      'maxCheckpoint',
                                                      )
    run_config['log_step_count_steps'] = config.getint('checkpointConfiguration',
                                                       'logSteps',
                                                       )

    run_config['save_summary_steps'] = config.getint('checkpointConfiguration',
                                                           'summarySteps',
                                                           )
    """"""
    return run_config
