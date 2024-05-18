import yaml

def load_config_file(fname=None):
    # Created by Santiago 26/04/2024
    ''' Helper method to load the configuration parameters from yaml file
        first the default yaml file is loaded and the upadted with the
        new most updated configuration yaml file given by fname.
        args:
            fname: string-> path to the yaml config file
        output:
            parameters: python dict
    '''
    
    with open(fname) as config:
        parameters=yaml.safe_load(config)
    
    return parameters