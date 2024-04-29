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
    default_fname = "configs/default.yaml"
    
    
    with open(default_fname) as default_config: parameters=yaml.safe_load(default_config)
    if fname!=None:
        with open(fname, 'r') as update_config:
            try:
                # Converts yaml document to python object
                update=yaml.safe_load(update_config)

            except yaml.YAMLError as e:
                print(e)
        # updates default dict with most updates parameters
        parameters.update(update)
    
    return parameters