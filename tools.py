"""
"""
import os

import numpy as np
import tensorflow as tf
import langdetect


def set_gpu_configurations(params):
    ''' Avoid repetition of this GPU setting block '''
    import tensorflow as tf

    print('Setting GPU configurations.')
    # This block avoids GPU configuration errors
    if params['use_gpu']:
        # This prevents CuDNN 'Failed to get convolution algorithm' error
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)
        # To see list of allocated tensors in case of OOM
        tf.compat.v1.RunOptions(report_tensor_allocations_upon_oom = True)
    else:
        try:
            # Disable all GPUs
            tf.config.set_visible_devices([], 'GPU')
            visible_devices = tf.config.get_visible_devices()
            for device in visible_devices:
                assert device.device_type != 'GPU'
        except:
            print('Invalid device or cannot modify virtual devices once initialized.')
        pass
    return None













    
