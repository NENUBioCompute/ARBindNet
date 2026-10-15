
import sys
import os
import time
import numpy as np
import configparser
import abc 
from six import with_metaclass

import tensorflow as tf

#DEBUG
#from tensorflow.python import debug as tf_debug
import warnings

current_milli_time = lambda: time.time() * 1000.0

class TestLoop(with_metaclass(abc.ABCMeta)):
    """Class to test a model.

    Attributes:
        logFolder_ (string): Path to the log folder.
        numVotes_ (int): Number of votes on the test.
        gpuId_ (string): Id of the GPU Device used.
        gpuMem_ (float): Percentage of the gpu memory used.
        sess_ (Session): Tensorflow session.
    """
    
    def __init__(self, pConfig, pdbname):
        """Constructor.

        Args:
            pConfig (dictionary): Dictionary with the configuration of the 
                train loop.
        """

        #Load the train loop parameters.
        self.gpuId_ = pConfig['gpuid']
        self.gpuMem_ = float(pConfig['gpumem'])

        #Initialize the train loop.
        self.__create_datasets__()
        self.__create_model__()
        self.__create_tf_session__()
        self.__create_savers__()


        #Init variables.
        self.sess_.run(tf.global_variables_initializer())
        self.sess_.run(tf.local_variables_initializer())
        
        self.__load_parameters__()


    @abc.abstractmethod
    def __create_datasets__(self):
        """Method to create the datasets.
        """
        pass


    @abc.abstractmethod
    def __create_model__(self):
        """Method to create the model.
        """
        pass


    def __create_tf_session__(self):
        """Method to create the tensorflow session and other tensorflow objects.
        """

        #Create session
        gpuOptions = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpuMem_, 
            visible_device_list=self.gpuId_)
        self.sess_ = tf.Session(config=tf.ConfigProto(gpu_options=gpuOptions))

        #DEBUG
        #self.sess_ = tf_debug.LocalCLIDebugWrapperSession(self.sess_)
        #self.sess_.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

    
    @abc.abstractmethod
    def __create_savers__(self):
        """Method to create the saver objects.
        """
        pass
        

    @abc.abstractmethod
    def __load_parameters__(self):
        """Method to load the parameters of a model.
        """
        pass


    @abc.abstractmethod
    def __test_one_epoch__(self):
        """Private method to test on voting step.

        Args:
            pNumVote (int): Current number of vote.
        """
        pass




    def test(self):
        """Method to test the model.
        """

        self.__test_one_epoch__()
