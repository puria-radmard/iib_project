from classes_utils.ensemble import *

ensemble_method_dict = {
    "basic": AudioEncoderBasicEnsemble, 
    "dropout": AudioEncoderDropoutEnsemble, 
    "add_noise": AudioEncoderAdditiveNoiseEnsemble, 
    "mult_noise": AudioEncoderMultiplicativeNoiseEnsemble
}