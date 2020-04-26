import pandas as pd
import numpy as np
import sklearn
import librosa
import matplotlib.pyplot as plt
import utils

def import_data():
    tracks = utils.load("fma_metadata/tracks.csv")
    # Gives one string of a genre per track, enough as label?
    y = tracks[('track','genre_top')]
    
    features = utils.load("fma_metadata/features.csv")
    # this seems to be statistics over all frames per track of the mfcc:s, like mean, std and kurtosis. 
    X = features["mfcc"]

    
if __name__ == "__main__":
    import_data()