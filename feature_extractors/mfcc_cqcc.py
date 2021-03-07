# Script to extract MFCC & CQCC Features from .flac files
# Requirments
#
#
#
#
#
#
#
#
#
#
# How to Run


# python file_name.py --data-path --output-path --labels-path

from python_speech_features import mfcc
from CQCC.cqcc import cqcc
import scipy.io.wavfile as wav
import soundfile as sf
import os
import numpy as np
import pickle
import argparse