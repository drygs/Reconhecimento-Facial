import cv2
from deepface import DeepFace
import os
import time
import numpy as np
import pickle
from multiprocessing import Pool, cpu_count
import sys
import captura_fotos
import reconhecimento