from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import random

