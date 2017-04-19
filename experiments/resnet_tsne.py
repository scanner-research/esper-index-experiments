import configparser
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot  as plt
import os
import pandas
import scipy
#from sklearn.manifold import TSNE
from MulticoreTSNE import MulticoreTSNE as TSNE # https://github.com/DmitryUlyanov/Multicore-TSNE
import sys

def main():
    DIRNAME = os.path.dirname(os.path.abspath(__file__))

    config = configparser.ConfigParser()
    config.read(os.path.join(DIRNAME, "experiments.ini"))
    config = config['current']

    video_path = config['video_path']
    resnet_features_path = config['resnet_features_path']
    yolo_labels_path = config['yolo_labels']
    results_dir = config['results_dir']
    
    print('load feature vectors')
    resnet_features = np.load(os.path.join(DIRNAME, resnet_features_path))

    print('load the labels from yolo')
    yolo_labels = pandas.read_csv(yolo_labels_path)
    yolo_labels = map(eval, yolo_labels['label'].tolist())
    colors = map(lambda x: 'r' if x else 'b', yolo_labels)
    
    print('project into 2d')
    projection = np.load('/tmp/tnse.npy')
    
    #model = TSNE(n_components = 2, random_state = 0, n_jobs=2)
    #projection = model.fit_transform(resnet_features)
    #np.save('/tmp/tnse.npy', projection)
    
    print('plot the results')
    plt.scatter(projection[:,0], projection[:,1], c=colors)
    plt.savefig(os.path.join(results_dir, 'resnet_tsne.png'))
    
if (__name__ == '__main__'):
    main()
