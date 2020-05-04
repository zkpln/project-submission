from matplotlib.image import imread
import sys
import os
from six.moves import cPickle
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

def main(argv):
    if len(argv) != 2:
        print("Usage: `python3 {} <batch>` where batch is 0-10".format(argv[0]))
        sys.exit(1)

    # Path to batch
    path = os.path.abspath("data/ims/ads/{}".format(argv[1]))
    npath = os.path.abspath("data/ims/not_ads/{}".format(argv[1]))

    data = []

    # Iterate through every file in the directory
    for filename in os.listdir(path):
        img_path = os.path.join(path, filename)
        f, ext = os.path.splitext(img_path)

        if ext == '.jpg':
            img = {}
            img['data'] = imread(img_path)
            if len(img['data'].shape) == 2:
                dim = np.zeros((200, 200))
                img['data'] = np.stack((img['data'], dim, dim), axis=2)

            img['data'] = img['data']
            img['label'] = 'ad'

            data.append(img)

    # Iterate through every non-ad file in the directory
    for filename in os.listdir(npath):
        img_path = os.path.join(npath, filename)
        f, ext = os.path.splitext(img_path)

        if ext == '.jpg':
            img = {}
            img['data'] = imread(img_path)
            if len(img['data'].shape) == 2:
                dim = np.zeros((200, 200))
                img['data'] = np.stack((img['data'], dim, dim), axis=2)

            img['data'] = img['data']
            img['label'] = 'not_ad'

            data.append(img)

    outfile = os.path.join(path, 'data')
    with open(outfile, 'wb') as out:
        cPickle.dump(data, out)

    print('Saved pickle to {}'.format(outfile))

if __name__ == "__main__":
    main(sys.argv)
