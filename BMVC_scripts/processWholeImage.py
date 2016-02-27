#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import os
import sys
import argparse
import glob
import time
import cv2
import Queue

from dataHelper import WriterThread

import caffe

def getCutout(image, x1, y1, x2, y2, border):
    assert(x1 >= 0 and y1 >= 0)
    assert(x2 > x1 and y2 >y1)
    assert(border >= 0)

    return cv2.getRectSubPix(image, (y2-y1 + 2*border, x2-x1 + 2*border), (((y2-1)+y1) / 2.0, ((x2-1)+x1) / 2.0))


def main(argv):
    pycaffe_dir = os.path.dirname(__file__)

    parser = argparse.ArgumentParser()
    # Optional arguments.
    parser.add_argument(
        "--model_def",
        help="Model definition file.",
        required=True
    )
    parser.add_argument(
        "--pretrained_model",
        help="Trained model weights file.",
        required=True
    )
    parser.add_argument(
        "--out_scale",
        help="Scale of the output image.",
        required=True,
        type=float
    )
    parser.add_argument(
        "--extract_layer",
        help="Name of the blob to extract.",
        required=True
    )
    parser.add_argument(
        "--output_is_image",
        help="Name of the blob to extract.",
        action="store_true"
    )
    parser.add_argument(
        "--output_path",
        help="Output path.",
        default=''
    )
    parser.add_argument(
        "--border",
        help="Name of the blob to extract.",
        required=True,
        type=int
    )
    parser.add_argument(
        "--tile_resolution",
        help="Resolution of processing tile.",
        required=True,
        type=int
    )
    parser.add_argument(
        "--suffix",
        help="Suffix of the output file.",
        default="-deblur",
    )
    parser.add_argument(
        "--gpu",
        action='store_true',
        help="Switch for gpu computation."
    )
    args = parser.parse_args()

    # Make classifier.
    if args.gpu:
        print('GPU mode', file=sys.stderr)
        caffe.set_mode_gpu()
    classifier = caffe.Classifier(args.model_def, args.pretrained_model, caffe.TEST)


    inputs = [line.strip() for line in sys.stdin]

    print("Classifying %d inputs." % len(inputs), file=sys.stderr)

    resolution = args.tile_resolution
    outResolution = int(resolution * args.out_scale)
    boundary = args.border

    writerQueue = Queue.Queue(32)
    writer = WriterThread(queue=writerQueue, path=args.output_path)
    writer.start()

    for fileName in inputs:
        img = cv2.imread(fileName).astype(np.float32)
        original = np.copy(img)
        img[:,:,0] -= 103.939
        img[:,:,1] -= 116.779
        img[:,:,2] -= 123.68
        #img -= 127
        img *= 0.004

        outShape = [int(img.shape[0] * args.out_scale),
                    int(img.shape[1] * args.out_scale),
                    classifier.blobs[args.extract_layer].channels]
        imgOut = np.zeros(outShape)

        imageStartTime = time.time()
        for x, xOut in zip(range(0, img.shape[0], resolution), range(0, imgOut.shape[0], outResolution)):
            for y, yOut in zip(range(0, img.shape[1], resolution), range(0, imgOut.shape[1], outResolution)):

                start = time.time()

                region = getCutout(img, x, y, x+resolution, y+resolution, boundary)
                data = region.transpose([2, 0, 1]).reshape(1, 3, region.shape[0], region.shape[1])

                out = classifier.forward_all(data=data)
                L=args.extract_layer
                out = out[L].reshape(out[L].shape[1], out[L].shape[2], out[L].shape[3]).transpose(1, 2, 0)

                if imgOut.shape[2] == 3:
                    out /= 0.004
                    #out += 127
                    out[:,:,0] += 103.939
                    out[:,:,1] += 116.779
                    out[:,:,2] += 123.68


                if out.shape[0] != outResolution:
                    print("Warning: size of net output is %d px and it is expected to be %d px" % (out.shape[0], outResolution))
                if out.shape[0] < outResolution:
                    print("Error: size of net output is %d px and it is expected to be %d px" % (out.shape[0], outResolution))
                    exit()

                xRange = min((outResolution, imgOut.shape[0] - xOut))
                yRange = min((outResolution, imgOut.shape[1] - yOut))

                imgOut[xOut:xOut+xRange, yOut:yOut+yRange, :] = out[0:xRange, 0:yRange, :]
                imgOut[xOut:xOut+xRange, yOut:yOut+yRange, :] = out[0:xRange, 0:yRange, :]

                #imgOut = out
                #print("%.2fs" % (time.time() - start), end=" ", file=sys.stderr)
                print(".", end="", file=sys.stderr)
                sys.stdout.flush()

                #scale = 1280.0 / max(imgOut.shape)
                #cv2.imshow('OUT', cv2.resize(imgOut, dsize=(0,0), fx=scale, fy=scale)/255)
                #cv2.waitKey(1)

        print("IMAGE DONE %s" % (time.time() - imageStartTime))
        #imgOut = imgOut[2:-2, 2:-2, :]
        #imgOut = np.append(imgOut[:, 1, :].reshape(200,1,3), imgOut[:, 0:-1, :], axis=1)
        #imgOut = np.append(imgOut[1, :, :].reshape(1,200,3), imgOut[0:-1, :, :], axis=0)

        writerQueue.put((fileName + args.suffix, imgOut))

    writerQueue.put((None, None))
    writer.join()

if __name__ == '__main__':
    main(sys.argv)
