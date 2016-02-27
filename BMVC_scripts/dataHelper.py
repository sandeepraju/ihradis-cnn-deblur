from __future__ import print_function
import cv2
import numpy as np
import threading
import sys
import os
import math

import matplotlib.pyplot as plt
from matplotlib._image import NEAREST


class ReaderThread(threading.Thread):
    def __init__(self, imageQueue, fileList, imageDir, scaleFactor=1.0):
        threading.Thread.__init__(self)
        self.imageQueue = imageQueue
        self.fileList = fileList
        self.imageDir = imageDir
        self.scaleFactor = scaleFactor

    def run(self):
        for name in self.fileList:
            try:
                name = self.imageDir + name
                imOrg = cv2.imread(name)
                if imOrg is None:
                    print(name, " ERROR - could not read image.", file=sys.stderr)
                    self.imageQueue.put(False)
                else:
                    if self.scaleFactor != 1.0:
                        imOrg = cv2.resize(imOrg, dsize=(0,0), fx=self.scaleFactor, fy=self.scaleFactor, interpolation=cv2.INTER_AREA)
                    self.imageQueue.put(imOrg)
            except cv2.error as e:
                print(name, " ERROR - cv2.error", str(e), file=sys.stderr)
                self.imageQueue.put(False)
            except:
                print(name, " ERROR - UNKNOWN:", sys.exc_info()[0], file=sys.stderr)                
                self.imageQueue.put(False)

        self.imageQueue.put(None)


class WriterThread(threading.Thread):
    def __init__(self, queue, path):
        threading.Thread.__init__(self)
        self.queue = queue
        self.path = path

    def run(self):
        while True:
            name, data = self.queue.get()
            if name is None:
                break

            if data.shape[2] == 1 or data.shape[2] == 3:

                name += '.png'
                cv2.imwrite(os.path.join(self.path + name), data)
                #imgOut = cv2.resize(imgOut, dsize=(img.shape[1],img.shape[0]))
                #original[:,:,0] = np.repeat(np.mean(original, axis=2, keepdims=True), 3, axis=2)
                #original[:,:,0] *= 1-imgOut* 1.3
                #original[:,:,1] *= 1-imgOut* 1.3
                #original[:,:,2] *= imgOut* 1.3
                #cv2.imshow('OUT2', original /255)
                #cv2.waitKey(1)
                #cv2.imwrite('%s-shown.png' % fileName, original)
            else:
                name += '.npz'
                np.savez_compressed(os.path.join(self.path + name), data=data)

