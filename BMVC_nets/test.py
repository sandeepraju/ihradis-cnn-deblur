import numpy as np
import matplotlib.pyplot as plt
import caffe
import os
import skimage.color as color
import scipy.ndimage.interpolation as sni
from PIL import Image

def main():
    # enable gpu mode for caffe
    caffe.set_mode_gpu()

    # load the model
    net = caffe.Net(
        './net.deploy',
        './net_iter_10000.caffemodel',
        caffe.TEST)

    print [(k, v.data.shape) for k, v in net.blobs.items()]
    print [(k, v[0].data.shape, v[1].data.shape) for k, v in net.params.items()]
    
    OUT_LAYER = 'conv1'
    (h_in, w_in) = net.blobs['data'].data.shape[2:]
    (h_out, w_out) = net.blobs[OUT_LAYER].data.shape[2:]

    # import ipdb; ipdb.set_trace()
    
    # load input and configure preprocessing
    # im = caffe.io.load_image('./input.png', color=True)
    im = np.array(Image.open('./input.png'))
    transformer = caffe.io.Transformer({
        'data': net.blobs['data'].data.shape
    })
    # transformer.set_mean('data', np.load('../python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
    transformer.set_transpose('data', (2,0,1))
    transformer.set_channel_swap('data', (2,1,0))
    # transformer.selfet_raw_scale('data', 255.0)
    # make classification map by forward and print prediction indices at each location
    _i = np.asarray([transformer.preprocess('data', im)])
    # out = net.forward_all(data=_i)
    import ipdb; ipdb.set_trace()
    # BIAS
    # ipdb> net.params['conv1'][1].data.shape
    # (3,)
    # SET: np.array([1.0,1.0,1.0], dtype=np.float32) 
    # ipdb> net.params['conv1'][0].data.shape
    # (3, 3, 1, 1)
    # ipdb> net.params['conv1'][0].data[:,:,0,0]
    # array([[             nan,              nan,              nan],
    #        [ -1.10639974e-01,  -1.43969283e-01,  -1.45278290e-01],
    #        [ -2.69767375e+05,  -3.09310781e+05,  -4.07637500e+05]], dtype=float32)
    # ipdb> net.params['conv1'][0].data[:,:,0,0].shape
    # (3, 3)

    out = net.forward(data=_i, end=OUT_LAYER)
    
    # img_rgb = caffe.io.load_image('./input.png')
    # import ipdb; ipdb.set_trace()
    # # net.blobs['input_data'] = img_rgb
    # im = np.array(img_rgb)
    # # im_input = im[np.newaxis, :, :, :]
    # im_input = im.transpose([2, 0, 1]).reshape(1, 3, 50, )
    # net.forward_all(data=im_input)
    # _img = np.zeros((h_out, w_out, 3), 'uint8')
    # _img[..., 0] = net.blobs['conv2'].data[0][0]
    # _img[..., 1] = net.blobs['conv2'].data[0][1]
    # _img[..., 2] = net.blobs['conv2'].data[0][2]
    # img_out = Image.fromarray(np.array(_img), mode='RGB')
    o = transformer.deprocess('data', net.blobs[OUT_LAYER].data[0])
    # import ipdb; ipdb.set_trace()
    img_out = Image.fromarray(o, mode='RGB')
    img_out.save('/home/srp243/tmp/net_out.png')

if __name__ == '__main__':
    main()
