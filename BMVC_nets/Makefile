.PHONY: clean train draw test

clean:
	find . -name *.caffemodel  -delete
	find . -name *.solverstate -delete

train:
	~/caffe/build/tools/caffe train --solver=./solver.prototxt 2>&1

draw:
	/usr/bin/python ~/caffe/python/draw_net.py ./net.prototxt ~/tmp/net.png

test:
	/usr/bin/python ./test.py
