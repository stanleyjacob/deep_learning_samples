import sys
import caffe
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, 'python')
caffe.set_mode_gpu()

solver = caffe.SGDSolver('solver.prototxt')
solver.net.params['lstm1'][2].data[15:30] = 5

a = np.arange(0, 32, 0.01)
d = 0.5 * np.sin(2*a) - 0.05 * np.cos( 17 * a + 0.8  ) + 0.05 * np.sin( 25 * a + 10 ) - 0.02 * np.cos( 45 * a + 0.3)
d = d / max(np.max(d), -np.min(d))
d = d - np.mean(d)

niter = 5000
train_loss = np.zeros(niter)
solver.net.params['lstm1'][2].data[15:30]=5
solver.net.blobs['clip'].data[...] = 1
for i in range(niter) :
    seq_idx = i % (len(d) / 320)
    solver.net.blobs['clip'].data[0] = seq_idx > 0
    solver.net.blobs['label'].data[:,0] = d[ seq_idx * 320 : (seq_idx+1) * 320 ]
    solver.step(1)
    train_loss[i] = solver.net.blobs['loss'].data
   
plt.plot(np.arange(niter), train_loss)

solver.test_nets[0].blobs['data'].reshape(2,1)
solver.test_nets[0].blobs['clip'].reshape(2,1)
solver.test_nets[0].reshape()
solver.test_nets[0].blobs['clip'].data[...] = 1
preds = np.zeros(len(d))
for i in range(len(d)):
    solver.test_nets[0].blobs['clip'].data[0] = i > 0
    preds[i] =  solver.test_nets[0].forward()['ip1'][0][0]
    
plt.plot(np.arange(len(d)), preds)
plt.plot(np.arange(len(d)), d)
plt.show()
