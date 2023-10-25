
import os
import time
import tensorflow as tf
import numpy as np
from PIL import Image

data_path = '../all_data/NORMAL/IM-0170-0001.jpeg'
cmd = f"ls {data_path}"

# print(os.listdir(data_path)[0])
a = np.array(Image.open(data_path))
b = a.T

print(a.shape, b.shape)
total = 0
n = 100
for i in range(n):
    t1 = time.time()
    c = tf.matmul(
        tf.convert_to_tensor(a, np.float32), 
        tf.convert_to_tensor(b, np.float32)
    )
    t2 = time.time()
    total += (t2-t1)
    print(round(t2-t1, 3), end=' ')

print(c.shape)
print(f"Time: {total/n}")