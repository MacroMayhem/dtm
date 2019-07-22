import keras
import keras.backend as K
from keras.preprocessing import image
from utils.small_fcns import FCN
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

parent = image.load_img('sample.jpg', target_size=(224, 224))
template = parent.crop((64, 64, 128, 128))
fcn = FCN(num_channels=3).load()
cropped_from = (64, 64, 128, 128)
print(fcn.summary())

parent_width, parent_height = parent.size
template_width, template_height = template.size
col_steps = int(parent_width/template_width)
row_steps = int(parent_height/template_height)

template_img = np.array(template).reshape([-1, 64, 64, 3])

for r in range(row_steps):
    for c in range(col_steps):
        lx, ly, rx, ry = c*template_width, r*template_height, (c+1)*template_width, (r+1)*template_height
        print((lx, ly, rx, ry))
        cropped_parent_img = np.array(parent.crop((lx, ly, rx, ry))).reshape([-1, 64, 64, 3])
        if cropped_from[0] == lx and cropped_from[1] == ly and cropped_from[2] == rx and cropped_from[3] == ry:
            y = Image.new('L', (64, 64), color=255)
        else:
            y = Image.new('L', (64, 64), color=0)

        target = np.array(y).reshape([-1, 64, 64, 1])

        fcn.fit([cropped_parent_img, template_img], target, epochs=50, verbose=1)
        y_pred = fcn.predict([cropped_parent_img, template_img])
        print(y_pred.shape)
        plot_parent_crop = parent.crop((lx, ly, rx, ry))
        plt.figure(1)
        plt.subplot(311)
        plt.imshow(plot_parent_crop)
        plt.subplot(312)
        plt.imshow(template)
        plt.subplot(313)
        plt.imshow(y_pred[0].reshape([64, 64]), cmap='gray', vmin=0, vmax=1)
        plt.show()