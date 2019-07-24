import keras
import keras.backend as K
from keras.preprocessing import image
from utils.small_fcns import IndicatorNN
from PIL import Image
import numpy as np
from keras.datasets import cifar100
import matplotlib.pyplot as plt
from cifarGenerator import CifarFeatureGenerator
import itertools

train_generator = CifarFeatureGenerator(mode='train')
val_generator = CifarFeatureGenerator(mode='test')

try:
    fcn = keras.models.load_model('feat_cifar_model.h5')
except:
    fcn = IndicatorNN(num_channels=3).load()

save_callback = keras.callbacks.ModelCheckpoint('feat_cifar_model.h5', monitor='loss', verbose=1, save_best_only=True, mode='min')
tb_callback = keras.callbacks.TensorBoard(log_dir='./logs/', histogram_freq=0, write_graph=True, write_images=False)
history = fcn.fit_generator(generator=train_generator, validation_data=val_generator, epochs=2, callbacks=[save_callback, tb_callback])


fcn = keras.models.load_model('cifar_model.h5')
val_generator = CifarGenerator(mode='test')
predictions = fcn.predict_generator(val_generator)

tp = 0
fp = 0
fn = 0
tn = 0

p = 0
n = 0

for key, val in val_generator.labels.items():

    x, t, gt = val[0], val[1], val[2]
    '''plt.figure(1)
    plt.subplot(311)
    plt.imshow((x * 255).astype(np.uint8), vmin=0, vmax=255)
    plt.subplot(312)
    plt.imshow((t * 255).astype(np.uint8), vmin=0, vmax=255)
    plt.subplot(313)

    plt.imshow(pred, cmap='gray', vmin=0, vmax=1)
    #plt.show()
    plt.savefig('results/{}.png'.format(key))
    plt.close()'''

    pred = np.squeeze(predictions[key])
    label = np.mean(pred)
    if label > 0.9:
        label = 1
    else:
        label = 0
    if gt == 1:
        p += 1
    else:
        n += 1

    if label == gt and label == 1:
        tp += 1.0
    elif label == gt and label == 0:
            tn += 1.0
    elif label != gt and label == 1:
                fp += 1.0
    else:
                fn += 1.0
print('Precision: {}\nRecall: {}'.format(tp/(tp+fp), tp/p))


parent = image.load_img('sample.jpg', target_size=(32, 32))
template = parent.crop((16, 16, 32, 32))
parent_width, parent_height = parent.size
template_width, template_height = template.size
col_steps = int(parent_width/template_width)
row_steps = int(parent_height/template_height)

template_img = np.array(template).reshape([-1, 16, 16, 3])

for r in range(row_steps):
    for c in range(col_steps):
        lx, ly, rx, ry = c*template_width, r*template_height, (c+1)*template_width, (r+1)*template_height
        print((lx, ly, rx, ry))
        cropped_parent_img = np.array(parent.crop((lx, ly, rx, ry))).reshape([-1, 16, 16, 3])
        y_pred = fcn.predict([cropped_parent_img, template_img])
        print(y_pred.shape)
        plot_parent_crop = parent.crop((lx, ly, rx, ry))
        plt.figure(1)
        plt.subplot(311)
        plt.imshow(plot_parent_crop)
        plt.subplot(312)
        plt.imshow(template)
        plt.subplot(313)
        plt.imshow(y_pred[0].reshape([16, 16]), cmap='gray', vmin=0, vmax=1)
        plt.show()
