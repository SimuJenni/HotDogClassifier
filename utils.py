import os
from scipy.misc import imread, imresize
import numpy as np
import matplotlib.pyplot as plt


def count_images(dir, classes):
    counts = {}
    for c in classes:
        class_dir = os.path.join(dir, c)
        counts[c] = len([name for name in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, name))])
    count = sum(counts.values())
    print('Examples per class in {}: {}'.format(dir, counts))
    return count


def test_on_image(model, image_path, target_size, classes):
    img_orig = imread(image_path)
    img = imresize(img_orig, target_size)
    img_ = np.expand_dims(img, 0)
    pred = model.predict(img_, batch_size=1)
    pred_idx = np.argmax(pred, axis=1)
    predicted_class = classes[pred_idx[0].astype(np.int)]
    plt.figure()
    plt.imshow(img_orig)
    plt.title('Predicted class: {}'.format(predicted_class))
    plt.show()
