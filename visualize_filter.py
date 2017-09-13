import numpy as np
from keras.models import load_model
from vis.utils import utils
from vis.visualization import visualize_activation, get_num_filters
from vis.input_modifiers import Jitter

import matplotlib.pyplot as plt

model_path = 'first_try.h5'
model = load_model(model_path)
model.summary()


# The name of the layer we want to visualize
# You can see this in the model definition.
layer_name = 'activation_3'
layer_idx = utils.find_layer_idx(model, layer_name)
num_filters = 10

# Visualize num_filters random filters in this layer.
filters = np.random.permutation(get_num_filters(model.layers[layer_idx]))[:num_filters]

# Generate input image for each filter.
vis_images = []
for i, idx in enumerate(filters):
    print('Filter {}/{}'.format(i+1, num_filters))
    img = visualize_activation(model, layer_idx, filter_indices=idx, input_range=(0., 1.0),
                               lp_norm_weight=0.04, tv_weight=0.004, input_modifiers=[Jitter(0.05)])
    vis_images.append(img)

# Generate stitched image palette with 8 cols.
stitched = utils.stitch_images(vis_images, cols=5)
plt.axis('off')
plt.imshow(stitched)
plt.title(layer_name)
plt.show()