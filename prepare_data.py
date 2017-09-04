import os
from shutil import copyfile

source_dir = '../food-101'
target_dir = '../seefood'

train_file = os.path.join(source_dir, 'meta/train.txt')
test_file = os.path.join(source_dir, 'meta/test.txt')
imgs_dir = os.path.join(source_dir, 'images')

with open(train_file) as tf:
    img_names = tf.readlines()
    for line in img_names:
        label_img = line.strip()
        label, im_id = label_img.split('/')
        img_path = os.path.join(imgs_dir, label, '{}.jpg'.format(im_id))
        dst_dir = os.path.join(target_dir, 'train', label)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

        dst_path = os.path.join(dst_dir, '{}.jpg'.format(im_id))
        if not os.path.exists(dst_path):
            copyfile(img_path, dst_path)

with open(test_file) as tf:
    img_names = tf.readlines()
    for line in img_names:
        label_img = line.strip()
        label, im_id = label_img.split('/')
        img_path = os.path.join(imgs_dir, label, '{}.jpg'.format(im_id))
        dst_dir = os.path.join(target_dir, 'test', label)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

        dst_path = os.path.join(dst_dir, '{}.jpg'.format(im_id))
        if not os.path.exists(dst_path):
            copyfile(img_path, dst_path)
