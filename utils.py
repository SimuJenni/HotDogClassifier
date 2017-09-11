import os


def count_images(dir, classes):
    count = 0
    for c in classes:
        class_dir = os.path.join(dir, c)
        count += len([name for name in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, name))])
    return count
