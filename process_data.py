from pycocotools.coco import COCO
import numpy as np
import cv2

BANANA_COLOR = (0, 255, 0) # green
APPLE_COLOR = (0, 0, 255) # red
ORANGE_COLOR = (255, 0, 0) # blue

def create_blank_image(width, height):
    image = np.zeros((height, width, 3), np.uint8)
    image[:] = tuple((255, 255, 255))
    return image

def create_abstract_image(coco, coco_image):
    image = create_blank_image(width=coco_image['width'], height=coco_image['height'])

    # get all image containing apple, banana, orange
    apple_category = coco.getCatIds(catNms=['apple'])[0]
    banana_category = coco.getCatIds(catNms=['banana'])[0]
    orange_category = coco.getCatIds(catNms=['orange'])[0]
    fruit_category = coco.getCatIds(catNms=['apple', 'banana', 'orange'])

    annotation_id = coco.getAnnIds(imgIds=coco_image['id'], catIds=fruit_category, iscrowd=False)
    annotations = coco.loadAnns(annotation_id)
    for annotation in annotations:
        segmentation = annotation['segmentation'][0]
        points = np.reshape(segmentation, (len(segmentation)//2, 2))
        points = points.reshape((-1, 1, 2))
        if annotation['category_id'] == apple_category:
            cv2.fillPoly(image, np.int32([points]), APPLE_COLOR, 8)
        elif annotation['category_id'] == banana_category:
            cv2.fillPoly(image, np.int32([points]), BANANA_COLOR, 8)
        elif annotation['category_id'] == orange_category:
            cv2.fillPoly(image, np.int32([points]), ORANGE_COLOR, 8)

    return image

def get_fruit_images_id(coco):
    categories = coco.loadCats(coco.getCatIds())

    # get all image containing apple, banana, orange
    apple_category = coco.getCatIds(catNms=['apple'])
    banana_category = coco.getCatIds(catNms=['banana'])
    orange_category = coco.getCatIds(catNms=['orange'])

    apple_images_id = coco.getImgIds(catIds=apple_category)
    banana_images_id = coco.getImgIds(catIds=banana_category)
    orange_images_id = coco.getImgIds(catIds=orange_category)

    fruit_images_id = list(set(apple_images_id) | set(banana_images_id) | set(orange_images_id))
    return fruit_images_id

def generate_dataset(coco, source_dir, target_dir, coco_images_id):
    for coco_image_id in coco_images_id:
        # open image from coco
        coco_image = coco.loadImgs(coco_image_id)[0]
        original_image = cv2.imread(source_dir + coco_image['file_name'])
        # save original image into ground_truth
        cv2.imwrite(target_dir + "ground_truth/" + coco_image['file_name'], original_image)
        # generate abstract_image
        abstract_image = create_abstract_image(coco, coco_image)
        # save into abstract images
        cv2.imwrite(target_dir + "abstract_images/" + coco_image['file_name'], abstract_image)


# edit these base on your own configurations
training_data_dir = "/home/ernestlwt/data/coco/train2017/"
validation_data_dir = "/home/ernestlwt/data/coco/val2017/"
training_annotation_file = "/home/ernestlwt/data/coco/annotations/instances_train2017.json"
validation_annotation_file = "/home/ernestlwt/data/coco/annotations/instances_val2017.json"

save_train_dir = "./train/"
save_val_dir = "./val/"

coco_val = COCO(validation_annotation_file)
coco_train = COCO(training_annotation_file)

val_images_id = get_fruit_images_id(coco_val)
generate_dataset(coco_val, validation_data_dir, save_val_dir, val_images_id)

train_images_id = get_fruit_images_id(coco_train)
generate_dataset(coco_train, training_data_dir, save_train_dir, train_images_id)
