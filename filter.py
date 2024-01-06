import json

def find_seg_boxes(annotation_file):
    # Load the annotation file
    with open(annotation_file, 'r') as f:
        data = json.load(f)

    # Map image IDs to filenames
    id_to_filename = {image['id']: image['file_name'] for image in data['images']}

    # Set to store unique image ids with bounding box annotations
    segment_image_ids = set()

    # Loop over all annotations
    for annotation in data['annotations']:
        if 'segmentation' in annotation and 'bbox' in annotation and annotation['segmentation']:
            segment_image_ids.add(annotation['image_id'])


    # Get unique filenames for the image ids
    bounding_box_image_filenames = [id_to_filename[image_id] for image_id in segment_image_ids]

    return bounding_box_image_filenames

if __name__ == "__main__":
    annotation_file = 'C:\\Users\\enoch\\OneDrive\\桌面\\YOLO\\COCO\\test\\_annotations.coco.json'
    segment_image_ids = find_seg_boxes(annotation_file)

    print("Found bounding boxes in the following images:")
    for i in segment_image_ids:
        # print(i.split(".")[0].replace("_jpg",".jpg"))
        print(i)