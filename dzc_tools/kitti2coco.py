import cv2 
import os 
import json
from tqdm import tqdm

def get_prototype():
    ret = {}
    ret["info"] = {"Description": "kitti dataset used to test mono-3d det made by dzc"}
    ret["licenses"] = []
    ret["images"] = []
    ret["annotations"] = []
    ret["categories"] = []
    return ret

def txt2boxes(lbls_file, invalid_lbls = ["DontCare", "Misc"]):
    with open(lbls_file, "r") as f:
        lines = f.readlines()
    bboxes = []
    for line in lines:
        line = line.strip()
        info = line.split(" ")
        if info[0] in invalid_lbls:
            continue
        bboxes.append(box2dic(info))
    return bboxes

def box2dic(box):
    ret = {}
    ret["x0"], ret["y0"], ret["x1"], ret["y1"] = map(float, box[4:8])
    ret["lbl"] = box[0]
    ret["is_truncated"] = int(float(box[1]))
    ret["is_occluded"] = int(float(box[2]))
    ret["alpha"] = float(box[3])
    ret["dimension"] = box[8:11]
    ret["location"] = box[11:14]
    ret["rotation_y"] = box[14]
    return ret

def fill_image_info(img, ret, img_id, img_name):
    im = cv2.imread(img)  
    h,w,_ = im.shape
    tmp_info = {"id": img_id, "file_name": img_name, "width": w, "height": h}
    ret["images"].append(tmp_info)
    return ret

def fill_cate_info(total_ret):
    ret = total_ret["categories"]
    ret.append({"supercategory": "vehicle", "id":1, "name":"Car"})
    ret.append({"supercategory": "vehicle", "id":2, "name":"Truck"})
    ret.append({"supercategory": "vehicle", "id":3, "name":"Van"})
    ret.append({"supercategory": "vehicle", "id":4, "name":"Tram"})
    ret.append({"supercategory": "cyclist", "id":5, "name":"Cyclist"}) 
    ret.append({"supercategory": "pedestrian", "id":6, "name":"Pedestrian"})
    ret.append({"supercategory": "pedestrian", "id":7, "name":"Person_sitting"})
    
    cls2idx_dict = {"Car": 1, "Truck": 2, "Van": 3, "Tram": 4, "Cyclist": 5, "Pedestrian": 6,  "Person_sitting": 7}
    return total_ret, cls2idx_dict

def fill_anno_info(lbl_file, ret, img_id, box_id, cls2idx_dict):
    bboxes = txt2boxes(lbl_file)
    for box in bboxes:
        tmp_info = {}
        tmp_info["iscrowd"] = 0
        tmp_info["image_id"] = img_id
        tmp_info["bbox"] = [box["x0"], box["y0"], box["x1"] - box["x0"] + 1, box["y1"] - box["y0"] + 1]
        tmp_info["area"] = (box["x1"] - box["x0"] + 1) * (box["y1"] - box["y0"] + 1)
        tmp_info["category_id"] = cls2idx_dict[box["lbl"]]
        tmp_info["id"] = box_id
        box_id += 1
        ret["annotations"].append(tmp_info)
    return ret, box_id

if __name__ == "__main__":
    imgs_dir = "/data/ori_kitti/kitti/training/image_2/"
    lbls_dir = "/data/ori_kitti/kitti/training/label_2/"

    ret = get_prototype()
    # fill category info
    ret, cls2idx_dict = fill_cate_info(ret)

    # we should traverse dir and fill images and annotations for each frame 
    img_id = 0
    box_id = 0
    for img in tqdm(os.listdir(imgs_dir)):
        # fill image info 
        ret = fill_image_info(os.path.join(imgs_dir, img), ret, img_id, img)
        img_id += 1
        # fill annotation_info
        ret, box_id = fill_anno_info(os.path.join(lbls_dir, img.split(".")[0] + ".txt"), ret, img_id, box_id, cls2idx_dict)

    with open("./kitti.json","w") as f:
        json.dump(ret,f)



