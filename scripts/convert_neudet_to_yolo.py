import xml.etree.ElementTree as ET
import glob
import os
import shutil
from tqdm import tqdm
from sklearn.model_selection import train_test_split

raw_data_dir = "../data/NEU-DET"             
output_dir = "../datasets/yolo_dataset"              

for split in ['train', 'val', 'test']:
    os.makedirs(f"{output_dir}/images/{split}", exist_ok=True)
    os.makedirs(f"{output_dir}/labels/{split}", exist_ok=True)

CLASSES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']

def convert_box(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x, y, w, h)

def process_dataset():
    process_split('validation', 'val')
    xml_files = glob.glob(f"{raw_data_dir}/train/annotations/*.xml")
    train_files, test_files = train_test_split(xml_files, test_size=0.2, random_state=42)
    
    process_files(train_files, 'train', 'train')
    process_files(test_files, 'train', 'test')
    
def process_split(source_split, target_split):
    """Xử lý toàn bộ một split"""
    xml_files = glob.glob(f"{raw_data_dir}/{source_split}/annotations/*.xml")
    process_files(xml_files, source_split, target_split)

def process_files(xml_files, source_split, target_split):
    """Xử lý danh sách các file XML"""
    for xml_file in tqdm(xml_files):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        filename = root.find('filename').text
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
             filename = filename + ".jpg" 
             
        obj = root.find('object')
        if obj is not None:
            class_name = obj.find('name').text
        else:
            continue  
             
        src_img_path = f"{raw_data_dir}/{source_split}/images/{class_name}/{filename}"
        dst_img_path = f"{output_dir}/images/{target_split}/{filename}"
        
        if os.path.exists(src_img_path):
            shutil.copy(src_img_path, dst_img_path)
        else:
            basename = os.path.splitext(filename)[0]
            possible_exts = ['.jpg', '.JPG', '.bmp', '.BMP', '.png', '.PNG']
            found = False
            for ext in possible_exts:
                src_img_path = f"{raw_data_dir}/{source_split}/images/{class_name}/{basename}{ext}"
                if os.path.exists(src_img_path):
                    shutil.copy(src_img_path, f"{output_dir}/images/{target_split}/{basename}{ext}")
                    filename = basename + ext  
                    found = True
                    break
            if not found:
                print(f"Không tìm thấy ảnh: {filename} trong {class_name}/")
                continue 

        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        
        txt_filename = os.path.splitext(filename)[0] + ".txt"
        txt_path = f"{output_dir}/labels/{target_split}/{txt_filename}"
        
        with open(txt_path, 'w') as out_file:
            for obj in root.iter('object'):
                class_id = 0 
                
                xmlbox = obj.find('bndbox')
                b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), 
                     float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
                
                bb = convert_box((w, h), b)
                out_file.write(str(class_id) + " " + " ".join([str(a) for a in bb]) + '\n')

def create_data_yaml():
    """Tạo file data.yaml cho YOLO"""
    yaml_content = """# NEU-DET Steel Surface Defect Detection Dataset
# Single class: defect (gom tất cả các loại lỗi)

path: /kaggle/working/yolo_dataset
train: images/train
val: images/val
test: images/test

# Classes
nc: 1  # number of classes
names: ['defect']  # class names
"""
    
    yaml_path = f"{output_dir}/data.yaml"
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)

if __name__ == "__main__":
    process_dataset()
    create_data_yaml()