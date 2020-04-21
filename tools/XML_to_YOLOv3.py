import xml.etree.ElementTree as ET
import os
import glob

data_dir = '/OID/Dataset/'
Dataset_names_path = "OID/Dataset/Dataset_names.txt"
Dataset_train = "OID/Dataset/Dataset_train.txt"
Dataset_test = "OID/Dataset/Dataset_test.txt"
is_subfolder = True

Dataset_names = []
      
def ParseXML(img_folder, file):
    for xml_file in glob.glob(img_folder+'/*.xml'):
        tree=ET.parse(open(xml_file))
        root = tree.getroot()
        image_name = root.find('filename').text
        img_path = img_folder+'/'+image_name
        for i, obj in enumerate(root.iter('object')):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in Dataset_names:
                Dataset_names.append(cls)
            cls_id = Dataset_names.index(cls)
            xmlbox = obj.find('bndbox')
            OBJECT = str(int(xmlbox.find('xmin').text))+','+str(int(xmlbox.find('ymin').text))+','+str(int(xmlbox.find('xmax').text))+','+str(int(xmlbox.find('ymax').text))+','+str(cls_id)
            img_path += ' '+OBJECT
        print(img_path)
        file.write(img_path+'\n')


for i, folder in enumerate(['train','test']):
    with open([Dataset_train,Dataset_test][i], "w") as file:
        img_path = os.path.join(os.getcwd()+data_dir+folder)
        if is_subfolder:
            for directory in os.listdir(img_path):
                xml_path = os.path.join(img_path, directory)
                ParseXML(xml_path, file)
        else:
            ParseXML(img_path, file)


print("Dataset_names:", Dataset_names)
with open(Dataset_names_path, "w") as file:
    for name in Dataset_names:
        file.write(str(name)+'\n')
