#================================================================
#
#   File name   : Detection_to_XML.py
#   Author      : PyLessons
#   Created date: 2020-09-27
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
#   Description : converts YOLO detection to XML file
#
#===============================================================
from textwrap import dedent
from lxml import etree
import glob
import os
import cv2
import time

def CreateXMLfile(path, file_name, image, bboxes, NUM_CLASS):
    boxes = []
    for bbox in bboxes:
        boxes.append([bbox[0].astype(int), bbox[1].astype(int), bbox[2].astype(int), bbox[3].astype(int), NUM_CLASS[int(bbox[5])]])#, bbox[4], NUM_CLASS[int(bbox[5])]])

    if not os.path.exists(path):
        os.makedirs(path)
    os.chdir(path)

    img_name = "XML_"+file_name+".png"
    
    cv2.imwrite(img_name,image)

    annotation = etree.Element("annotation")

    folder = etree.Element("folder")
    folder.text = os.path.basename(os.getcwd())
    annotation.append(folder)

    filename_xml = etree.Element("filename")
    filename_str = img_name.split(".")[0]
    filename_xml.text = img_name
    annotation.append(filename_xml)

    path = etree.Element("path")
    path.text = os.path.join(os.getcwd(), filename_str + ".jpg")
    annotation.append(path)

    source = etree.Element("source")
    annotation.append(source)

    database = etree.Element("database")
    database.text = "Unknown"
    source.append(database)

    size = etree.Element("size")
    annotation.append(size)

    width = etree.Element("width")
    height = etree.Element("height")
    depth = etree.Element("depth")

    img = cv2.imread(filename_xml.text)

    width.text = str(img.shape[1])
    height.text = str(img.shape[0])
    depth.text = str(img.shape[2])

    size.append(width)
    size.append(height)
    size.append(depth)

    segmented = etree.Element("segmented")
    segmented.text = "0"
    annotation.append(segmented)

    for Object in boxes:
        class_name = Object[4]
        xmin_l = str(int(float(Object[0])))
        ymin_l = str(int(float(Object[1])))
        xmax_l = str(int(float(Object[2])))
        ymax_l = str(int(float(Object[3])))

        obj = etree.Element("object")
        annotation.append(obj)

        name = etree.Element("name")
        name.text = class_name
        obj.append(name)

        pose = etree.Element("pose")
        pose.text = "Unspecified"
        obj.append(pose)

        truncated = etree.Element("truncated")
        truncated.text = "0"
        obj.append(truncated)

        difficult = etree.Element("difficult")
        difficult.text = "0"
        obj.append(difficult)

        bndbox = etree.Element("bndbox")
        obj.append(bndbox)

        xmin = etree.Element("xmin")
        xmin.text = xmin_l
        bndbox.append(xmin)

        ymin = etree.Element("ymin")
        ymin.text = ymin_l
        bndbox.append(ymin)

        xmax = etree.Element("xmax")
        xmax.text = xmax_l
        bndbox.append(xmax)

        ymax = etree.Element("ymax")
        ymax.text = ymax_l
        bndbox.append(ymax)

    # write xml to file
    s = etree.tostring(annotation, pretty_print=True)
    with open(filename_str + ".xml", 'wb') as f:
        f.write(s)
        f.close()

    os.chdir("..")
