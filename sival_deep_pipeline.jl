using PyCall
using DrWatson: datadir
using JLD: save
using DataFrames: DataFrame
using CategoricalArrays: CategoricalArray
using StatsBase
using StatsBase: fit, ZScoreTransform

py"""
import sys
import os
import skimage.io
import torch
from PIL import Image
from torchvision import transforms
import torchvision.models as models
import cv2 as cv
import numpy as np
import pickle
import xml.etree.ElementTree as ET
import glob

def read_content(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    list_with_all_boxes = []
    labels = []
    filename = root.find('filename').text

    for boxes in root.iter('object'):
        ymin, xmin, ymax, xmax = None, None, None, None

        ymin = int(boxes.find("bndbox/ymin").text)
        xmin = int(boxes.find("bndbox/xmin").text)
        ymax = int(boxes.find("bndbox/ymax").text)
        xmax = int(boxes.find("bndbox/xmax").text)
        labels.append(boxes.find('name').text)
        list_with_single_boxes = [xmin, ymin, xmax, ymax]
        list_with_all_boxes.append(list_with_single_boxes)

    return filename, list_with_all_boxes, labels

def generate_proposal_bag(image_file, model, proposal_mode = 'edgeboxes', cnn = 'alexnet', max_boxes=200, min_score=0.04, eta=0.2):
    if proposal_mode == 'selectivesearch':
        image = cv.imread(image_file)
        ss = cv.ximgproc.segmentation.createSelectiveSearchSegmentation()
        ss.setBaseImage(image)
        ss.switchToSelectiveSearchFast()
        boxes = ss.process()
    elif proposal_mode == 'edgeboxes':
        edge_model = 'data/model.yml'
        image = cv.imread(image_file)

        edge_detection = cv.ximgproc.createStructuredEdgeDetection(edge_model)
        rgb_im = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        edges = edge_detection.detectEdges(np.float32(rgb_im) / 255.0)

        orimap = edge_detection.computeOrientation(edges)
        edges = edge_detection.edgesNms(edges, orimap)

        edge_boxes = cv.ximgproc.createEdgeBoxes()
        edge_boxes.setMaxBoxes(max_boxes)
        edge_boxes.setMinScore(min_score)
        edge_boxes.setEta(eta)
        boxes = edge_boxes.getBoundingBoxes(edges, orimap)
    else:
        raise ValueError("Please choose the correct proposal_mode: 'edgeboxes' or 'selectivesearch'.")

    proposals = [image[y:y+h,x:x+w] for x,y,w,h in boxes]
    boxes = [np.array([x, y, x+w, y+h]) for x,y,w,h in boxes]
    
    proposal_features = []
    
    if cnn == 'alexnet':
        model.eval()
        preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
        for proposal in proposals:
            image = Image.fromarray(proposal)
            input_tensor = preprocess(image)
            input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
            x = model.avgpool(model.features(input_batch))
            x = torch.flatten(x, 1)
            proposal_features.append(model.classifier[0:2](x).detach().numpy())
    else:
        model.eval()
        preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
        for proposal in proposals:
            image = Image.fromarray(proposal)
            input_tensor = preprocess(image)
            input_batch = input_tensor.unsqueeze(0)
            x = model.features(input_batch)
            x = model.avgpool(x)
            x = torch.flatten(x, 1)
            proposal_features.append(model.classifier[0](x).detach().numpy())
            
    return proposal_features, boxes
        
def create_bags_from_sival_dataset(proposal_mode='edgeboxes', cnn='alexnet', path=None, imgs_per_class=-1, max_boxes=200, min_score=0.04, eta=0.2):
    bags = []
    labels = []
    boxes_bags = []
    paths = []

    if cnn == 'alexnet':
        model = models.alexnet(pretrained=True)
    else:
        model = models.vgg11(pretrained=True)
    
    for i, foldername in enumerate(os.listdir(path)):
        imgs = glob.glob(path+foldername+'/*.jpg')[0:imgs_per_class]
        for img_path in imgs:
            print(img_path)
            features, proposal_boxes = generate_proposal_bag(img_path, model, proposal_mode, cnn, max_boxes, min_score, eta)
            bags.append(features)
            boxes_bags.append(proposal_boxes)
            labels.append(foldername)
            paths.append(img_path)
                
    return bags, labels, boxes_bags, paths
"""

max_boxes = 200
min_score = 0.04
eta = 0.2

if !isfile(datadir("model.yml"))
    message = "

The EdgeBoxes model does not exist at: " * datadir("model.yml") * ".
Please download it using the following command in the shell:

wget https://github.com/opencv/opencv_extra/raw/master/testdata/cv/ximgproc/model.yml.gz -O " * datadir("model.yml.gz") * "
cd " * datadir() * "
gunzip model.yml.gz
"

    throw(ErrorException(message))
end

bags, labels, boxes, img_paths = py"create_bags_from_sival_dataset"(path=datadir("SIVAL/"), imgs_per_class=-1, max_boxes=max_boxes, min_score=min_score, eta=eta)
bounding_boxes = [boxes[i,:] for i in 1:length(labels)]

# Remove bags without any instances returned by the Edgebox detector
nis = [size(b, 1) for b in bags]
[deleteat!(data, nis .== 0) for data in (bags, labels, bounding_boxes, img_paths)]
println("Removed " * string(sum(nis .== 0)) * " bags where no EdgeBoxes were detected.")

# Standardize features
nis = [size(b, 1) for b in bags]
X_cut = [sum(nis[1:ni])-nis[ni]+1:sum(nis[1:ni]) for ni in 1:length(nis)]
X = vcat([vcat(bags[i]...) for i in 1:size(bags, 1)]...)
dt = fit(ZScoreTransform, X, dims=1)
X_norm = StatsBase.transform(dt, X)
bags = [X_norm[cut,:] for cut in X_cut]

filename = "sival_deep_" * string(max_boxes) * "_" * string(min_score) * "_" * string(eta) * ".jld"
save(datadir(filename), "bags", bags, "labels", labels, "bounding_boxes", bounding_boxes, "img_paths", img_paths)

println("Saved processed SIVAL dataset in " * datadir(filename))
