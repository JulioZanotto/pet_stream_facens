import os
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from PIL import Image
from sklearn.svm import SVC
import torch
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

classifier = joblib.load('svm_pet.joblib')

pre_trained = models.resnet18(pretrained=True)
feat_extractor = nn.Sequential(*list(pre_trained.children())[:-1])
feat_extractor.to(device)
feat_extractor.eval()

transform_image = transforms.Compose([transforms.ToTensor(),
                                     transforms.Resize((224,224)),
                                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


def transform_img(file):
    img = Image.open(file)
    img_t = transform_image(img)
    arr = feat_extractor(img_t.unsqueeze(0).to(device))
    arr = torch.reshape(arr, (1, 512))
    pred = classifier.predict(arr.detach().cpu().numpy())
    return pred


st.title("Pet Classifier")
st.header("Please upload a pet image")

file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    product = transform_img(file)
    product = 'Cat' if product[0] == 0 else 'Dog'
    st.write("## The pet is a {}".format(product))
