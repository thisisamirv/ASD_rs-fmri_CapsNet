#!/bin/bash

cd /Users/av/Desktop/rs-fMRI/Data

for i in *; do
    mv /Users/av/Desktop/rs-fMRI/Data/$i/session_1/anat_1/mprage.nii.gz /Users/av/Desktop/rs-fMRI/Data/$i/session_1/anat_1/$i.nii.gz
done
