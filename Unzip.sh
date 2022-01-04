#!/bin/bash

cd /Users/av/Desktop/rs-fMRI/Caltech/Nifti

for i in *; do
    gunzip /Users/av/Desktop/rs-fMRI/Caltech/Nifti/$i/ses-1/func/*.nii.gz
    gunzip /Users/av/Desktop/rs-fMRI/Caltech/Nifti/$i/ses-1/anat/*.nii.gz
done
