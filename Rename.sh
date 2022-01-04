#!/bin/bash

cd /Users/av/Desktop/rs-fMRI/Caltech/Nifti

for i in *; do
    sub_name="sub-"
    sub_name="$sub_name$i"
    file_name="sub-"
    file_name="$file_name$i"
    file_name_suffix="_ses-1_task-rest_acq-TR2500_bold"
    file_name="$file_name$file_name_suffix"
    mkdir -p /Users/av/Desktop/rs-fMRI/Caltech/Nifti/$sub_name/ses-1/func
    mv /Users/av/Desktop/rs-fMRI/Caltech/Nifti/$i/session_1/rest_1/rest.nii.gz /Users/av/Desktop/rs-fMRI/Caltech/Nifti/$sub_name/ses-1/func/$file_name.nii.gz
done
