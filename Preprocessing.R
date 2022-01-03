#!/usr/local/bin/Rscript

# Set working directory
setwd("/Users/av/Desktop/rs-fMRI/CALTECH/Rest")

# Import packages
suppressMessages(library(RNifti))
suppressMessages(library(fslr))

# Import data
suppressMessages(raw_data <- readNifti(list.files(path=".", pattern=".nii.gz")))
print("Step 0 - Data imported")

# Split data
split_data <- list()
l <- length(raw_data)
suppressMessages(for (i in 1:l){
    lower <- 150*(i-1)+1
    upper <- 150*i
    split_data[upper:lower] <- fslsplit(raw_data[[i]], direction="t", verbose=FALSE)
})
print("Step 1 - Splitting process done")

# Brain extraction
brain_e <- list()
suppressMessages(for (i in 1:300){
    temp <- fslbet(split_data[i])
    cog = cog(temp, ceil=TRUE)
    cog = paste("-c", paste(cog, collapse=" "))
    temp2 <- fslbet(temp, opts=cog)
    brain_e <- append(brain_e, list(temp2))
})
print("Step 2 - Brain extraction done")

# Bias correction
bias_c <- list()
suppressMessages(for (i in 1:300){
    temp <- fsl_biascorrect(brain_e[i])
    bias_c <- append(bias_c, list(temp))
})
print("Step 3 - Bias correcion done")

# Merge
merged <- list()
suppressMessages(for (i in 1:2){
    lower <- 150*(i-1)+1
    upper <- 150*i
    temp <- fslmerge(bias_c[upper:lower], opts="-t")
    merged <- append(merged, list(temp))
})
print("Merged")

# Motion correction
motion_c <- list()
suppressMessages(for (i in 1:2){
    temp <- mcflirt(merged[i])
    motion_c <- append(motion_c, list(temp))
})
print("Step 4 - Motion correction done")

# Slice timing correction
slice_c <- list()
suppressMessages(for (i in 1:2){
    temp <- fslslicetimer(motion_c[i], tr=2, acq_order="interleaved", verbose=FALSE)
    slice_c <- append(slice_c, list(temp))
})
print("Step 5 - Slice timing correction done")

# Temporal filtering
temporal_f <- list()
suppressMessages(for (i in 1:2){
    temp <- fslmaths(slice_c[i], opts="-bptf")
    temporal_f <- append(temporal_f, list(temp))
})
print("Step 6 - Temporal filtering done")

# Spatial smoothing
smooth <- list()
suppressMessages(for (i in 1:2){
    temp <- fslsmooth(temporal_f[i], sigma=3.125, smooth_mask=FALSE, verbose=FALSE)
    smooth <- append(smooth, list(temp))
})
print("Step 7 - Spatial smoothing done")

# Done
print("Done!")
