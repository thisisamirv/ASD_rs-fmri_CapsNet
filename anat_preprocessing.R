# Brain extraction
25 brain_e <- list()
26 suppressMessages(for (i in 1:300){
27     temp <- fslbet(split_data[i])
28     cog = cog(temp, ceil=TRUE)
29     cog = paste("-c", paste(cog, collapse=" "))
30     temp2 <- fslbet(temp, opts=cog)
31     brain_e <- append(brain_e, list(temp2))
32 })
33 print("Step 2 - Brain extraction done")
