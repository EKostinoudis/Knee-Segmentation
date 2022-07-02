# Knee MRI image segmentation using sparse representation techniques

My diploma thesis. Implemented three different machine learning techniques for
segmenting knee MRI images.

## Project structure

This project structure is total mess, as everyone can see. If anyone read this
project i apologies for the mess, but i hacked everything around without
reformatting so it end up in this situation.

The [thesis](thesis) folder contains the write-up and the presentation of the
thesis.

The [cGen](cGen) folder contains the cython generated code i developed to speed
up the segmentation.

The [utils.py](utils.py) file contains utilities for this project.

The [requirements.txt](requirements.txt) file contains _some_ of the
requirements for this project.

The [registration.py](registration.py) file has the code for the registration
process.

The rest of the files are mainly for the segmentation and plotting.

## Requirements

Apart from the requirements in the [requirements.txt](requirements.txt) file,
you need also to install
[SPAMS](http://thoth.inrialpes.fr/people/mairal/spams/).

## Ruining segmentations in parallel

The notebooks that performs the segmentation tasks can run in parallel by
forking the process. This crashes the Jupiter notebook kernel, so in order to
avoid the crash and run in parallel extract the code from the notebook to a
python script (**.py** file) and run it. This was the easiest and simplest way
to make it run in parallel. 

