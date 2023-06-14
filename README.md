# Replicate Cog model for Deep Floyd IF v1.0


This is an implementation of [Deep Floyd IF](https://huggingface.co/DeepFloyd) as a Cog model. The pipeline is currently tuned for an RTX 3090 with 24gb of GPU memory. CPU offloading is utilized between stages.  16-bit floating point is configured for the precision of the stages as well.  However, the RTX 3090 is capable of utilizing the largest Deep Floyd IF models for each stage (that are available as of this writing):

* first stage:  `DeepFloyd/IF-I-XL-v1.0`
* second stage: `DeepFloyd/IF-II-L-v1.0`
* third stage: `stabilityai/stable-diffusion-x4-upscaler`


## Schedulers

The following schedulers are available:
* `DDPM`
* `DPMSolverMultistep`
* `SDE-DPMSolverMultistep`
* `DPMSolverSinglestep`

Schedulers can be configured for the first stage via the `scheduler` argument, and schedulers can also be independently configured for the second stage via the `super_scheduler` argument.


## Installation and Usage


Provide Hugging Face token:

	export HUGGINGFACE_TOKEN='{{YOUR HUGGINGFACE API TOKEN HERE}}'

Download the pre-trained weights (80+ gb):

    cog run trial-run-pull

Then, you can run predictions:

    cog predict -i prompt="cement mixer churning the World's best potato salad inside"

Or build a standalone image (very large: 82.9 gb on my system):

	cog build

