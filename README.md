# Kandinsky v3.0

This is an implementation of a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

First, download the pre-trained weights:

    cog run script/download-weights

Then, you can run

**text2img** predictions:

    cog predict -i prompt="a moss covered astronaut with a black background"

**img2img** predictions:
	
	cog predict -i image=@astro.png -i prompt="photo of a lone astronaut standing on a barren planet"
	
