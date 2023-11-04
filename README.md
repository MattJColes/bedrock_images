# Experimenting with Stable Diffusion

### Getting Started:
* Have Python and Pip installed, then in this directory run
    * Run `python3 -m venv venv`
    * Run `source venv/bin/activate`
    * Run `pip install -r requirements.txt`
* Download some photos and update the list of photos in `main.py` people_photos array. We resize the photo if not 512x512 but for best results have your face in the middle of the image.
* In `python main.py` set the self.compute_type field to the GPU or CPU you are using. `cuda` = nvidia, `mps` = apple silicon, `cpu` = non gpu accelerated.
* Run `python main.py`
* And Enjoy!

### Notes:
Currently am using just Stable Diffusion as the model `stable-inpainting-512-v2-0` has yet to come into Bedrock and is needed for best results.