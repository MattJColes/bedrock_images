# Experimenting with Bedrock and Stable Diffusion

### Getting Started:
* Setup an Stability.AI account and get an API key https://platform.stability.ai/docs/getting-started/authentication
* Store the API key in a .env file. I have an example without a key called example.env here
* Have Python and Pip installed, then in this directory run
    * Run `python3 -m venv venv`
    * Run `source venv/bin/activate`
    * Run `pip install -r requirements.txt`
* Download some photos and update the list of photos in `main.py` people_photos array.
* Run `python main.py`
* And Enjoy!

### Notes:
Currently am using just Stable Diffusion as I need the model `stable-inpainting-512-v2-0` within Bedrock for best results.