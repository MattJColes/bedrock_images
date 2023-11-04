from enum import Enum
import logging

import numpy as np
from PIL import Image, ImageOps
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import cv2
import torch
from torchvision.transforms import GaussianBlur

env = "DEV" # Change to PROD for production
if env == "DEV":
    logging.basicConfig(filename='images.log',level=logging.INFO)
else:
    logging.basicConfig(filename='images.log',level=logging.WARNING)


class PhotoTheme(Enum):
    CHRISTMAS = 1
    EASTER = 2
    CAVEMAN = 3
    PROGRAMMER = 4
    DEVSHED = 5
    SUPERSTAR = 6


class PhotoManipulation():
    def __init__(self, theme: PhotoTheme = PhotoTheme.CHRISTMAS):
        # cuda = nvidia, mps = apple silicon, cpu = non gpu accelerated
        if torch.cuda.is_available():
            logging.info("Using Nvidia CUDA")
            self.compute_type = "cuda"
        elif torch.backends.mps.is_available():
            logging.info("Using Apple Silicon")
            self.compute_type = "mps"
        else:
            logging.info("Using CPU")
            self.compute_type = "cpu"
        self.repo_id = "stabilityai/stable-diffusion-2-inpainting"
        self.pipe = DiffusionPipeline.from_pretrained(self.repo_id, torch_dtype=torch.float16, revision="fp16")
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.photo_x: int = 512
        self.photo_y: int = 512
        self.pipe.to(self.compute_type)
        self.theme: PhotoTheme = theme
        self.prompt_text: str = "Photo"
        self.negative_prompt_text: str = "out of frame, lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature, closed eyes, squinting, deformed eyes"
        self._define_theme_prompt()

    def _define_theme_prompt(self):
        if self.theme == PhotoTheme.CHRISTMAS:
            self.prompt_text = "Photo of a person wearing a Santa hat and a Santa outfit. The background of the photo is a snowy forest full of christmas trees during winter season with snow flakes falling from sky and christmas lights in the distance. Reindeer and elves."
        elif self.theme == PhotoTheme.EASTER:
            self.prompt_text = "Photo of a person wearing an easter bunny outfit. The background of the photo is a forest during autumn season with easter eggs scattered throughout"
        elif self.theme == PhotoTheme.CAVEMAN:
            self.prompt_text = "Photo of a extremely hairy person wearing caveman outfit. The background of the photo is a forest during autumn with a cave in the distance and a fireplace nearby"
        elif self.theme == PhotoTheme.PROGRAMMER:
            self.prompt_text = "Photo of a scruffy long haired programmer wearing a hoodie and tshirt. The background of the photo is the inside of their computer lab at night time, very dark, with lots of computer keyboards and screens as well as a few books scattered in the distance, and a coffee mug, lots of dust on a bookshelf"
        elif self.theme == PhotoTheme.DEVSHED:
            self.prompt_text = "Photo of a scruffy, hairy programmer wearing a hoodie and tshirt. Set in the 1990's. They are inside a wood garden shed cabin with corrugated iron, lots of book shelves, computers, books, keyboards and hardware"
        elif self.theme == PhotoTheme.SUPERSTAR:
            self.prompt_text = "Photo of a famous popstar wearing a red dress. Background of a music concert"

    def crop_photo(self, photo):
        # Make picture 512x512
        width, height = photo.size
        if width == self.photo_x and height == self.photo_y:
            # This is the right size!
            logging.info("No cropping needed")
            photo_crop = photo
        elif width > self.photo_x and height > self.photo_y:
            # Crop as too big
            logging.info(f"Cropping needed - image is above {self.photo_x}x{self.photo_y}")
            left = (width - self.photo_x)/2
            top = (height - self.photo_y)/2
            right = (width + self.photo_x)/2
            bottom = (height + self.photo_y)/2
            photo_crop = photo.crop((left, top, right, bottom))
        else:
            # Scale up then crop as too small
            logging.info(f"Scaling and cropping needed - image is below {self.photo_x}x{self.photo_y}")
            aspect_ratio = photo.height / photo.width 
            new_width = self.photo_x
            new_height = int(new_width * aspect_ratio)
            photo_crop = photo.resize((new_width, new_height), Image.Resampling.LANCZOS)
        if env == "DEV":
            photo_crop.save(f"__crop_{person}")
        return photo_crop

    def create_mask(self, photo) -> Image:
        logging.info("Creating mask image for photo")
        photo = photo.convert('RGB')
        cv_image = np.array(photo) 
        mask_image = np.zeros((self.photo_x, self.photo_y, 1), dtype = np.uint8)
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            cv2.rectangle(mask_image, (x, y), (x+w, y+h), (255, 255, 255), -1)
        mask_image = np.squeeze(mask_image, axis=2)

        image_mask = Image.fromarray(mask_image)
        image_mask = ImageOps.invert(image_mask) # invert the black and white masking image for SD local
        blur = GaussianBlur(11,20)
        mask: Image = blur(image_mask)
        if env == "DEV":
            mask.save(f"__mask_{person}")
        return mask

    def update_photo(self, photo, mask):
        logging.info("Making output photo from base photo")
        photo_output = self.pipe(
            prompt=self.prompt_text,
            negative_prompt=self.negative_prompt_text,
            image=photo,
            mask_image=mask,
            num_images_per_prompt=1,
            num_inference_steps=80,
            guidance_scale=9.5,
            width=self.photo_x,
            height=self.photo_y,
            eta=0.0,
            generator=torch.Generator(device=self.compute_type),
            output_type="pil"
        ).images[0]
        photo_output.save(f"output_{person}") # Save our completed image with its seed number as the filename.


def process_photo(image_filename: str):
    pm = PhotoManipulation(theme=PhotoTheme.CHRISTMAS)
    photo: Image = Image.open(image_filename)
    crop_photo: Image = pm.crop_photo(photo)
    mask_photo: Image = pm.create_mask(crop_photo)
    pm.update_photo(photo=crop_photo, mask=mask_photo)


people_photos: list = ["photo.png", "photo2.png", "photo_aaron.png", "photo_paul.png", "photo_matt.png", "photo_bella.png"]
for person in people_photos:
    process_photo(person)
