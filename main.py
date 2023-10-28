import io
import os
from enum import Enum

import numpy as np
from PIL import Image
from dotenv import load_dotenv
import cv2
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
from torchvision.transforms import GaussianBlur


class PhotoTheme(Enum):
    CHRISTMAS = 1
    EASTER = 2


class PhotoManipulation():
    def __init__(self, theme: PhotoTheme = PhotoTheme.CHRISTMAS):
        load_dotenv()
        self.env: dict = dict(os.environ)
        self.stability_api = client.StabilityInference(
            key=self.env["STABILITY_KEY"],
            verbose=True,
            engine="stable-inpainting-512-v2-0",
        )
        self.theme: PhotoTheme = theme
        self.prompt_text: str = "Photo"
        self.negative_prompt_text = "Blurred, stylized, cartoony, summer, bokeh, murky"
        self._define_theme_prompt()

    def _define_theme_prompt(self):
        if self.theme == PhotoTheme.CHRISTMAS:
            self.prompt_text = "Photo of a person wearing a Santa hat and a Santa outfit. The background of the photo is a snowy forest during winter christmas season"
        elif self.theme == PhotoTheme.EASTER:
            self.prompt_text = "Photo of a person wearing an easter bunny outfit. The background of the photo is a forest during autumn season with easter eggs scattered throughout"

    def create_photo(self):
        pass

    def create_mask(self, photo) -> Image:
        image = cv2.imread(photo)
        cv_image = np.zeros((512, 512, 1), dtype = np.uint8)
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            cv2.rectangle(cv_image, (x, y), (x+w, y+h), (255, 255, 255), -1)
        cv_image = np.squeeze(cv_image, axis=2)

        image_mask = Image.fromarray(cv_image)
        blur = GaussianBlur(11,20)
        mask: Image = blur(image_mask)
        return mask

    def update_photo(self, photo, mask):
        answers = self.stability_api.generate(
            prompt=[generation.Prompt(text=self.prompt_text,parameters=generation.PromptParameters(weight=1)), generation.Prompt(text=self.negative_prompt_text,parameters=generation.PromptParameters(weight=-1))],
            init_image=photo,
            mask_image=mask,
            start_schedule=1.0,
            end_schedule=0.1,
            cfg_scale=15.0,
            steps=50,
            width=1024,
            height=1024,
            sampler=generation.SAMPLER_K_DPMPP_2S_ANCESTRAL,
            guidance_preset=generation.GUIDANCE_PRESET_FAST_BLUE,
            style_preset="photographic"
        )
        for resp in answers:
            for artifact in resp.artifacts:
                if artifact.type == generation.ARTIFACT_IMAGE:
                    global img2
                    img2 = Image.open(io.BytesIO(artifact.binary))
                    img2.save(f"output_{person}") # Save our completed image with its seed number as the filename.


def process_photo(image_filename: str):
    pm = PhotoManipulation(PhotoTheme.EASTER)
    photo: Image = Image.open(image_filename)
    mask: Image = pm.create_mask(image_filename)
    pm.update_photo(photo, mask)


people_photos = ["photo_aaron.png", "photo_bella.png", "photo_paul.png", "photo_matt.png"]
for person in people_photos:
    process_photo(person)
