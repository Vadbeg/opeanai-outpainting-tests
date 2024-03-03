from io import BytesIO
from pathlib import Path

import requests
from openai import OpenAI
from PIL import Image
from tqdm import tqdm


class OpenAIOutpainting:
    def __init__(self, size: int = 1024, percentage: float = 0.2):
        self._client = OpenAI()
        self._size = size
        self._percentage = percentage

    def load_image(
        self,
        image_path: str | Path,
        desired_height: int = 1024,
    ) -> Image.Image:
        image = Image.open(image_path)
        image = self._resize_image_with_proportions(image, height=desired_height)
        return image

    def _resize_image_with_proportions(
        self, image: Image.Image, width: int | None = None, height: int | None = None
    ) -> Image.Image:
        if width and height:
            image = image.resize((width, height))
        elif width:
            ratio = width / float(image.size[0])
            height = int(float(image.size[1]) * float(ratio))
            image = image.resize((width, height))
        elif height:
            ratio = height / float(image.size[1])
            width = int(float(image.size[0]) * float(ratio))
            image = image.resize((width, height))
        return image

    def _prepare_image(
        self,
        image: Image.Image,
        percentage: float = 0.2,
    ) -> Image.Image:
        image = image.convert("RGBA")
        factor = 1 + percentage * 2
        new_image = Image.new(
            "RGBA", (int(image.width * factor), image.height), (0, 0, 0, 0)
        )
        new_image.paste(image, (int(image.width * percentage), 0))
        return new_image

    def _split_image(
        self, image: Image.Image, size: tuple[int, int] = (1024, 1024)
    ) -> tuple[Image.Image, Image.Image]:
        width, height = image.size
        half_width = width - size[0]
        left_image = image.crop((0, 0, size[0], size[1]))
        right_image = image.crop((half_width, 0, width, size[1]))
        return left_image, right_image

    def _convert_image_to_bytes(self, image: Image.Image) -> bytes:
        with BytesIO() as output:
            image.save(output, format="PNG")
            return output.getvalue()

    def _outpaint_image(
        self,
        image: Image.Image,
        model: str = "dall-e-2",
    ) -> str:
        image_bytes = self._convert_image_to_bytes(image)
        openai_size = f"{self._size}x{self._size}"
        response = self._client.images.edit(
            model=model,
            image=image_bytes,
            prompt=" ",
            n=1,
            size=openai_size,
        )
        image_url = response.data[0].url
        return image_url

    def _combine_images(
        self,
        original_image: Image.Image,
        left_image: Image.Image,
        right_image: Image.Image,
    ) -> Image.Image:
        width, _ = original_image.size
        width_right_start = width - self._size
        original_image.paste(left_image, (0, 0))
        original_image.paste(right_image, (width_right_start, 0))
        return original_image

    def perform_outpainting(
        self,
        image: Image.Image,
    ) -> Image.Image:
        image_with_padding = self._prepare_image(image, percentage=self._percentage)

        left_image, right_image = self._split_image(
            image_with_padding, size=(self._size, self._size)
        )

        left_image_url = self._outpaint_image(left_image, model="dall-e-2")
        right_image_url = self._outpaint_image(right_image, model="dall-e-2")

        left_image_bytes = requests.get(left_image_url).content
        right_image_bytes = requests.get(right_image_url).content
        left_image = Image.open(BytesIO(left_image_bytes))
        right_image = Image.open(BytesIO(right_image_bytes))

        final_image = self._combine_images(image_with_padding, left_image, right_image)
        final_image = final_image.convert("RGB")

        return final_image


if __name__ == "__main__":
    client = OpenAI()

    image_folder = Path("./images")
    result_folder = Path("./results")

    image_paths = list(image_folder.glob("*.jpg"))

    size = 1024
    percentage = 0.3

    openai_outpainting = OpenAIOutpainting(size=size, percentage=percentage)

    for image_path in tqdm(image_paths, postfix="Outpainting..."):
        image = openai_outpainting.load_image(image_path, desired_height=size)
        final_image = openai_outpainting.perform_outpainting(
            image=image,
        )

        final_image.save(result_folder / f"{image_path.stem}.jpg")
