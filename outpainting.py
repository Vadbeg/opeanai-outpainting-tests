from io import BytesIO
from pathlib import Path

import requests
from openai import OpenAI
from PIL import Image
from tqdm import tqdm


def load_image(image_path: str | Path) -> Image.Image:
    image = Image.open(image_path)
    image = resize_image_with_proportions(image, height=1024)

    return image


def resize_image_with_proportions(
    image: Image.Image, width: int | None = None, height: int | None = None
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


def prepare_image(image: Image.Image) -> Image.Image:
    # Convert to RGBA
    image = image.convert("RGBA")

    # Adds 20% of the image width to the left and 20% to the right as padding with alpha=0
    new_image = Image.new("RGBA", (int(image.width * 1.4), image.height), (0, 0, 0, 0))
    new_image.paste(image, (int(image.width * 0.2), 0))

    return new_image


def split_image(
    image: Image.Image, size: tuple[int, int] = (1024, 1024)
) -> tuple[Image.Image, Image.Image]:
    width, height = image.size
    half_width = width - size[0]

    left_image = image.crop((0, 0, size[0], size[1]))
    right_image = image.crop((half_width, 0, width, size[1]))

    return left_image, right_image


def convert_image_to_bytes(image: Image.Image) -> bytes:
    with BytesIO() as output:
        image.save(output, format="PNG")
        return output.getvalue()


def outpaint_image(image: Image.Image, client: OpenAI, model: str = "dall-e-2") -> str:
    image_bytes = convert_image_to_bytes(image)

    response = client.images.edit(
        model=model,
        image=image_bytes,
        prompt=" ",
        n=1,
        size="1024x1024",
    )
    image_url = response.data[0].url

    return image_url


def combine_images(
    original_image: Image.Image, left_image: Image.Image, right_image: Image.Image
) -> Image.Image:
    width, height = original_image.size
    width_right_start = width - 1024

    original_image.paste(left_image, (0, 0))
    original_image.paste(right_image, (width_right_start, 0))

    return original_image


if __name__ == "__main__":
    client = OpenAI()

    image_folder = Path("./images")
    result_folder = Path("./results")

    image_paths = list(image_folder.glob("*.jpg"))

    for image_path in tqdm(image_paths, postfix="Outpainting..."):
        image = load_image(image_path)
        original_width, original_height = image.size

        image_with_padding = prepare_image(image)
        left_image, right_image = split_image(image_with_padding)

        left_image_url = outpaint_image(left_image, client=client)
        right_image_url = outpaint_image(right_image, client=client)

        left_image_bytes = requests.get(left_image_url).content
        right_image_bytes = requests.get(right_image_url).content

        left_image = Image.open(BytesIO(left_image_bytes))
        right_image = Image.open(BytesIO(right_image_bytes))

        final_image = combine_images(image_with_padding, left_image, right_image)
        final_image = final_image.convert("RGB")

        final_image.save(result_folder / f"{image_path.stem}.jpg")
