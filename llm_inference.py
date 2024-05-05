from openai import OpenAI
import base64
from PIL import Image

# Set your OpenAI API key
api_key = ""
client = OpenAI(api_key=api_key)


def resize_image(image_path, target_size=(448, 448)):
    try:
        # Open the image file
        image = Image.open(image_path)

        # Resize the image
        resized_image = image.resize(target_size)

        return resized_image
    except Exception as e:
        # Handle any errors
        print("Error resizing image:", e)
        return None


def send_prompt_to_gpt4_vision(prompt, image_path):
    try:
        resized_image = resize_image(image_path)
        if resized_image is None:
            return None

        # Convert the resized image to base64
        with Image.new("RGB", resized_image.size) as img:
            img.paste(resized_image)
            with open("resized_image.jpg", "wb") as f:
                img.save(f)
            with open("resized_image.jpg", "rb") as f:
                resized_image_data = base64.b64encode(f.read()).decode("utf-8")

        # Include the image data in the prompt
        # prompt_with_image = f"data:image/jpeg;base64,{resized_image_data}\n{prompt}"

        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{resized_image_data}",
                                "detail": "low",
                            },
                        },
                    ],
                }
            ],
            max_tokens=300,
            temperature=0.5,
            # stop=None
        )

        # Return the generated response
        return response.choices[0].message.content
    except Exception as e:
        # Handle any errors
        print("Error:", e)
        return None


# Example usage
prompt = """
The following image is of a body of water.
The body of water might have trash in it, even far in the distance, an oil spill, or might be clean.
If there is trash in the water, please output the label "trash".
If there is an oil spill, please output the label "oil".
Otherwise output the label "clean".
Only output the label, nothing else.
"""
# Explain what you see in the image
image_path = "/Users/carter/dev/src/github/CV_Final_Project/examples/image3.jpeg"  # Path to your image file
response = send_prompt_to_gpt4_vision(prompt, image_path)
if response:
    print("Generated description:", response)
