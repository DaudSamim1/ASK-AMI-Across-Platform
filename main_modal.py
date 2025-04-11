import modal
from main import DepoAPI

# Define the Modal image with all required packages
image = modal.Image.debian_slim().pip_install_from_requirements(
    "requirements.txt",
)

# Modal App definition
app = modal.App("depoiq-ask-ami-across-platform")


# Define the function to run the FastAPI app
@app.function(image=image, secrets=[modal.Secret.from_name("ask-ami")])
@modal.asgi_app()
def fastapi_app():
    # Initialize your FastAPI app
    api = DepoAPI()
    fastapi_app = api.app

    # You can add additional FastAPI middleware or routes here if needed
    return fastapi_app
