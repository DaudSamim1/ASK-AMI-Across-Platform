import modal
from asgiref.wsgi import WsgiToAsgi

from main import DepoAPI


# Define the Modal image with all required packages
image = modal.Image.debian_slim().pip_install_from_requirements(
    "requirements.txt",
)

# Modal App definition
app = modal.App("depoiq-ask-ami-acroess-platform")


# Define the function to run the Flask app
@app.function(image=image, secrets=[modal.Secret.from_name("ask-ami")])
@modal.asgi_app()
def depo_app():
    flask_app = DepoAPI().app
    return WsgiToAsgi(flask_app)
