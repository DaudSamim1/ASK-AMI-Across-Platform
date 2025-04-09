from flask import Flask, jsonify
from flasgger import Swagger, swag_from
from dotenv import load_dotenv

from controllers.depo_controller import DepoController
from helperFunctions.general_helpers import cPrint
from swagger_config import template

load_dotenv()


class DepoAPI:
    def __init__(self):
        self.app = Flask(__name__)
        self.swagger = Swagger(self.app, template=template)
        self.depo_controller = DepoController()
        self.configure_routes()
        self.configure_cors()

    def configure_cors(self):
        @self.app.after_request
        def disable_cors(response):
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
            return response

    def configure_routes(self):
        @self.app.route("/", methods=["GET"])
        def home():
            return jsonify({"message": "Welcome to the Python Project API!"})

        @self.app.route("/depo/<string:depoiq_id>", methods=["GET"])
        @swag_from("docs/get_depo_by_id.yml")
        def get_depo_by_Id(depoiq_id):
            try:
                return self.depo_controller.get_depo_by_Id(depoiq_id)
            except Exception as e:
                cPrint(e, "🔹 Error in get_depo_by_Id:", "red")
                return jsonify({"error": "Something went wrong in get_depo_by_Id", "details": str(e)}), 500

        @self.app.route("/depo/add/<string:depoiq_id>", methods=["GET"])
        @swag_from("docs/add_depo.yml")
        def add_depo(depoiq_id):
            try:
                return self.depo_controller.add_depo(depoiq_id)
            except Exception as e:
                cPrint(e, "🔹 Error in add_depo:", "red")
                return jsonify({"status": "error", "message": "Something went wrong in add_depo", "details": str(e)}), 500

        @self.app.route("/depo/talk", methods=["POST"])
        @swag_from("docs/talk_depo.yml")
        def talk_depo():
            try:
                return self.depo_controller.talk_depo()
            except Exception as e:
                return jsonify({"error": "Something went wrong in talk_summary", "details": str(e)}), 500

        @self.app.route("/depo/answer_validator", methods=["POST"])
        @swag_from("docs/answer_validator.yml")
        def answer_validator():
            try:
                return self.depo_controller.answer_validator()
            except Exception as e:
                return jsonify({"error": "Something went wrong in answer_validator", "details": str(e)}), 500

        @self.app.route("/depo/ask-ami-agent", methods=["POST"])
        @swag_from("docs/ask_ami_agent.yml")
        def ask_ami_agent():
            try:
                return self.depo_controller.ask_ami_agent()
            except Exception as e:
                return jsonify({"error": "Something went wrong in ask_ami_agent", "details": str(e)}), 500

    def run(self, **kwargs):
        self.app.run(**kwargs)


if __name__ == "__main__":
    api = DepoAPI()
    api.run(debug=True)
