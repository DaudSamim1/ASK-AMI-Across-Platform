from flask import Flask, jsonify
from helperFunctions.general_helpers import cPrint
from controllers.depo_controller import DepoController
from dotenv import load_dotenv
from flasgger import Swagger

# Load environment variables
load_dotenv()


class DepoAPI:
    def __init__(self):
        self.app = Flask(__name__)
        self.swagger = Swagger(self.app)
        self.depo_controller = DepoController()
        self.configure_routes()
        self.configure_cors()

    def configure_cors(self):
        @self.app.after_request
        def disable_cors(response):
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Methods"] = (
                "GET, POST, PUT, DELETE, OPTIONS"
            )
            response.headers["Access-Control-Allow-Headers"] = (
                "Content-Type, Authorization"
            )
            return response

    def configure_routes(self):
        @self.app.route("/", methods=["GET"])
        def home():
            return jsonify({"message": "Welcome to the Python Project API!"})

        @self.app.route("/depo/<string:depoiq_id>", methods=["GET"])
        def get_depo_by_Id(depoiq_id):
            """
            Get Depo By depoiq_id
            ---
            tags:
              - Depo
            parameters:
              - name: depoiq_id
                in: path
                type: string
                required: true
                description: The ID of the depo
              - name: category
                in: query
                type: string
                required: false
                description: Optional category filter
            responses:
              200:
                description: Returns the success message
              500:
                description: Internal server error
            """
            try:
                return self.depo_controller.get_depo_by_Id(depoiq_id)
            except Exception as e:
                cPrint(e, "🔹 Error in get_depo_by_Id:", "red")
                return (
                    jsonify(
                        {
                            "error": "Something went wrong in get_depo_by_Id",
                            "details": str(e),
                        }
                    ),
                    500,
                )

        @self.app.route("/depo/add/<string:depoiq_id>", methods=["GET"])
        def add_depo(depoiq_id):
            """
            Add depo data to Pinecone
            ---
            tags:
              - Depo
            parameters:
              - name: depoiq_id
                in: path
                type: string
                required: true
                description: The ID of the depo
            responses:
              200:
                description: Returns the success message
              500:
                description: Internal server error
            """
            try:
                return self.depo_controller.add_depo(depoiq_id)
            except Exception as e:
                cPrint(e, "🔹 Error in add_depo:", "red")
                return (
                    jsonify(
                        {
                            "status": "error",
                            "message": "Something went wrong in add_depo",
                            "details": str(e),
                        }
                    ),
                    500,
                )

        @self.app.route("/depo/talk", methods=["POST"])
        def talk_depo():
            """
            Talk to depo data
            ---
            tags:
              - Depo
            parameters:
              - name: body
                in: body
                required: true
                schema:
                  type: object
                  properties:
                    depoiq_ids:
                      type: array
                      items:
                        type: string
                    user_query:
                      type: string
                    category:
                      type: string
                      enum: ["text", "keywords", "synonyms", "all"]
                    is_unique:
                      type: boolean
            responses:
              200:
                description: Returns the success message
            """
            try:
                return self.depo_controller.talk_depo()
            except Exception as e:
                return (
                    jsonify(
                        {
                            "error": "Something went wrong in talk_summary",
                            "details": str(e),
                        }
                    ),
                    500,
                )

        @self.app.route("/depo/answer_validator", methods=["POST"])
        def answer_validator():
            """
            Validate answers
            ---
            tags:
              - Depo
            parameters:
              - name: body
                in: body
                required: true
                schema:
                  type: object
                  properties:
                    questions:
                      type: array
                      items:
                        type: string
                    depoiq_id:
                      type: string
                    category:
                      type: string
                      enum: ["text", "keywords", "synonyms", "all"]
            responses:
              200:
                description: Returns the success message
            """
            try:
                return self.depo_controller.answer_validator()
            except Exception as e:
                return (
                    jsonify(
                        {
                            "error": "Something went wrong in answer_validator",
                            "details": str(e),
                        }
                    ),
                    500,
                )

        @self.app.route("/depo/ask-ami-agent", methods=["POST"])
        def ask_ami_agent():
            """
            Ask AMI agent
            ---
            tags:
              - Depo
            parameters:
              - name: body
                in: body
                required: true
                schema:
                  type: object
                  properties:
                    depoiq_ids:
                      type: array
                      items:
                        type: string
                    user_query:
                      type: string
            responses:
              200:
                description: Returns the success message
            """
            try:
                return self.depo_controller.ask_ami_agent()
            except Exception as e:
                return (
                    jsonify(
                        {
                            "error": "Something went wrong in ask_ami_agent",
                            "details": str(e),
                        }
                    ),
                    500,
                )

    def run(self, **kwargs):
        self.app.run(**kwargs)


if __name__ == "__main__":
    api = DepoAPI()
    api.run(debug=True)
