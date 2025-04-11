from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from dotenv import load_dotenv
from typing import Optional
from controllers.depo_controller import DepoController
from helperFunctions.utils import cPrint
from docs.types import (
    SummaryCategoriesType,
    TalkDepoRequest,
    AnswerValidatorRequest,
    AskAmiAgentRequest,
)

load_dotenv()


class DepoAPI:
    def __init__(self):
        self.app = FastAPI(
            title="Depo API",
            description="API for legal deposition analysis",
            version="1.0.0",
        )
        self.depo_controller = DepoController()
        self.configure_middleware()
        self.configure_routes()
        self.configure_openapi()

    def configure_middleware(self):
        # Configure CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def configure_openapi(self):
        def custom_openapi():
            if self.app.openapi_schema:
                return self.app.openapi_schema

            openapi_schema = get_openapi(
                title="Depo API",
                version="1.0.0",
                description="API for legal deposition analysis",
                routes=self.app.routes,
            )

            # Customize the OpenAPI schema if needed
            self.app.openapi_schema = openapi_schema
            return self.app.openapi_schema

        self.app.openapi = custom_openapi

    def configure_routes(self):
        @self.app.get("/", tags=["Root"])
        async def home():
            return {"message": "Welcome to the Python Project API!"}

        @self.app.get("/depo/{depoiq_id}", tags=["Depo"])
        async def get_depo_by_Id(
            depoiq_id: str, category: Optional[SummaryCategoriesType] = None
        ):
            try:
                categoryVal = category.value if category else None
                response = await self.depo_controller.get_depo_by_Id(
                    depoiq_id, categoryVal
                )
                return response
            except Exception as e:
                cPrint(e, "🔹 Error in get_depo_by_Id:", "red")
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": "Something went wrong in get_depo_by_Id",
                        "details": str(e),
                    },
                )

        @self.app.get("/depo/add/{depoiq_id}", tags=["Depo"])
        async def add_depo(
            depoiq_id: str, category: Optional[SummaryCategoriesType] = None
        ):
            try:
                categoryVal = category.value if category else None
                response = self.depo_controller.add_depo(depoiq_id, categoryVal)
                return response
            except Exception as e:
                cPrint(e, "🔹 Error in add_depo:", "red")
                raise HTTPException(
                    status_code=500,
                    detail={
                        "status": "error",
                        "message": "Something went wrong in add_depo",
                        "details": str(e),
                    },
                )

        @self.app.post("/depo/talk", tags=["Depo"])
        async def talk_depo(request: TalkDepoRequest):
            try:
                response = self.depo_controller.talk_depo(request)
                return response
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": "Something went wrong in talk_summary",
                        "details": str(e),
                    },
                )

        @self.app.post("/depo/answer_validator", tags=["Depo"])
        async def answer_validator(request: AnswerValidatorRequest):
            try:
                response = self.depo_controller.answer_validator(request)
                return response
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": "Something went wrong in answer_validator",
                        "details": str(e),
                    },
                )

        @self.app.post("/depo/ask-ami-agent", tags=["Depo"])
        async def ask_ami_agent(request: AskAmiAgentRequest):
            try:
                response = self.depo_controller.ask_ami_agent(request)
                return response
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": "Something went wrong in ask_ami_agent",
                        "details": str(e),
                    },
                )

    def run(self, **kwargs):
        import uvicorn

        uvicorn.run(self.app, **kwargs)


if __name__ == "__main__":
    api = DepoAPI()
    api.run(host="0.0.0.0", port=8000)
