template = {
    "swagger": "2.0",
    "info": {
        "title": "Depo API",
        "description": "REST API for interacting with deposition data including summaries, transcripts, contradictions, and admissions.",
        "version": "1.0.1",
    },
    # "host": "localhost:5000",
    "basePath": "/",
    # "schemes": ["http", "https"],
    "consumes": ["application/json"],
    "produces": ["application/json"],
    # "securityDefinitions": {
    #     "BearerAuth": {
    #         "type": "apiKey",
    #         "name": "Authorization",
    #         "in": "header",
    #         "description": "JWT Authorization header using the Bearer scheme. Example: 'Authorization: Bearer {token}'",
    #     }
    # },
    "tags": [
        {"name": "Depo", "description": "Endpoints related to deposition processing"}
    ],
}
