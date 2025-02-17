from flask import Flask, jsonify, request
from flasgger import Swagger
import requests
import jwt
import time
import uuid
import numpy as np
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from enum import Enum
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve API keys from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
GRAPHQL_URL = os.getenv("GRAPHQL_URL")

# Ensure keys are loaded
if not OPENAI_API_KEY or not PINECONE_API_KEY:
    raise ValueError("Missing API Keys! Check your .env file.")


app = Flask(__name__)
swagger = Swagger(app)

# Dummy data storage
items = [
    {"id": 1, "name": "Item 1", "description": "This is item 1"},
    {"id": 2, "name": "Item 2", "description": "This is item 2"},
]


class SummaryCategory(Enum):
    ADMISSION = "admission"
    CONTRADICTION = "contradiction"
    HIGH_LEVEL_SUMMARY = "high_level_summary"


def generate_token():
    # Generate access token
    access_token_payload = {
        "userId": "677ffbb1f728963ffa7b8dca",
        "companyId": "66ff06f4c50afa83deecb020",
        "isSuperAdmin": False,
        "role": "ADMIN",
    }

    # generate access token
    accessToken = "Bearer " + jwt.encode(
        access_token_payload, JWT_SECRET_KEY, algorithm="HS256"
    )

    return accessToken


# Function to generate embeddings from OpenAI
def generate_embedding(text, model="text-embedding-3-small"):
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        text = text.replace("\n", " ")  # Clean text
        response = client.embeddings.create(input=[text], model=model)
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return np.zeros(1536).tolist()  # Return a zero vector if embedding fails


# Function to store deposition data in Pinecone
def store_depositions_in_pinecone(deposition_data, depo_id):
    try:
        print(f"Starting Pinecone insertion for depoIQ_ID: {depo_id}")

        # Initialize Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)

        # Define index name
        index_name = "depositions-index"

        # Create index if it doesn't exist
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=1536,  # OpenAI embedding dimension
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws", region="us-east-1"
                ),  # Use supported region
            )

        # Wait until the index is ready
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)

        index = pc.Index(index_name)

        # Check if deposition already exists
        dummy_vector = np.zeros(1536).tolist()
        existing_entries = index.query(
            vector=dummy_vector, filter={"depoIQ_ID": depo_id}, top_k=1
        )

        if existing_entries["matches"]:
            print(f"Deposition {depo_id} already exists in Pinecone. Skipping insert.")
            return {
                "status": "skipped",
                "message": "Deposition already exists in Pinecone.",
            }

        # Insert depositions into Pinecone
        inserted_count = 0
        vectors_to_upsert = []

        for key, value in deposition_data["summary"].items():
            text = value["text"]
            embedding = generate_embedding(text)

            metadata = {
                "depoIQ_ID": depo_id,
                "category": SummaryCategory.HIGH_LEVEL_SUMMARY.value,
                "summary_ID": "summary_" + depo_id,
            }

            vectors_to_upsert.append(
                {
                    "id": str(uuid.uuid4()),  # Unique ID
                    "values": embedding,  # Embedding vector
                    "metadata": metadata,  # Metadata
                }
            )

        # Bulk upsert into Pinecone
        if vectors_to_upsert:
            index.upsert(vectors=vectors_to_upsert)
            inserted_count = len(vectors_to_upsert)
            print(
                f"Successfully inserted {inserted_count} deposition summaries into Pinecone."
            )

        return {
            "status": "success",
            "message": f"Stored {inserted_count} depositions in Pinecone.",
        }

    except Exception as e:
        print(f"Error inserting into Pinecone: {e}")
        return {"status": "error", "message": str(e)}


# 🏠 Home Endpoint
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Welcome to the Python Project API!"})


# Function to query Pinecone and match depositions
def query_pinecone(query_text, depo_id=None, top_k=3):
    try:
        print(f"Querying Pinecone for: {query_text} and depo_id: {depo_id}")

        # Initialize Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index_name = "depositions-index"
        index = pc.Index(index_name)

        # Generate query embedding
        query_vector = generate_embedding(query_text)

        # Define filter criteria (search within `depo_id`)
        filter_criteria = {}
        if depo_id:
            filter_criteria["depoIQ_ID"] = depo_id

        # Search in Pinecone
        results = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            filter=filter_criteria,
        )

        # Format the results
        matched_results = []
        for match in results["matches"]:
            matched_results.append(
                {
                    "score": match["score"],
                    "category": match["metadata"]["category"],
                    "depoIQ_ID": match["metadata"].get("depoIQ_ID"),
                }
            )

        return {"query": query_text, "matches": matched_results}

    except Exception as e:
        print(f"Error querying Pinecone: {e}")
        return {"status": "error", "message": str(e)}


@app.route("/add-summary/<string:depoId>", methods=["GET"])
def add_summary(depoId):
    """
    Get Summaries from Depo and store into pinecon
    ---
    tags:
      - Summary
    parameters:
      - name: depoId
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
        # Define GraphQL query
        query = """
        query GetDepoSummary($depoId: ID!) {
          getDepo(depoId: $depoId) {
            summary {
              overview {
                title
                description
                text
              }
              highLevelSummary {
                title
                description
                text
              }
              detailedSummary {
                title
                description
                text
              }
              topicalSummary {
                title
                description
                text
              }
              visualization {
                title
                description
                text
              }
            }
          }
        }
        """

        # Set request payload
        payload = {"query": query, "variables": {"depoId": depoId}}

        token = generate_token()

        # Set headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": token,
        }

        # Make request to GraphQL API
        response = requests.post(GRAPHQL_URL, json=payload, headers=headers)

        # Handle response
        if response.status_code == 200:
            response = store_depositions_in_pinecone(
                response.json()["data"]["getDepo"], depoId
            )
            print(response)
            # return jsonify(response.json()["data"]["getDepo"]), 200
            return jsonify(response), 200
        elif response.status_code == 401:
            return jsonify({"error": "Unauthorized - Invalid Token"}), 401
        else:
            return (
                jsonify({"error": f"Failed to fetch depo data: {response.text}"}),
                response.status_code,
            )

    except Exception as e:
        return jsonify({"error": "Something went wrong", "details": str(e)}), 500


@app.route("/talk-summary", methods=["POST"])
def talk_summary():
    """
    Talk to summaries by depo_id
    ---
    tags:
      - Summary
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            depo_id:
              type: string
            user_query:
              type: string
    responses:
      200:
        description: Returns the success message
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid request, JSON body required"}), 400

        depo_id = data.get("depo_id")
        user_query = data.get("user_query")

        if not depo_id or not user_query:
            return jsonify({"error": "Missing depo_id or user_query"}), 400

        response = query_pinecone(user_query, depo_id, top_k=3)

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": "Something went wrong", "details": str(e)}), 500


@app.route("/items", methods=["GET"])
def get_items():
    """Get all items
    ---
    tags:
      - Items
    responses:
      200:
        description: List of all items
    """
    return jsonify(items)


@app.route("/items/<int:item_id>", methods=["GET"])
def get_item(item_id):
    """Get item by ID
    ---
    tags:
      - Items
    parameters:
      - name: item_id
        in: path
        type: integer
        required: true
    responses:
      200:
        description: Returns the requested item
    """
    item = next((item for item in items if item["id"] == item_id), None)
    return jsonify(item) if item else (jsonify({"error": "Item not found"}), 404)


@app.route("/items", methods=["POST"])
def add_item():
    """Add a new item
    ---
    tags:
      - Items
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            name:
              type: string
            description:
              type: string
    responses:
      201:
        description: Item added successfully
    """
    data = request.json
    new_item = {
        "id": len(items) + 1,
        "name": data["name"],
        "description": data["description"],
    }
    items.append(new_item)
    return jsonify(new_item), 201


@app.route("/items/<int:item_id>", methods=["PUT"])
def update_item(item_id):
    """Update an existing item
    ---
    tags:
      - Items
    parameters:
      - name: item_id
        in: path
        type: integer
        required: true
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            name:
              type: string
            description:
              type: string
    responses:
      200:
        description: Item updated successfully
    """
    item = next((item for item in items if item["id"] == item_id), None)
    if not item:
        return jsonify({"error": "Item not found"}), 404

    data = request.json
    item["name"] = data["name"]
    item["description"] = data["description"]
    return jsonify(item)


@app.route("/items/<int:item_id>", methods=["DELETE"])
def delete_item(item_id):
    """Delete an item
    ---
    tags:
      - Items
    parameters:
      - name: item_id
        in: path
        type: integer
        required: true
    responses:
      200:
        description: Item deleted successfully
    """
    global items
    items = [item for item in items if item["id"] != item_id]
    return jsonify({"message": "Item deleted successfully"})


if __name__ == "__main__":
    app.run(debug=True)
