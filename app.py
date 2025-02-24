from flask import Flask, jsonify, request
from flasgger import Swagger
import requests
import jwt
import time
import uuid
import numpy as np
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv
import re
import json
from bs4 import BeautifulSoup

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
        print(f"Error generating embedding: {e}\n\n\n\n text is -- {text}")
        return np.zeros(1536).tolist()


def camel_to_snake(name):
    snake_case = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()

    # Check if '_summary' already exists at the end
    if not snake_case.endswith("_summary"):
        snake_case += "_summary"

    return snake_case


# Function to generate a nearest relevant query based on the given reference text and user query.
def generate_nearest_query(user_query, reference_text):
    prompt = f"""
    You are an AI assistant that extracts precise answers from a given document.
    Your task is to provide the most accurate and structured response based on the context.

    Context:
    "{reference_text}"

    User Question:
    "{user_query}"

    - If the exact answer is found, provide it concisely.
    - If relevant information is available but not an exact match, summarize the key points.
    - If there is no clear answer, state "The document does not provide specific details, but it discusses related topics such as [mention relevant details]."
    - Do NOT simply return "No relevant details were provided" unless the document is completely unrelated.

    Provide your response below:
    """

    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4",  # Use GPT-4 or another available model
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI assistant that refines queries based on a reference paragraph.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=100,
        )

        # Correct way to access the response
        refined_query = response.choices[0].message.content.strip()
        print(f"Refined query: {refined_query}")
        return refined_query

    except Exception as e:
        return f"Error generating query: {str(e)}"


# Function to query Pinecone and match summaries
def query_pinecone(query_text, depo_id=None, top_k=3):
    try:
        print(f"Querying Pinecone for: {query_text} and depo_id: {depo_id}")

        # Generate query embedding
        query_vector = generate_embedding(query_text)

        # Define filter criteria (search within `depo_id`)
        filter_criteria = {}
        if depo_id:
            # Add depo_id filter
            filter_criteria["depoIQ_ID"] = depo_id

        # Search in Pinecone
        results = summariesIndex.query(
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
                    # "score": match["score"],
                    "category": match["metadata"]["category"],
                    "depoIQ_ID": match["metadata"].get("depoIQ_ID"),
                    "sub_category": match["metadata"].get("sub_category", None),
                    "text": match["metadata"].get("text", "No text found"),
                }
            )

        # depoSummary = getDepoSummary(depo_id)  # Fetch depo summary to get text

        # Add text to matched results from depo summary
        # for i in range(len(matched_results)):
        #     text_from_AI = generate_nearest_query(
        #         query_text, matched_results[i]["text"]
        #     )
        #     matched_results[i]["query_answer"] = text_from_AI

        response = {
            "answer_for_query": (
                matched_results[0]["text"] if matched_results else "No answer found"
            ),
            "user_query": query_text,
            "metadata": matched_results,
        }

        return json.dumps(response, indent=4)

        # prompt = f"""
        #           You are an AI assistant that organizes and sorts JSON data efficiently.
        #           Your task is to **analyze and sort the given data** based on relevance to the provided query.

        #           ---

        #           ### **Given Data:**
        #           {response}

        #           ---

        #           ### **Instructions:**
        #           - You will receive a **JSON payload** containing:
        #             - A **query**
        #             - Multiple **matches**, each with a `query_answer` field.
        #           - **Your goal is to determine which `query_answer` is the best fit** for `query` and **sort the results accordingly**.
        #           - **Sort the results based on relevance**, ensuring the most accurate and precise response appears **first**.
        #           - **If two answers have similar relevance, prioritize the one with more detailed information.**
        #           - **DO NOT modify the structure of the JSON**.
        #           - **DO NOT change or analyze `query_answer` text**—only rank them based on relevance.

        #           ---

        #           ### **Sorting Criteria:**
        #           - **Highest Relevance:** The `query_answer` that directly and accurately responds to `query` should appear at the top.
        #           - **Medium Relevance:** Responses that partially answer the query but may lack specifics should be placed lower.
        #           - **Lowest Relevance:** If `query_answer` is vague, indirect, or missing critical details, place it at the bottom.
        #           - **If relevance is equal, prioritize the response that provides more details and context.**

        #           ---

        #           ### **Additional Formatting Rules:**
        #           - **Do NOT modify the JSON structure.** Return the same format as received.
        #           - **Ensure the JSON format remains unchanged**—simply reordering based on relevance.
        #           - **Retain all existing fields** without modification.
        #           """

        # client = OpenAI(api_key=OPENAI_API_KEY)
        # ai_response = client.chat.completions.create(
        #     model="gpt-4o-mini",
        #     messages=[
        #         {
        #             "role": "system",
        #             "content": "You are an AI assistant that extracts precise answers from multiple given documents efficiently.",
        #         },
        #         {"role": "user", "content": prompt},
        #     ],
        #     max_tokens=1000,
        #     response_format={
        #         "type": "json_schema",
        #         "json_schema": {
        #             "name": "query_response",
        #             "description": "A structured response for a query and its matched answers.",
        #             "schema": {
        #                 "type": "object",
        #                 "properties": {
        #                     "query": {"type": "string"},
        #                     "matches": {
        #                         "type": "array",
        #                         "items": {
        #                             "type": "object",
        #                             "properties": {
        #                                 "category": {"type": "string"},
        #                                 "sub_category": {"type": "string"},
        #                                 "depoIQ_ID": {"type": "string"},
        #                                 "query_answer": {"type": "string"},
        #                             },
        #                             "required": [
        #                                 "category",
        #                                 "sub_category",
        #                                 "depoIQ_ID",
        #                                 "query_answer",
        #                             ],
        #                         },
        #                     },
        #                 },
        #                 "required": ["query", "matches"],
        #             },
        #         },
        #     },
        # )

        # # Convert response to JSON
        # ai_json_response = json.loads(ai_response.choices[0].message.content.strip())

        # # Print formatted output
        # print(json.dumps(ai_json_response, indent=4))

        # return json.dumps(ai_json_response, indent=4)

    except Exception as e:
        print(f"Error querying Pinecone: {e}")
        return {"status": "error", "message": str(e)}


# Function to generate a nearest relevant query based on the given reference text and user query.
def getDepoSummary(depoIQ_ID):
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

    payload = {"query": query, "variables": {"depoId": depoIQ_ID}}

    token = generate_token()

    headers = {
        "Content-Type": "application/json",
        "Authorization": token,
    }

    response = requests.post(GRAPHQL_URL, json=payload, headers=headers)

    if response.status_code == 200:
        return response.json()["data"]["getDepo"]["summary"]
    elif response.status_code == 401:
        raise Exception("Unauthorized - Invalid Token")
    else:
        raise Exception(f"Failed to fetch depo data: {response.text}")


# Initialize Pinecone and create the index if it doesn't exist
def initialize_pinecone_index(index_name):
    """Initialize Pinecone and create the index if it doesn't exist."""
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # List existing indexes
    existing_indexes = pc.list_indexes().names()

    if index_name not in existing_indexes:
        print(f"🔍 Index '{index_name}' not found. Creating it now...")

        pc.create_index(
            name=index_name,
            dimension=1536,  # OpenAI embedding dimension
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

        # Wait until the index is ready
        while not pc.describe_index(index_name).status["ready"]:
            print("⏳ Waiting for index to be ready...")
            time.sleep(2)

        print(f"✅ Index '{index_name}' created and ready.")
    else:
        print(f"✅ Index '{index_name}' already exists.")

    return pc.Index(index_name)


# Initialize Pinecone index
summariesIndex = initialize_pinecone_index("summaries-index")


# Function to convert camelCase to snake_case
def extract_text(value):
    """Extracts plain text from HTML or returns simple text."""
    if value.startswith("<"):
        soup = BeautifulSoup(value, "html.parser")
        return soup.get_text()
    return value


# Function to split text into paragraphs
def split_into_chunks(text):
    """Splits text into paragraphs while removing empty strings."""
    return [chunk.strip() for chunk in re.split("\n\n", text.strip()) if chunk.strip()]


# Function to check if a summary already exists in Pinecone
def check_existing_entry(depoIQ_ID, category, sub_category):
    """Checks if a summary already exists in Pinecone."""
    try:
        existing_entries = summariesIndex.query(
            vector=np.random.rand(1536).tolist(),  # Use a dummy query vector
            filter={
                "depoIQ_ID": depoIQ_ID,
                "sub_category": sub_category,
                "category": category,
            },
            top_k=1,
        )
        return bool(existing_entries["matches"])
    except Exception as e:
        print(f"⚠️ Error querying Pinecone: {e}")
        return False


# Function to store summaries in Pinecone with embeddings
def store_summaries_in_pinecone(depoIQ_ID, category, text_chunks):
    """Stores text chunks in Pinecone with embeddings."""
    vectors_to_upsert = []
    skipped_chunks = []

    for chunk_index, chunk_value in enumerate(text_chunks):
        sub_category = f"index__{chunk_index}__{category}"

        print(f"🔹 Processing chunk {chunk_index + 1} of {category}")

        if not chunk_value or not isinstance(chunk_value, str):
            print(f"⚠️ Skipping invalid text chunk: {sub_category}")
            continue

        # Check if summary already exists
        if check_existing_entry(depoIQ_ID, category, sub_category):
            skipped_chunks.append(sub_category)
            print(f"⚠️ Summary already exists: {sub_category}, skipping...")
            continue

        # Generate embedding
        embedding = generate_embedding(chunk_value)

        if not any(embedding):  # Check for all zero vectors
            print(f"⚠️ Skipping zero-vector embedding for: {sub_category}")
            continue

        # Metadata
        metadata = {
            "depoIQ_ID": depoIQ_ID,
            "category": category,
            "sub_category": sub_category,
            "text": chunk_value,
        }

        # Add to batch
        vectors_to_upsert.append(
            {
                "id": str(uuid.uuid4()),  # Unique ID
                "values": embedding,  # Embedding vector
                "metadata": metadata,  # Metadata
            }
        )

    # Bulk upsert to Pinecone
    if vectors_to_upsert:

        summariesIndex.upsert(vectors=vectors_to_upsert)
        print(
            f"✅ Successfully inserted {len(vectors_to_upsert)} summaries in Pinecone."
        )

    return len(vectors_to_upsert), skipped_chunks


# 🏠 Home Endpoint for testing
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Welcome to the Python Project API!"})


@app.route("/add-summaries/<string:depoIQ_ID>", methods=["GET"])
def add_summary(depoIQ_ID):
    """
    Get Summaries from Depo and store into Pinecone
    ---
    tags:
      - Summary
    parameters:
      - name: depoIQ_ID
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
        depo_summary = getDepoSummary(depoIQ_ID)
        excluded_keys = ["visualization"]
        total_inserted = 0
        skipped_sub_categories = []

        for key, value in depo_summary.items():
            if key in excluded_keys:
                continue

            category = camel_to_snake(key)  # Convert key to category format
            text = extract_text(value["text"])  # Extract clean text
            text_chunks = split_into_chunks(text)  # Split into paragraphs

            inserted_count, skipped_chunks = store_summaries_in_pinecone(
                depoIQ_ID, category, text_chunks
            )

            total_inserted += inserted_count
            skipped_sub_categories.extend(skipped_chunks)

        # Response
        response = {
            "status": "success",
            "message": f"Stored {total_inserted} summaries in Pinecone for depoIQ_ID {depoIQ_ID}.",
            "data": {
                "total_inserted": total_inserted,
                "depoIQ_ID": depoIQ_ID,
                "skipped_sub_categories": skipped_sub_categories,
                "skipped_count": len(skipped_sub_categories),
            },
        }

        return jsonify(response), 200

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

        return response, 200

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
