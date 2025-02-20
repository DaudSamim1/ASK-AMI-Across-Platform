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
        print(f"Error generating embedding: {e}")
        return np.zeros(1536).tolist()  # Return a zero vector if embedding fails


def camel_to_snake(name):
    snake_case = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()

    # Check if '_summary' already exists at the end
    if not snake_case.endswith("_summary"):
        snake_case += "_summary"

    return snake_case


# Function to store summary data in Pinecone
def store_summaries_in_pinecone(summary_data, depo_id):
    try:
        print(f"🔹 Starting Pinecone insertion for depoIQ_ID: {depo_id}")

        # Initialize Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)

        # Define index name
        index_name = "summaries-index"

        # Create index if it doesn't exist
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=1536,  # OpenAI embedding dimension
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

        # Wait until the index is ready
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)

        index = pc.Index(index_name)

        # Check if summary already exists
        dummy_vector = np.zeros(1536).tolist()
        existing_entries = index.query(
            vector=dummy_vector, filter={"depoIQ_ID": depo_id}, top_k=1
        )

        if existing_entries["matches"]:
            print(f"⚠️ Summary {depo_id} already exists in Pinecone. Skipping insert.")
            return {
                "status": "skipped",
                "message": f"Summary with depoIQ_ID {depo_id} already exists in Pinecone. No new data inserted.",
            }

        # Insert summaries into Pinecone
        inserted_count = 0
        vectors_to_upsert = []
        inserted_categories = []

        for key, value in summary_data.items():
            text = value["text"]
            embedding = generate_embedding(text)
            # Convert camelCase keys to snake_case
            category = camel_to_snake(key)

            metadata = {
                "depoIQ_ID": depo_id,
                "category": category,
                "summary_ID": f"{key}__{depo_id}",
            }

            vectors_to_upsert.append(
                {
                    "id": str(uuid.uuid4()),  # Unique ID
                    "values": embedding,  # Embedding vector
                    "metadata": metadata,  # Metadata
                }
            )
            inserted_categories.append(category)  # Keep track of inserted categories

        # Bulk upsert into Pinecone
        if vectors_to_upsert:
            index.upsert(vectors=vectors_to_upsert)
            inserted_count = len(vectors_to_upsert)
            print(f"✅ Successfully inserted {inserted_count} summaries into Pinecone.")

        return {
            "status": "success",
            "message": f"Successfully stored {inserted_count} summaries in Pinecone for depoIQ_ID {depo_id}.",
            "data": {
                "total_inserted": inserted_count,
                "depoIQ_ID": depo_id,
                "categories": inserted_categories,
            },
        }

    except Exception as e:
        print(f"❌ Error inserting into Pinecone: {e}")
        return {
            "status": "error",
            "message": f"An error occurred while storing summaries data: {str(e)}",
        }


# 🏠 Home Endpoint
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Welcome to the Python Project API!"})


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

        # Initialize Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index_name = "summaries-index"
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
                    # "score": match["score"],
                    "category": match["metadata"]["category"],
                    "depoIQ_ID": match["metadata"].get("depoIQ_ID"),
                    "summary_ID": match["metadata"].get("summary_ID"),
                }
            )

        depoSummary = getDepoSummary(depo_id)  # Fetch depo summary to get text

        for i in range(
            len(matched_results)
        ):  # Add text to matched results from depo summary
            key = matched_results[i]["summary_ID"].split("__")[0]
            if key in depoSummary:
                text_from_AI = generate_nearest_query(
                    query_text, depoSummary[key]["text"]
                )
                matched_results[i]["content"] = depoSummary[key]["text"]
                matched_results[i]["query_answer"] = text_from_AI
                del matched_results[i]["summary_ID"]
            else:
                matched_results[i]["text"] = "No text found"

        response = {"query": query_text, "matches": matched_results}

        prompt = f"""
                  You are an AI assistant that organizes and sorts JSON data efficiently.  
                  Your task is to **analyze and sort the given data** based on relevance to the provided query.  

                  ---

                  ### **Given Data:**
                  {response}

                  ---

                  ### **Instructions:**
                  - You will receive a **JSON payload** containing:
                    - A **query**
                    - Multiple **matches**, each with a `query_answer` field.
                  - **Your goal is to determine which `query_answer` is the best fit** for `query` and **sort the results accordingly**.
                  - **Sort the results based on relevance**, ensuring the most accurate and precise response appears **first**.
                  - **If two answers have similar relevance, prioritize the one with more detailed information.**
                  - **DO NOT modify the structure of the JSON**.
                  - **DO NOT change or analyze `query_answer` text**—only rank them based on relevance.

                  ---

                  ### **Sorting Criteria:**
                  - **Highest Relevance:** The `query_answer` that directly and accurately responds to `query` should appear at the top.
                  - **Medium Relevance:** Responses that partially answer the query but may lack specifics should be placed lower.
                  - **Lowest Relevance:** If `query_answer` is vague, indirect, or missing critical details, place it at the bottom.
                  - **If relevance is equal, prioritize the response that provides more details and context.**

                  ---

                  ### **Additional Formatting Rules:**
                  - **Do NOT modify the JSON structure.** Return the same format as received.
                  - **Ensure the JSON format remains unchanged**—simply reordering based on relevance.
                  - **Retain all existing fields** without modification.
                  """

        client = OpenAI(api_key=OPENAI_API_KEY)
        ai_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI assistant that extracts precise answers from multiple given documents efficiently.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=1000,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "query_response",
                    "description": "A structured response for a query and its matched answers.",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "matches": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "category": {"type": "string"},
                                        "depoIQ_ID": {"type": "string"},
                                        "query_answer": {"type": "string"},
                                        "one_word_answer": {"type": "string"},
                                    },
                                    "required": [
                                        "category",
                                        "depoIQ_ID",
                                        "query_answer",
                                        "one_word_answer",
                                    ],
                                },
                            },
                        },
                        "required": ["query", "matches"],
                    },
                },
            },
        )

        # Convert response to JSON
        ai_json_response = json.loads(ai_response.choices[0].message.content.strip())

        # Print formatted output
        print(json.dumps(ai_json_response, indent=4))

        return json.dumps(ai_json_response, indent=4)

    except Exception as e:
        print(f"Error querying Pinecone: {e}")
        return {"status": "error", "message": str(e)}


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


# Function to query Pinecone and match summaries
def query_topical_pinecone(query_text, depo_id=None, top_k=3):
    try:
        print(f"Querying Pinecone for: {query_text} and depo_id: {depo_id}")

        # Initialize Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index_name = "summaries-index-sub-categories"
        index = pc.Index(index_name)

        # Generate query embedding
        query_vector = generate_embedding(query_text)

        # Define filter criteria (search within `depo_id`)
        filter_criteria = {
            "category": camel_to_snake("topicalSummary"),
            "depoIQ_ID": depo_id,
        }

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
            # if match["score"] < 0.5:
            #     continue
            matched_results.append(
                {
                    "category": match["metadata"]["category"],
                    "depoIQ_ID": match["metadata"].get("depoIQ_ID"),
                    "summary_ID": match["metadata"].get("summary_ID"),
                    "sub_category": match["metadata"].get("sub_category", None),
                }
            )

        depoSummary = getDepoSummary(depo_id)  # Fetch depo summary to get text
        topicalSummary = depoSummary["topicalSummary"]["text"]
        topical_summary_object_data = {}
        # Parse the HTML using BeautifulSoup
        soup = BeautifulSoup(topicalSummary, "html.parser")

        # Iterate over each list item
        for li in soup.find_all("li"):
            title = li.find("h4")
            value = li.find("p")
            if title and value:
                topical_summary_object_data[title.text] = value.text

        for i in range(
            len(matched_results)
        ):  # Add text to matched results from depo summary
            key = matched_results[i]["summary_ID"].split("__")[0]
            if key in depoSummary:
                sub_category = matched_results[i]["sub_category"]
                if sub_category:
                    [keyType, title_Key] = sub_category.split("__")
                    isIndex = keyType == "index"
                    print(isIndex, "isIndex", title_Key, "title_Key", sub_category)
                else:
                    print("sub_category not found")
                if title_Key in topical_summary_object_data:
                    text_from_AI = generate_nearest_query(
                        query_text, topical_summary_object_data[title_Key]
                    )
                    matched_results[i]["content"] = topical_summary_object_data[
                        title_Key
                    ]
                else:
                    text_from_AI = generate_nearest_query(
                        query_text, depoSummary[key]["text"]
                    )
                    matched_results[i]["content"] = depoSummary[key]["text"]
                matched_results[i]["query_answer"] = text_from_AI
                del matched_results[i]["summary_ID"]
                del matched_results[i]["sub_category"]
            else:
                matched_results[i]["text"] = "No text found"

        response = {"query": query_text, "matches": matched_results}

        prompt = f"""
                  You are an AI assistant that organizes and sorts JSON data efficiently.
                  Your task is to **analyze and sort the given data** based on relevance to the provided query.

                  ---

                  ### **Given Data:**
                  {response}

                  ---

                  ### **Instructions:**
                  - You will receive a **JSON payload** containing:
                    - A **query**
                    - Multiple **matches**, each with a `query_answer` field.
                  - **Your goal is to determine which `query_answer` is the best fit** for `query` and **sort the results accordingly**.
                  - **Sort the results based on relevance**, ensuring the most accurate and precise response appears **first**.
                  - **If two answers have similar relevance, prioritize the one with more detailed information.**
                  - **DO NOT modify the structure of the JSON**.
                  - **DO NOT change or analyze `query_answer` text**—only rank them based on relevance.

                  ---

                  ### **Sorting Criteria:**
                  - **Highest Relevance:** The `query_answer` that directly and accurately responds to `query` should appear at the top.
                  - **Medium Relevance:** Responses that partially answer the query but may lack specifics should be placed lower.
                  - **Lowest Relevance:** If `query_answer` is vague, indirect, or missing critical details, place it at the bottom.
                  - **If relevance is equal, prioritize the response that provides more details and context.**

                  ---

                  ## **Important Note:**
                  - Show only the **three most relevant** results.
                  - The most relevant response should appear **first**, followed by the second most relevant, and then the third.
                  - Ensure that only these **top three** results are included in the final output.

                  ---

                  ### **Additional Formatting Rules:**
                  - **Do NOT modify the JSON structure.** Return the same format as received.
                  - **Ensure the JSON format remains unchanged**—simply reordering based on relevance.
                  - **Retain all existing fields** without modification.
                  """

        client = OpenAI(api_key=OPENAI_API_KEY)
        ai_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI assistant that extracts precise answers from multiple given documents efficiently.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=1000,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "query_response",
                    "description": "A structured response for a query and its matched answers.",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "matches": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "category": {"type": "string"},
                                        "depoIQ_ID": {"type": "string"},
                                        "query_answer": {"type": "string"},
                                        "one_word_answer": {"type": "string"},
                                    },
                                    "required": [
                                        "category",
                                        "depoIQ_ID",
                                        "query_answer",
                                        "one_word_answer",
                                    ],
                                },
                            },
                        },
                        "required": ["query", "matches"],
                    },
                },
            },
        )

        # Convert response to JSON
        ai_json_response = json.loads(ai_response.choices[0].message.content.strip())

        # Print formatted output
        print(json.dumps(ai_json_response, indent=4))

        return json.dumps(ai_json_response, indent=4)
        # return jsonify(response)

    except Exception as e:
        print(f"Error querying Pinecone: {e}")
        return {"status": "error", "message": str(e)}


@app.route("/add-summaries/<string:depoIQ_ID>", methods=["GET"])
def add_summary(depoIQ_ID):
    """
    Get Summaries from Depo and store into pinecone
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
        response = store_summaries_in_pinecone(depo_summary, depoIQ_ID)
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


@app.route("/add-topical-summary/<string:depoIQ_ID>", methods=["GET"])
def add_topical_summary(depoIQ_ID):
    """
    Get Topical Summary from Depo and store into pinecone
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
        topical_summary = depo_summary["topicalSummary"]["text"]

        # Parse the HTML using BeautifulSoup
        soup = BeautifulSoup(topical_summary, "html.parser")

        # # Initialize Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)

        # # Define index name
        index_name = "summaries-index-sub-categories"

        # Create index if it doesn't exist
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=1536,  # OpenAI embedding dimension
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        # Wait until the index is ready
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)

        index = pc.Index(index_name)

        # Wait until the index is ready
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)

        index = pc.Index(index_name)

        # Create dictionary to store extracted data
        data_dict = {}
        vectors_to_upsert = []
        inserted_sub_categories = []

        # Iterate over each list item
        for li in soup.find_all("li"):
            title = li.find("h4")
            value = li.find("p")
            if title and value:
                data_dict[title.text] = value.text
                # store it pinecone here and add sub_category in metadata

                # Convert camelCase keys to snake_case
                category = camel_to_snake("topicalSummary")

                # Check if summary already exists
                dummy_vector = np.zeros(1536).tolist()
                existing_entries = index.query(
                    vector=dummy_vector,
                    filter={"depoIQ_ID": depoIQ_ID, "category": category},
                    top_k=1,
                )
                print(existing_entries, "existing_entries")

                if existing_entries["matches"]:
                    return {
                        "status": "skipped",
                        "message": f"Summary with depoIQ_ID {depoIQ_ID} already exists in Pinecone. No new data inserted.",
                    }

                text = title.text + "\n" + value.text
                print(
                    f"\n\n\n the text of {title.text} is {text} \n\n and storing value for pinecone \n\n\n"
                )
                embedding = generate_embedding(text)

                metadata = {
                    "depoIQ_ID": depoIQ_ID,
                    "category": category,
                    "sub_category": f"key__{title.text}",
                    "summary_ID": f"topicalSummary__{depoIQ_ID}",
                }

                vectors_to_upsert.append(
                    {
                        "id": str(uuid.uuid4()),  # Unique ID
                        "values": embedding,  # Embedding vector
                        "metadata": metadata,  # Metadata
                    }
                )
                inserted_sub_categories.append(
                    f"key__{title.text}"
                )  # Keep track of inserted categories

        # Bulk upsert into Pinecone
        if vectors_to_upsert:
            index.upsert(vectors=vectors_to_upsert)
            inserted_count = len(vectors_to_upsert)
            print(f"✅ Successfully inserted {inserted_count} summaries into Pinecone.")

        response = {
            "status": "success",
            "message": f"Successfully stored topicalSummary {inserted_count} chhunks summaries in Pinecone for depoIQ_ID {depoIQ_ID}.",
            "data": {
                "total_inserted": inserted_count,
                "depoIQ_ID": depoIQ_ID,
                "category": "topicalSummary",
                "sub_categories": inserted_sub_categories,
            },
        }

        # response = store_summaries_in_pinecone(depo_summary, depoIQ_ID)
        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": "Something went wrong", "details": str(e)}), 500


@app.route("/talk-topical-summary", methods=["POST"])
def talk_topical_summary():
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

        response = query_topical_pinecone(user_query, depo_id, top_k=8)

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
