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
from bson import ObjectId

# Load environment variables from .env file
load_dotenv()

# Retrieve API keys from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
GRAPHQL_URL = os.getenv("GRAPHQL_URL")
DEPO_INDEX_NAME = os.getenv("DEPO_INDEX_NAME", "depo-index")

# Ensure keys are loaded
if not OPENAI_API_KEY or not PINECONE_API_KEY:
    raise ValueError("Missing API Keys! Check your .env file.")


app = Flask(__name__)
swagger = Swagger(app)


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


# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)


# Initialize Pinecone index
depoIndex = initialize_pinecone_index(DEPO_INDEX_NAME)


def generate_token(
    user_id="677ffbb1f728963ffa7b8dca",
    company_id="66ff06f4c50afa83deecb020",
    isSuperAdmin=False,
    role="ADMIN",
):
    # Generate access token
    access_token_payload = {
        "userId": user_id,
        "companyId": company_id,
        "isSuperAdmin": isSuperAdmin,
        "role": role,
    }

    # generate access token
    accessToken = "Bearer " + jwt.encode(
        access_token_payload, JWT_SECRET_KEY, algorithm="HS256"
    )

    return accessToken


# Function to generate embeddings from OpenAI
def generate_embedding(text):
    try:
        text = text.replace("\n", " ").replace("\r", " ")
        response = openai_client.embeddings.create(
            input=[text], model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return False


# Function to convert camelCase to snake_case
def camel_to_snake(name):
    snake_case = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
    if not snake_case.endswith("_summary"):
        snake_case += "_summary"
    return snake_case


# snake to camel case and also remove _summary
def snake_to_camel(name, isSummary=True):
    if isSummary:
        name = name.replace("_summary", "")
    parts = name.split("_")
    camel_case = parts[0] + "".join(word.capitalize() for word in parts[1:])
    return camel_case


# Function to group by depoIQ_ID for Pinecone results
def group_by_depoIQ_ID(data, isSummary=False, isTranscript=False):
    grouped_data = {}
    for entry in data:
        depoIQ_ID = entry["depoIQ_ID"]
        if depoIQ_ID not in grouped_data:
            print(f"🔍 Fetching Depo: {depoIQ_ID} fro DB\n\n\n")

            if isTranscript and not isSummary:
                grouped_data[depoIQ_ID] = getDepoTranscript(depoIQ_ID)
                return grouped_data

            if isSummary and not isTranscript:
                grouped_data[depoIQ_ID] = getDepoSummary(depoIQ_ID)
                return grouped_data

            grouped_data[depoIQ_ID] = getDepo(
                depoIQ_ID, isSummary=True, isTranscript=True
            )
    print("Grouped Data fetched from DB \n\n")
    return grouped_data


# Function to query Pinecone and match return top k results
def query_pinecone(query_text, depo_ids=[], top_k=5, is_unique=False):
    try:
        print(
            f"\n🔍 Querying Pinecone for : query_text-> '{query_text}' (depo_id->: {depo_ids})\n\n\n"
        )

        # Generate query embedding
        query_vector = generate_embedding(query_text)

        if not query_vector:
            return json.dumps(
                {"status": "error", "message": "Failed to generate query embedding."}
            )

        # Define filter criteria (search within `depo_id`)
        transcript_category = "transcript"
        filter_criteria = {}
        if depo_ids and len(depo_ids) > 0:
            # Add depo_id filter
            filter_criteria["depoIQ_ID"] = {"$in": depo_ids}

        # Search in Pinecone
        results = depoIndex.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            filter=filter_criteria,
        )

        # Check if any matches found
        if not results["matches"] or not any(results["matches"]):
            print("\n\n No matches found. in depo pinecone \n\n\n")
            return json.dumps({"status": "Not Found", "message": "No matches found."})

        # Extract matched results
        matched_results = [
            {
                "depoIQ_ID": match["metadata"].get("depoIQ_ID"),
                "category": match["metadata"]["category"],
                "chunk_index": match["metadata"].get("chunk_index", None),
            }
            for match in results.get("matches", [])
        ]

        print(f"Matched Results: {matched_results} \n\n\n")

        # Group results by depoIQ_ID
        grouped_result = group_by_depoIQ_ID(
            matched_results, isSummary=True, isTranscript=True
        )

        # print(f"Grouped Results: {grouped_result} \n\n\n")

        custom_response = []  # Custom response

        for match in matched_results:
            depoIQ_ID = match["depoIQ_ID"]
            category = match["category"]
            isSummary = category != transcript_category
            isTranscript = category == transcript_category
            depo_data = grouped_result.get(depoIQ_ID, {})

            if isSummary:
                chunk_index = int(match["chunk_index"])
                db_category = snake_to_camel(category, isSummary=False)
                db_category_2 = snake_to_camel(category)
                summaries = depo_data.get("summary", {})

                db_text = (
                    summaries.get(db_category)
                    if db_category in summaries
                    else summaries.get(db_category_2, "No data found - summary")
                )
                text = extract_text(db_text["text"])
                text_chunks = split_into_chunks(text)
                text = (
                    text_chunks[chunk_index] if chunk_index < len(text_chunks) else text
                )
                custom_response.append(
                    {
                        "depoIQ_ID": depoIQ_ID,
                        "category": category,
                        "chunk_index": chunk_index,
                        "text": text,
                    }
                )

            if isTranscript:
                chunk_index = match["chunk_index"]
                start_page, end_page = map(int, chunk_index.split("-"))
                transcripts = depo_data.get("transcript", {})
                pages = [
                    page
                    for page in transcripts
                    if start_page <= page["pageNumber"] <= end_page
                ]
                custom_response.append(
                    {
                        "depoIQ_ID": depoIQ_ID,
                        "category": category,
                        "chunk_index": chunk_index,
                        "text": " ".join(
                            [
                                line["lineText"].strip()
                                for page in pages
                                for line in page.get("lines", [])
                                if line["lineText"].strip()
                            ]
                        ),
                    }
                )

        text_list = [entry["text"] for entry in custom_response]
        # print(f"\n\n\n\n\nText List: {text_list} \n\n\n\n\n")

        # Instantiate the Pinecone client
        pc = Pinecone(api_key=PINECONE_API_KEY)

        from pinecone import RerankModel

        # Perform reranking to get top_n results based on the query
        reranked_results = pc.inference.rerank(
            model=RerankModel.Bge_Reranker_V2_M3,
            query=query_text,
            documents=text_list,
            top_n=top_k if is_unique else 8,
            return_documents=True,
        )

        rerank_response = []
        for rank_match in reranked_results.data:
            index = rank_match.index
            response = custom_response[index]
            rerank_response.append(response)

        if is_unique:
            unique_set_response = {}
            for entry in rerank_response:
                depoIQ_ID = entry["depoIQ_ID"]
                if depoIQ_ID not in unique_set_response:
                    unique_set_response[depoIQ_ID] = []
                unique_set_response[depoIQ_ID].append(entry)

            unique_set_response_keys = list(unique_set_response.keys())

            final_response = []

            # Ensure we have exactly 8 entries in final_response
            while len(final_response) < 8:
                for depoIQ_ID in unique_set_response_keys:
                    if len(final_response) >= 8:
                        break
                    try:
                        final_response.append(unique_set_response[depoIQ_ID].pop(0))
                        # if not unique_set_response[depoIQ_ID]:
                        #     unique_set_response_keys.remove(depoIQ_ID)
                    except IndexError:
                        continue

            return {
                "user_query": query_text,
                "depoIQ_ID": unique_set_response_keys,
                "metadata": final_response,
                "is_unique": is_unique,
            }
        else:

            grouped_result_keys = list(grouped_result.keys())

            response = {
                "user_query": query_text,
                "depoIQ_ID": grouped_result_keys,
                "metadata": rerank_response,
                "is_unique": is_unique,
            }

        return response

    except Exception as e:
        print(f"Error querying Pinecone: {e}")
        return {"status": "error", "message": str(e)}


def get_answer_from_AI(response):
    try:
        # Construct prompt
        prompt = f"""
                You are an AI assistant that extracts precise and relevant answers from multiple transcript sources.

                ### **Task:**
                - Analyze the provided `"metadata"` and generate a direct, relevant answer to the `"user_query"`. 
                - Use the `"text"` field in `"metadata"` to form the most accurate response.
                - Ensure the generated response directly addresses the `"user_query"` without altering the original `"text"` in `"metadata"`.
                - Reference the source by indicating which `"metadata"` entries the answer was extracted from.

                ---

                ### **Given Data:**
                {json.dumps(response)}

                ---

                ### **Instructions:**
                - Identify the most relevant `"text"` entries from `"metadata"` that best answer the `"user_query"`.
                - Use the most relevant content to construct the response, ensuring clarity and completeness.
                - **Always return the same answer for identical inputs.**
                - **Do not modify, summarize, or rewrite `"text"` in `"metadata"`—only extract the most relevant portion.**
                - **Return only the extracted answer as plain text, followed by the metadata reference in square brackets.**
                - **Do not include any quotation marks, slashes, special characters, or extra formatting.**
                - **The output should be raw text only, with no extra symbols.**
                - **If multiple metadata sources are used, format the reference as `[metadata: X and metadata: Y]`, where X and Y are the index positions of the metadata entries.**

                ---

                ### **Final Output:**
                - Return only the extracted answer as raw text.
                - Append metadata references in the format `[metadata: X]`, where X is the index position of the metadata entry.
                - If multiple metadata entries are used, list them as `[metadata: X and metadata: Y]`.
                """

        # Call GPT-3.5 API with parameters for consistency
        ai_response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI assistant that extracts precise answers from multiple transcript sources efficiently and consistently. Your answers should be the same for identical queries.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=500,
            temperature=0,
            top_p=0.1,
        )

        # Extract plain text response
        extracted_text = ai_response.choices[0].message.content.strip()

        print(f"AI Response: {extracted_text} \n\n\n\n")

        return extracted_text

    except Exception as e:
        print(f"Error querying Pinecone: {e}")
        return str(e)


# Function to get Depo from DepoIQ_ID
def getDepo(depoIQ_ID, isSummary=True, isTranscript=True):
    try:
        # GraphQL query with conditional @include directives
        query = """
        query GetDepoSummary($depoId: ID!, $includeSummary: Boolean!, $includeTranscript: Boolean!) {
            getDepo(depoId: $depoId) {
                transcript @include(if: $includeTranscript) {
                    pageNumber
                    pageNumberAtBeforeLines
                    pageText
                    lines {
                        lineNumber
                        lineText
                    }
                }
                summary @include(if: $includeSummary) {
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

        payload = {
            "query": query,
            "variables": {
                "depoId": depoIQ_ID,
                "includeSummary": isSummary,
                "includeTranscript": isTranscript,
            },
        }

        token = generate_token()

        headers = {
            "Content-Type": "application/json",
            "Authorization": token,
        }

        response = requests.post(GRAPHQL_URL, json=payload, headers=headers)

        if response.status_code == 200:
            return response.json()["data"]["getDepo"]
        elif response.status_code == 401:
            raise Exception("Unauthorized - Invalid Token")
        else:
            raise Exception(f"Failed to fetch depo data: {response.text}")
    except Exception as e:
        print(f"Error fetching depo data: {e}")
        return {}


# Function to generate to Get Summary from Depo
def getDepoSummary(depoIQ_ID):
    """Get Depo Summary from DepoIQ_ID"""
    try:
        depo = getDepo(depoIQ_ID, isSummary=True, isTranscript=False)
        return depo["summary"]
    except Exception as e:
        print(f"Error fetching depo summary: {e}")
        return {}


# Function to generate Get Transcript from Depo
def getDepoTranscript(depoIQ_ID):
    """Get Depo Transcript from DepoIQ_ID"""
    try:
        depo = getDepo(depoIQ_ID, isSummary=False, isTranscript=True)
        return depo["transcript"]
    except Exception as e:
        print(f"Error fetching depo transcript: {e}")
        return {}


# Function to convert camelCase to snake_case
def extract_text(value):
    """Extracts plain text from HTML or returns simple text."""
    if value.startswith("<"):
        soup = BeautifulSoup(value, "html.parser")
        return soup.get_text()
    return str(value)


# Function to split text into paragraphs
import re


def split_into_chunks(text, min_chunk_size=300, max_chunk_size=500):
    """Splits text into balanced chunks while keeping sentences intact."""

    # Split text into paragraphs based on double newline
    paragraphs = [p.strip() for p in re.split(r"\n\n+", text.strip()) if p.strip()]

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        # If adding this paragraph keeps the chunk within limits, add it
        if len(current_chunk) + len(para) <= max_chunk_size:
            current_chunk += " " + para
        else:
            # If chunk is already long enough, store it and start a new chunk
            if len(current_chunk) >= min_chunk_size:
                chunks.append(current_chunk.strip())
                current_chunk = para  # Start a new chunk
            else:
                # If the paragraph is too big, split it into sentences
                sentences = re.split(r"(?<=[.!?]) +", para)
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) <= max_chunk_size:
                        current_chunk += " " + sentence
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence  # Start a new chunk

    # Add the last chunk if not empty
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


# Function to check if already exists in Pinecone
def check_existing_entry(depoIQ_ID, category, chunk_index):
    """Checks if already exists in Pinecone."""
    try:
        existing_entries = depoIndex.query(
            vector=np.random.rand(1536).tolist(),  # Use a dummy query vector
            filter={
                "depoIQ_ID": depoIQ_ID,
                "chunk_index": chunk_index,
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

        print(f"🔹 Processing chunk {chunk_index + 1} of {category}")

        if not chunk_value or not isinstance(chunk_value, str):
            print(f"⚠️ Skipping invalid text chunk: {chunk_index}")
            continue

        # Check if summary already exists
        if check_existing_entry(depoIQ_ID, category, chunk_index):
            skipped_chunks.append(chunk_index)
            print(f"⚠️ Summary already exists: {chunk_index}, skipping...")
            continue

        # Generate embedding
        embedding = generate_embedding(chunk_value)

        if not any(embedding):  # Check for all zero vectors
            print(f"⚠️ Skipping zero-vector embedding for: {chunk_index}")
            continue

        # Metadata
        metadata = {
            "depoIQ_ID": depoIQ_ID,
            "category": category,
            "chunk_index": chunk_index,
            # "text": chunk_value,
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

        depoIndex.upsert(vectors=vectors_to_upsert)
        print(
            f"✅ Successfully inserted {len(vectors_to_upsert)} summaries in Pinecone."
        )

    return len(vectors_to_upsert), skipped_chunks


def store_transcript_lines_in_pinecone(depoIQ_ID, category, transcript_data):
    vectors_to_upsert = []
    skipped_chunks = []
    chunk_page_range = 3

    for i in range(0, len(transcript_data), chunk_page_range):
        grouped_pages = transcript_data[i : i + chunk_page_range]

        chunk_index = (
            f"{grouped_pages[0]['pageNumber']}-{grouped_pages[-1]['pageNumber']}"
        )

        # check if already exists in pinecone
        if check_existing_entry(
            depoIQ_ID,
            category,
            chunk_index,
        ):
            skipped_chunks.append(
                "Pages " + str(i + 1) + "-" + str(i + chunk_page_range)
            )
            print(
                f"⚠️ Transcript already exists for pages {i+1}-{i+chunk_page_range}, skipping..."
            )
            continue

        # Extract transcript lines, ensuring each line is properly formatted
        raw_lines = [
            line["lineText"].strip()  # Ensure newline preservation
            for page in grouped_pages
            for line in page.get("lines", [])
            if line["lineText"].strip()  # Remove empty or whitespace-only lines
        ]

        # Ensure text formatting is preserved for embeddings
        trimmed_text = " ".join(raw_lines).strip()  # Preserve proper line structure

        if not trimmed_text:
            print(f"⚠️ Skipping empty text chunk for pages {i+1}-{i+chunk_page_range}")
            continue

        print(
            f"\n\n🔹 Storing text chunk (Pages {grouped_pages[0]['pageNumber']} - {grouped_pages[-1]['pageNumber']})\n\n"
        )

        # Generate embedding
        embedding = generate_embedding(trimmed_text)

        # Debug: Check if embedding is valid
        if not any(embedding):
            print(f"⚠️ Skipping empty embedding for pages {i+1}-{i+chunk_page_range}")
            continue

        # Add the actual transcript text in metadata for retrieval
        metadata = {
            "depoIQ_ID": depoIQ_ID,
            "category": category,
            "chunk_index": chunk_index,
        }

        vectors_to_upsert.append(
            {
                "id": str(uuid.uuid4()),  # Unique ID
                "values": embedding,  # Embedding vector
                "metadata": metadata,  # Metadata including the text
            }
        )

    if vectors_to_upsert:
        depoIndex.upsert(vectors=vectors_to_upsert)
        print(
            f"✅ Successfully inserted {len(vectors_to_upsert)} transcript chunks in Pinecone."
        )

    return len(vectors_to_upsert), skipped_chunks


# Function to generate add depo summaries to pinecone
def add_depo_summaries(depo_summary, depoIQ_ID):
    try:
        excluded_keys = ["visualization"]
        total_inserted = 0
        skipped_sub_categories = {}

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
            skipped_sub_categories[category] = skipped_chunks

        # Response
        response = {
            "status": "success",
            "message": f"Stored {total_inserted} summaries in Pinecone for depoIQ_ID {depoIQ_ID}.",
            "data": {
                "depoIQ_ID": depoIQ_ID,
                "total_inserted": total_inserted,
                "skipped_details": skipped_sub_categories,
                "skipped_count": sum(len(v) for v in skipped_sub_categories.values()),
            },
        }

        return response

    except Exception as e:
        return {"status": "error", "message": "Something went wrong", "details": str(e)}


def add_depo_transcript(transcript_data, depoIQ_ID):
    try:
        if not transcript_data:
            return {
                "status": "Not Found",
                "message": f"No transcript data found for depoIQ_ID {depoIQ_ID}.",
            }

        category = "transcript"  # Default category for transcripts

        # Store transcript data
        inserted_transcripts, skipped_transcripts = store_transcript_lines_in_pinecone(
            depoIQ_ID, category, transcript_data
        )

        response = {
            "status": "success",
            "message": f"Stored {inserted_transcripts} transcript chunks in Pinecone for depoIQ_ID {depoIQ_ID}.",
            "data": {
                "depoIQ_ID": depoIQ_ID,
                "total_inserted": inserted_transcripts,
                "skipped_details": skipped_transcripts,
                "skipped_count": len(skipped_transcripts),
            },
        }

        return response

    except Exception as e:
        return {"status": "error", "message": "Something went wrong", "details": str(e)}


# 🏠 Home Endpoint for testing
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Welcome to the Python Project API!"})


@app.route("/depo/<string:depoIQ_ID>", methods=["GET"])
def get_depo_by_Id(depoIQ_ID):
    """
    Get Depo By depoIQ_ID
    ---
    tags:
      - Depo
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
        depo = getDepo(depoIQ_ID, isSummary=True, isTranscript=True)
        return jsonify(depo), 200

    except Exception as e:
        return jsonify({"error": "Something went wrong", "details": str(e)}), 500


@app.route("/depo/add/<string:depoIQ_ID>", methods=["GET"])
def add_depo(depoIQ_ID):
    """
    Get Summaries & Transcript from Depo and store into Pinecone
    ---
    tags:
      - Depo
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
        # Get depo data
        depo = getDepo(depoIQ_ID, isSummary=True, isTranscript=True)

        # Extract summary & transcript data
        summary_data = depo.get("summary", {})
        transcript_data = depo.get("transcript", {})

        # Process & store summaries
        depo_response = add_depo_summaries(summary_data, depoIQ_ID)

        # Process & store transcript
        transcript_response = add_depo_transcript(transcript_data, depoIQ_ID)

        # Generate detailed response
        response = {
            "status": "success",
            "depo": depo_response,
            "transcript": transcript_response,
        }

        return jsonify(response), 200

    except Exception as e:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "Something went wrong",
                    "details": str(e),
                }
            ),
            500,
        )


@app.route("/depo/talk", methods=["POST"])
def talk_summary():
    """
    Talk to depo summaries & transcript by depoIQ_IDs
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
            depoIQ_IDs:
              type: array
              items:
                type: string
            user_query:
              type: string
            is_unique:
              type: boolean
    responses:
      200:
        description: Returns the success message
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid request, JSON body required"}), 400

        depoIQ_IDs = data.get("depoIQ_IDs")
        user_query = data.get("user_query")
        is_unique = data.get("is_unique", False)

        if depoIQ_IDs:
            if not user_query:
                return jsonify({"error": "Missing user_query"}), 400

            if len(depoIQ_IDs) == 0:
                return jsonify({"error": "Missing depoIQ_IDs list"}), 400

            if len(depoIQ_IDs) > 8:
                return jsonify({"error": "Too many depoIQ_IDs, max 8"}), 400
            # check all ids are valid and mongo Id
            for depo_id in depoIQ_IDs:
                if not ObjectId.is_valid(depo_id):
                    return jsonify({"error": "Invalid depo_id " + depo_id}), 400
            depoIQ_IDs_array_length = len(depoIQ_IDs) * 3
        else:
            depoIQ_IDs_array_length = 24

        top_k = depoIQ_IDs_array_length if is_unique else 10

        # ✅ Query Pinecone Safely
        query_pinecone_response = query_pinecone(
            user_query, depoIQ_IDs, top_k=top_k, is_unique=is_unique
        )

        print(
            f"\n\n✅ Query Pinecone Response: {json.dumps(query_pinecone_response, indent=2)}\n\n\n\n"
        )

        ai_resposne = get_answer_from_AI(query_pinecone_response)

        query_pinecone_response["answer"] = str(
            ai_resposne
        )  # Add AI response to the pinecone response

        return jsonify(query_pinecone_response), 200

    except Exception as e:
        return jsonify({"error": "Something went wrong", "details": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
