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
SUMMARIES_INDEX_NAME = os.getenv("SUMMARIES_INDEX_NAME", "summaries-index")
TRANSCRIPT_INDEX_NAME = os.getenv("TRANSCRIPT_INDEX_NAME", "transcript-index")

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
summariesIndex = initialize_pinecone_index(SUMMARIES_INDEX_NAME)
transcriptIndex = initialize_pinecone_index(TRANSCRIPT_INDEX_NAME)


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
            input=[text], model="text-embedding-ada-002"
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
            if isTranscript:
                grouped_data[depoIQ_ID] = getDepoTranscript(depoIQ_ID)

            if isSummary:
                grouped_data[depoIQ_ID] = getDepoSummary(depoIQ_ID)
    return grouped_data


# Function to query Pinecone and match summaries
def query_summaries_pinecone(query_text, depo_id=None, top_k=3):
    try:
        print(
            f"\n🔍 Querying Pinecone for summaries: '{query_text}' (depo_id: {depo_id})"
        )

        # Generate query embedding
        query_vector = generate_embedding(query_text)

        if not query_vector:
            return json.dumps(
                {"status": "error", "message": "Failed to generate query embedding."}
            )

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

        if not results["matches"] or not any(results["matches"]):
            print("\n\n No matches found. for summaries")
            return json.dumps({"status": "Not Found", "message": "No matches found."})

        matched_results = [
            {
                "category": match["metadata"]["category"],
                "db_category": snake_to_camel(
                    match["metadata"]["category"], isSummary=False
                ),
                "db_category_2": snake_to_camel(match["metadata"]["category"]),
                "depoIQ_ID": match["metadata"].get("depoIQ_ID"),
                "chunk_index": match["metadata"].get("chunk_index", None),
            }
            for match in results.get("matches", [])
        ]

        grouped_result = group_by_depoIQ_ID(
            matched_results, isSummary=True, isTranscript=False
        )

        custom_response = []

        for depData in matched_results:
            depo_id = depData["depoIQ_ID"]
            category = depData["db_category"]
            category_2 = depData["db_category_2"]
            chunk_index = int(depData["chunk_index"])
            summary = grouped_result.get(depo_id, {})

            db_text = (
                summary.get(category)
                if category in summary
                else summary.get(category_2, "No data found")
            )

            text = extract_text(db_text["text"])
            text_chunks = split_into_chunks(text)
            text = text_chunks[chunk_index] if chunk_index < len(text_chunks) else text

            custom_response.append(
                {
                    "depoIQ_ID": depo_id,
                    "category": depData["category"],
                    "chunk_index": chunk_index,
                    "text": text,
                }
            )

        # Construct initial response
        response = {
            "answer_for_query": (
                custom_response[0]["text"] if custom_response else "No answer found"
            ),
            "user_query": query_text,
            "metadata": custom_response,
        }

        # return json.dumps(response, indent=4)

        # Define GPT prompt
        prompt = f"""
                    You are an AI assistant that organizes and sorts JSON data efficiently.

                    Your task is to **analyze and sort the given JSON data** based on its relevance to the `user_query`
                    and return **only the top 3 most relevant results** while maintaining the original structure.

                    ---

                    ### **Given Data:**
                    {response}

                    ---

                    ### **Instructions:**
                    - You will receive a **JSON payload** containing:
                    - A **user_query**
                    - Multiple **metadata entries**, each containing a `"text"` field.
                    - **Your goal is to determine which `"text"` field best answers the `user_query` and rank the results accordingly.**
                    - **Select the most relevant `"text"`** from `metadata` and place it **exactly as it appears** in `"answer_for_query"`.  
                    - **Do NOT generate your own text.**
                    - **Only copy the most relevant `text` from metadata.**
                    - **Sort the top 3 most relevant results** in `"metadata"`, ensuring the best response is at the top.
                    - **DO NOT modify the JSON structure**. The only allowed modifications are:
                    - Assigning the most relevant `"text"` **exactly as it is** to `"answer_for_query"`.
                    - Reordering the `"metadata"` array to prioritize relevance.
                    - Ensuring only the **top 3 most relevant results** remain in `"metadata"`.

                    ---

                    ### **Sorting Criteria:**
                    - **Highest Relevance:** The `"text"` that directly and accurately answers the `"user_query"` should be placed in `"answer_for_query"`, without modifications.
                    - **Top 3 Results:** The `"metadata"` array should contain only the **three** most relevant entries.
                    - **If relevance is equal, prioritize the `"text"` that provides more details and context.**
                    - **Do not include irrelevant, vague, or unrelated responses.**
                    - **Do not summarize, rewrite, or alter any text. Return the original content as it is.**

                    ---
                    """

        # Call GPT-4o-mini API
        ai_response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI assistant that extracts precise answers from multiple given documents efficiently.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=2000,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "query_response",
                    "description": "A structured response for a query and its matched answers.",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "answer_for_query": {"type": "string"},
                            "user_query": {"type": "string"},
                            "metadata": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "category": {"type": "string"},
                                        "chunk_index": {"type": "string"},
                                        "depoIQ_ID": {"type": "string"},
                                        "text": {"type": "string"},
                                    },
                                    "required": [
                                        "category",
                                        "chunk_index",
                                        "depoIQ_ID",
                                        "text",
                                    ],
                                },
                            },
                        },
                        "required": ["answer_for_query", "user_query", "metadata"],
                    },
                },
            },
        )

        # Convert response to JSON
        ai_json_response = json.loads(ai_response.choices[0].message.content.strip())

        return json.dumps(ai_json_response, indent=4)

    except Exception as e:
        print(f"Error querying Pinecone: {e}")
        return {"status": "error", "message": str(e)}


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
def split_into_chunks(text):
    """Splits text into paragraphs while removing empty strings."""
    return [chunk.strip() for chunk in re.split("\n\n", text.strip()) if chunk.strip()]


# Function to check if a summary already exists in Pinecone
def check_existing_summary_entry(depoIQ_ID, category, chunk_index):
    """Checks if a summary already exists in Pinecone."""
    try:
        existing_entries = summariesIndex.query(
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


# Function to check if a transcript already exists in Pinecone
def check_existing_transcript_entry(depoIQ_ID, endPage, startPage):
    """Checks if a transcript already exists in Pinecone."""
    try:
        existing_entries = transcriptIndex.query(
            vector=np.random.rand(1536).tolist(),  # Use a dummy query vector
            filter={
                "depoIQ_ID": depoIQ_ID,
                "startPage": startPage,
                "endPage": endPage,
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
        if check_existing_summary_entry(depoIQ_ID, category, chunk_index):
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

        summariesIndex.upsert(vectors=vectors_to_upsert)
        print(
            f"✅ Successfully inserted {len(vectors_to_upsert)} summaries in Pinecone."
        )

    return len(vectors_to_upsert), skipped_chunks


def store_transcript_lines_in_pinecone(depoIQ_ID, transcript_data):
    vectors_to_upsert = []
    skipped_chunks = []
    chunk_page_range = 3

    for i in range(0, len(transcript_data), chunk_page_range):
        grouped_pages = transcript_data[i : i + chunk_page_range]

        # check if already exists in pinecone
        if check_existing_transcript_entry(
            depoIQ_ID,
            endPage=grouped_pages[-1]["pageNumber"],
            startPage=grouped_pages[0]["pageNumber"],
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
            "startPage": grouped_pages[0]["pageNumber"],
            "endPage": grouped_pages[-1]["pageNumber"],
            # "text": trimmed_text,  # Store transcript text in metadata
            # "embedding": json.dumps(embedding),  # Store embedding as JSON string
        }

        vectors_to_upsert.append(
            {
                "id": str(uuid.uuid4()),  # Unique ID
                "values": embedding,  # Embedding vector
                "metadata": metadata,  # Metadata including the text
            }
        )

    if vectors_to_upsert:
        transcriptIndex.upsert(vectors=vectors_to_upsert)
        print(
            f"✅ Successfully inserted {len(vectors_to_upsert)} transcript chunks in Pinecone."
        )

    return len(vectors_to_upsert), skipped_chunks


def query_transcript_pinecone(query_text, depo_id, top_k=5):
    """Queries Pinecone for transcript data and finds the best match directly."""
    try:
        print(
            f"\n🔍 Querying Pinecone for transcript: '{query_text}' (depo_id: {depo_id})"
        )

        # Generate embedding for the query
        query_vector = generate_embedding(query_text)

        if not query_vector:
            return json.dumps(
                {"status": "error", "message": "Failed to generate query embedding."}
            )

        # Define filter criteria to search within the specific depo_id
        filter_criteria = {"depoIQ_ID": depo_id}

        # Search in Pinecone
        results = transcriptIndex.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            filter=filter_criteria,
        )

        if not results["matches"] or not any(results["matches"]):
            print("\n\n No matches found. for transcript")
            return json.dumps({"status": "Not Found", "message": "No matches found."})

        matched_results = [
            {
                "depoIQ_ID": match["metadata"].get("depoIQ_ID"),
                "start_page": match["metadata"].get("startPage", 1),
                "end_page": match["metadata"].get("endPage", 1),
                "score": match["score"],
            }
            for match in results.get("matches", [])
        ]

        grouped_result = group_by_depoIQ_ID(
            matched_results, isSummary=False, isTranscript=True
        )

        custom_match = []

        for match in matched_results:
            depoIQ_ID = match["depoIQ_ID"]
            start_page = int(match["start_page"])
            end_page = int(match["end_page"])
            transcript = grouped_result.get(depoIQ_ID, {})

            pages = [
                page
                for page in transcript
                if start_page <= page["pageNumber"] <= end_page
            ]

            # Extract the text from the retrieved pages

            chunk_text = {}

            for page in pages:
                page_number = page["pageNumber"]
                pageLines = page.get("lines", [])
                pageLines = [
                    line["lineText"].strip()
                    for line in pageLines
                    if line["lineText"].strip()
                ]
                chunk_text[page_number] = " ".join(pageLines)

            custom_match.append(
                {
                    "depoIQ_ID": depoIQ_ID,
                    "start_page": start_page,
                    "end_page": end_page,
                    "chunk_text": chunk_text,
                }
            )

        # Construct response
        response = {"query_text": query_text, "metadata": custom_match}

        # return json.dumps(response, indent=4)

        # Example response structure
        prompt = f"""
                You are an AI assistant that extracts and organizes transcript data efficiently.

                Your task is to **analyze and sort the given transcript JSON data** based on its relevance to the `user_query`
                and return **only the top 3 most relevant responses** in `"metadata"` while maintaining the original structure.

                Additionally, extract the **most relevant page's data** from `"chunk_text"` and place it in `"answer_for_query"`.
                The **original structure of `"metadata"` must remain unchanged**, except for sorting based on relevance.

                ---

                ### **Given Data:**
                {response}

                ---

                ### **Instructions:**
                - You will receive a **JSON payload** containing:
                - A **user_query**
                - Multiple **metadata entries**, each containing:
                    - A `"depoIQ_ID"`
                    - `"start_page"` and `"end_page"`
                    - A `"chunk_text"` object, where keys represent page numbers and values contain transcript text.

                - **Your goal is to determine which page best answers the `user_query` and return its exact `"chunk_text"` in `"answer_for_query"`**.
                - **Ensure `"answer_for_query"` contains the exact text from the most relevant `"chunk_text"` page**.
                - **Extract the top three most relevant metadata entries and ONLY sort them** without modifying any data.
                - **DO NOT alter, modify, or rewrite `"chunk_text"` in `"metadata"`—it must remain exactly the same.**
                - **Include `"start_page"` and `"end_page"` in the main response, which should match the values from the most relevant `"chunk_text"`.**
                - **Verify multiple times** that `"chunk_text"` in `"metadata"` is **unchanged and complete** before returning the final output.
                - **DO NOT truncate, summarize, or alter any text.** The original structure must be preserved.
                - **Ensure that metadata contains exactly THREE entries** in the response.

                ---

                ### **Sorting Criteria:**
                - **Highest Relevance:** The `"chunk_text"` that directly and accurately answers the `"user_query"` should be placed in `"answer_for_query"`, using the **most relevant page only**.
                - **Metadata Sorting:** **Sort the `"metadata"` array** by relevance, keeping all entries unchanged except for order.
                - **Ensure that the selected `"answer_for_query"` comes from the correct `"chunk_text"` in `"metadata"`**.
                - **DO NOT modify or truncate `"chunk_text"` in `"metadata"`.**
                - **Return exactly THREE metadata entries, even if more exist.**


                ---
                """

        # Call GPT-4o-mini API
        ai_response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI assistant that extracts precise answers from multiple given transcripts efficiently.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=2000,
            response_format={
                "type": "json_schema",
                "json_schema": json.loads(
                    """
                {
                    "name": "query_response",
                    "description": "A structured response for a query and its matched answers.",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "answer_for_query": { "type": "string" },
                            "user_query": { "type": "string" },
                            "start_page": { "type": "integer" },
                            "end_page": { "type": "integer" },
                            "metadata": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "depoIQ_ID": { "type": "string" },
                                        "start_page": { "type": "integer" },
                                        "end_page": { "type": "integer" },
                                        "chunk_text": {
                                            "type": "object",
                                            "additionalProperties": { "type": "string" }
                                        }
                                    },
                                    "required": ["depoIQ_ID", "start_page", "end_page", "chunk_text"]
                                },
                                "minItems": 3,
                                "maxItems": 3
                            }
                        },
                        "required": ["answer_for_query", "user_query", "start_page", "end_page", "metadata"]
                    }
                }
                """
                ),
            },
        )

        print("Successfull response from GPT-4o-mini modal ->", ai_response)

        # Convert response to JSON
        ai_json_response = json.loads(ai_response.choices[0].message.content.strip())

        print(f"AI JSON Response: ---> {ai_json_response}")

        return json.dumps(ai_json_response, indent=4)

    except Exception as e:
        print(f"❌ Error querying Pinecone: {e}")
        return {"status": "error", "message": str(e)}


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

        # Store transcript data
        inserted_transcripts, skipped_transcripts = store_transcript_lines_in_pinecone(
            depoIQ_ID, transcript_data
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
def get_depo(depoIQ_ID):
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
    Talk to depo summaries & transcript by depo_id
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

        # Check if depo_id and user_query are provided
        if not depo_id or not user_query:
            return jsonify({"error": "Missing depo_id or user_query"}), 400

        # Query Pinecone for summary
        summaries_response = query_summaries_pinecone(user_query, depo_id, top_k=8)

        # Query Pinecone for transcript
        transcript_response = query_transcript_pinecone(user_query, depo_id, top_k=8)

        # Construct final response
        response = {
            "summaries": json.loads(summaries_response),
            "transcript": json.loads(transcript_response),
        }

        print(f"\n\n Final Response: {response}")

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": "Something went wrong", "details": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
