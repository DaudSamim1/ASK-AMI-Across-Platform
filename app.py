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
DEPO_INDEX_NAME = os.getenv("DEPO_INDEX_NAME")

# Ensure keys are loaded
if not OPENAI_API_KEY or not PINECONE_API_KEY or not DEPO_INDEX_NAME:
    raise ValueError("Missing API Keys! or DEPO_INDEX_NAME Check your .env file.")


app = Flask(__name__)
swagger = Swagger(app)


@app.after_request
def disable_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response


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
def group_by_depoIQ_ID(
    data, isSummary=False, isTranscript=False, isContradictions=False, isAdmission=False
):
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

            if isContradictions and not isSummary and not isTranscript:
                grouped_data[depoIQ_ID] = getDepoContradictions(depoIQ_ID)
                return grouped_data

            if (
                isAdmission
                and not isSummary
                and not isTranscript
                and not isContradictions
            ):
                grouped_data[depoIQ_ID] = getDepoAdmissions(depoIQ_ID)
                return grouped_data

            grouped_data[depoIQ_ID] = getDepo(
                depoIQ_ID,
                isSummary=True,
                isTranscript=True,
                isContradictions=True,
                isAdmission=True,
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
        contradiction_category = "contradiction"
        admission_category = "admission"
        filter_criteria = {}
        if depo_ids and len(depo_ids) > 0:
            # Add depo_id filter
            filter_criteria["depoIQ_ID"] = {"$in": depo_ids}
            filter_criteria["category"] = {"$ne": contradiction_category}

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
            return {"status": "Not Found", "message": "No matches found."}

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
            matched_results, isSummary=True, isTranscript=True, isContradictions=True
        )

        # print(f"Grouped Results: {grouped_result} \n\n\n")

        custom_response = []  # Custom response

        for match in matched_results:
            depoIQ_ID = match["depoIQ_ID"]
            category = match["category"]
            isSummary = (
                category != transcript_category
                and category != contradiction_category
                and category != admission_category
            )
            isTranscript = category == transcript_category
            isContradictions = category == contradiction_category
            isAdmission = category == admission_category

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

            # if isContradictions:
            #     chunk_index = match["chunk_index"]
            #     contradictions = depo_data.get("contradictions", {})
            #     contradiction = contradictions[int(chunk_index)]
            #     reason = contradiction.get("reason")
            #     initial_question = contradiction.get("initial_question")
            #     initial_answer = contradiction.get("initial_answer")
            #     contradictory_responses = contradiction.get("contradictory_responses")
            #     contradictory_responses_string = ""
            #     for contradictory_response in contradictory_responses:
            #         contradictory_responses_string += f"{contradictory_response['contradictory_question']} {contradictory_response['contradictory_answer']} "

            #     text = f"{initial_question} {initial_answer} {contradictory_responses_string} {reason}"
            #     custom_response.append(
            #         {
            #             "depoIQ_ID": depoIQ_ID,
            #             "category": category,
            #             "chunk_index": chunk_index,
            #             "text": text,
            #         }
            #     )

            if isAdmission:
                chunk_index = match["chunk_index"]
                admissions = depo_data.get("admissions", {})

                [admission] = [
                    admission
                    for admission in admissions["text"]
                    if admission["admission_id"] == chunk_index
                ]
                reason = admission.get("reason")
                question = admission.get("question")
                answer = admission.get("answer")
                text = f"Question: {question} Answer: {answer} Reason:  {reason}"

                custom_response.append(
                    {
                        "depoIQ_ID": depoIQ_ID,
                        "category": category,
                        "chunk_index": chunk_index,
                        "text": text,
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
                "status": "success",
                "user_query": query_text,
                "depoIQ_ID": unique_set_response_keys,
                "metadata": final_response,
                "is_unique": is_unique,
            }
        else:

            grouped_result_keys = list(grouped_result.keys())

            response = {
                "status": "success",
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
                - Analyze the provided "metadata" and generate direct, relevant, and **fully explained answers** for each question in "user_query".
                - **If "user_query" contains multiple questions, detect and SEPARATE each sub-question into its own distinct response.**
                - **Each sub-question must be answered separately, clearly, and thoroughly.**
                - Use the "text" field in "metadata" to form the most **detailed and informative** responses.
                - Ensure the generated responses **fully explain** each answer instead of providing short, incomplete summaries.
                - **Each answer must be at least 400 characters long but can exceed this if necessary.**
                - Reference the source by indicating all "metadata" **OBJECT POSITIONS IN THE ARRAY (NOT chunk_index, IDs, or any other identifiers).**

                ---

                ### **Given Data:**
                {json.dumps(response)}

                ---

                ### **🚨 STRICT INSTRUCTIONS (DO NOT VIOLATE THESE RULES):**
                - **FIRST: Identify if "user_query" contains multiple questions. If so, split them into individual questions.**
                - **SECOND: Generate a separate, fully detailed answer for EACH question. DO NOT merge answers together.**
                - **Each answer MUST be at least 400 characters. DO NOT provide less than 400 characters, but exceeding this is allowed.**
                - **DO NOT return "No relevant information available." if ANY metadata source contains relevant information.**
                - **Only generate "No relevant information available. &&metadataRef = []" if ZERO metadata sources contain ANY relevant information.**
                - **STRICTLY structure responses so that each question gets its own distinct, fully explained answer.**
                - **REFERENCE SOURCES USING ONLY OBJECT POSITIONS in the metadata array. These are the indices corresponding to the order of appearance of metadata items in the array.**
                - **DO NOT use chunk_index, IDs, hashes, or any other values. ONLY the object’s array position in the metadata list.**
                - **DO NOT use index ranges such as [4-6]. ALWAYS list individual indices explicitly, e.g., [4, 5, 6]. Range notation is STRICTLY FORBIDDEN.**
                - **INDEX POSITIONS must correspond exactly to each metadata item’s position in the "metadata" array as provided.**
                - **DO NOT OMIT &&metadataRef =. It MUST always be included at the end of the answer.**
                - **DO NOT add newline characters (\\n) before or after the response. The response must be in a SINGLE, FLAT LINE.**
                - **DO NOT enclose the response inside triple quotes (''', "" ") or markdown-style code blocks (``` ). Return it as PLAIN TEXT ONLY.**
                - **DO NOT add unnecessary labels like "Extracted Answer:" or "Answer:". The response must start directly with the extracted answer.**
                - **DO NOT format the response as JSON, XML, or any structured format—return plain text only.**
                - **DO NOT change sentence structure unless strictly necessary to form a complete, grammatically correct sentence.**
                - **FORCE OUTPUT AS A CLEAN, SINGLE LINE WITH NO EXTRA WHITESPACES OR NEWLINES.**
                - **ENSURE EACH ANSWER IS FULLY EXPLAINED, PROVIDING DETAILED CONTEXT SO THE USER GETS COMPLETE INFORMATION.**
                - **ENSURE EVERY RESPONSE CONTAINS AT LEAST 400 CHARACTERS. IF NECESSARY, ADD ADDITIONAL CONTEXT TO REACH THIS MINIMUM LENGTH.**
                - **DO NOT GROUP MULTIPLE QUESTIONS INTO A SINGLE ANSWER. EACH MUST BE SEPARATE.**

                ---

                ### **🚨 FINAL OUTPUT FORMAT (NO EXCEPTIONS, FOLLOW THIS EXACTLY):**

                <Detailed Answer for Question 1> &&metadataRef = [X, Y, Z]  
                <Detailed Answer for Question 2> &&metadataRef = [A, B, C]  
                <Detailed Answer for Question 3> &&metadataRef = [D, E, F]  
                <Detailed Answer for Question 4> &&metadataRef = [G, H, I]  

                - **Example of Correct Output for Multiple Questions (DETAILED RESPONSES, NO NEWLINES, MINIMUM 400 CHARACTERS PER ANSWER):**

                The expert witness deposed in this case was Mark Strassberg, M.D. He has significant experience in forensic psychiatry and has been involved in multiple legal cases, providing expert testimony. &&metadataRef = [0, 1]  

                Dr. Strassberg specializes in forensic and clinical practice. His expertise includes forensic psychiatry, medical evaluations, and expert testimony, with 85% of his forensic practice being for defendants. &&metadataRef = [2]  

                Dr. Strassberg was retained by Mr. Wilson for this case. Mr. Wilson has worked with Dr. Strassberg on multiple cases due to his specialization in forensic evaluations. &&metadataRef = [3]  

                Dr. Strassberg has worked with Mr. Wilson approximately 5 or 6 times before this case. However, he did not keep records of previous engagements and could not recall specific details of past collaborations. &&metadataRef = [4]  

                - **DO NOT use index ranges such as [4-6]. ALWAYS list each metadata index individually, e.g., [4, 5, 6]. Range notation is STRICTLY FORBIDDEN.**

                - **If no relevant metadata is found for a specific question, return EXACTLY this (NO MODIFICATIONS, NO EXTRA SPACES OR NEWLINES):**

                No relevant information available. &&metadataRef = []

                ### **🚨 ENFORCEMENT RULES (MUST BE FOLLOWED 100% EXACTLY):**
                - **FORCE SEPARATE ANSWERS FOR EACH QUESTION IN THE QUERY. DO NOT COMBINE MULTIPLE QUESTIONS INTO A SINGLE RESPONSE.**
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
def getDepo(
    depoIQ_ID,
    isSummary=True,
    isTranscript=True,
    isContradictions=True,
    isAdmission=True,
):
    try:
        # GraphQL query with conditional @include directives
        query = """
        query GetDepoSummary($depoId: ID!, $includeSummary: Boolean!, $includeTranscript: Boolean!, $includeContradictions: Boolean!, $includeAdmission: Boolean!) {
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
                contradictions @include(if: $includeContradictions) {
                    reason
                    initial_response_starting_page
                    initial_response_starting_line
                    initial_response_ending_page
                    initial_response_ending_line
                    initial_question
                    initial_answer
                    contradictory_responses {
                        contradictory_response_starting_page
                        contradictory_response_starting_line
                        contradictory_response_ending_page
                        contradictory_response_ending_line
                        contradictory_question
                        contradictory_answer
                    }
                }
                admissions @include(if: $includeAdmission) {
                    text {
                        reference {
                        start {
                            line
                            page
                        }
                        end {
                            page
                            line
                        }
                        }
                        admission_id
                        answer
                        question
                        reason
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
                "includeContradictions": isContradictions,
                "includeAdmission": isAdmission,
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
        depo = getDepo(
            depoIQ_ID,
            isSummary=True,
            isTranscript=False,
            isContradictions=False,
            isAdmission=False,
        )
        return depo["summary"]
    except Exception as e:
        print(f"Error fetching depo summary: {e}")
        return {}


# Function to generate Get Transcript from Depo
def getDepoTranscript(depoIQ_ID):
    """Get Depo Transcript from DepoIQ_ID"""
    try:
        depo = getDepo(
            depoIQ_ID,
            isSummary=False,
            isTranscript=True,
            isContradictions=False,
            isAdmission=False,
        )
        return depo["transcript"]
    except Exception as e:
        print(f"Error fetching depo transcript: {e}")
        return {}


# Function to genrate Get Contradictions from Depo
def getDepoContradictions(depoIQ_ID):
    """Get Depo Contradictions from DepoIQ_ID"""
    try:
        depo = getDepo(
            depoIQ_ID,
            isSummary=False,
            isTranscript=False,
            isContradictions=True,
            isAdmission=False,
        )
        return depo["contradictions"]
    except Exception as e:
        print(f"Error fetching depo contradictions: {e}")
        return {}


# Function to genrate Get Admissions from Depo
def getDepoAdmissions(depoIQ_ID):
    """Get Depo Admissions from DepoIQ_ID"""
    try:
        depo = getDepo(
            depoIQ_ID,
            isSummary=False,
            isTranscript=False,
            isContradictions=False,
            isAdmission=True,
        )
        return depo["admissions"]
    except Exception as e:
        print(f"Error fetching depo admissions: {e}")
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

        print(f"\n\n\n\n🔹 Processing chunk {chunk_index + 1} of {category}\n\n\n\n")

        if not chunk_value or not isinstance(chunk_value, str):
            print(f"⚠️ Skipping invalid text chunk: {chunk_index}\n\n\n\n")
            continue

        # Check if summary already exists
        if check_existing_entry(depoIQ_ID, category, chunk_index):
            skipped_chunks.append(chunk_index)
            print(f"⚠️ Summary already exists: {chunk_index}, skipping...\n\n\n\n")
            continue

        # Generate embedding
        embedding = generate_embedding(chunk_value)

        if not any(embedding):  # Check for all zero vectors
            print(f"⚠️ Skipping zero-vector embedding for: {chunk_index}\n\n\n\n")
            continue

        # Metadata
        metadata = {
            "depoIQ_ID": depoIQ_ID,
            "category": category,
            "chunk_index": chunk_index,
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
    max_chunk_upsert = 20

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

        if len(vectors_to_upsert) >= max_chunk_upsert:
            depoIndex.upsert(vectors=vectors_to_upsert)
            print(
                f"✅ Successfully inserted {len(vectors_to_upsert)} transcript chunks in Pinecone.\n\n"
            )
            vectors_to_upsert = []

    if vectors_to_upsert:
        depoIndex.upsert(vectors=vectors_to_upsert)
        print(
            f"✅ Successfully inserted {len(vectors_to_upsert)} transcript chunks in Pinecone."
        )

    return len(vectors_to_upsert), skipped_chunks


def store_contradictions_in_pinecone(depoIQ_ID, category, contradictions_data):
    vectors_to_upsert = []
    skipped_chunks = []
    max_chunk_upsert = 30

    try:
        print(
            f"🔹 Storing contradictions for depoIQ_ID: {depoIQ_ID} {len(contradictions_data)}"
        )

        for chunk_index in range(0, len(contradictions_data)):
            contradiction = contradictions_data[chunk_index]

            print(f"\n\n 🔹 Storing contradiction: {chunk_index} \n\n\n\n")

            # check if already exists in pinecone
            if check_existing_entry(
                depoIQ_ID,
                category,
                chunk_index,
            ):
                skipped_chunks.append(chunk_index)
                print(
                    f"⚠️ Contradiction already exists for pages {chunk_index}, skipping..."
                )
                continue

            # Extract contradiction text, ensuring each line is properly formatted
            reason = contradiction.get("reason")
            contradictory_responses = contradiction.get("contradictory_responses")
            initial_question = contradiction.get("initial_question")
            initial_answer = contradiction.get("initial_answer")
            contradictory_responses_string = ""
            for contradictory_response in contradictory_responses:
                contradictory_responses_string += f"{contradictory_response['contradictory_question']} {contradictory_response['contradictory_answer']} "

            contradiction_text = f"{initial_question} {initial_answer} {contradictory_responses_string} {reason}"

            # Generate embedding
            embedding = generate_embedding(contradiction_text)

            if not any(embedding):
                print(
                    f"⚠️ Skipping empty embedding for contradiction: {contradiction_text}"
                )
                continue

            # Add the actual transcript text in metadata for retrieval
            metadata = {
                "depoIQ_ID": depoIQ_ID,
                "category": category,
                "chunk_index": chunk_index,
            }

            # Add to batch
            vectors_to_upsert.append(
                {
                    "id": str(uuid.uuid4()),  # Unique ID
                    "values": embedding,  # Embedding vector
                    "metadata": metadata,  # Metadata including the text
                }
            )

            if len(vectors_to_upsert) >= max_chunk_upsert:
                depoIndex.upsert(vectors=vectors_to_upsert)
                print(
                    f"✅ Successfully inserted {len(vectors_to_upsert)} contradictions in Pinecone.\n\n"
                )
                vectors_to_upsert = []

        if vectors_to_upsert:
            depoIndex.upsert(vectors=vectors_to_upsert)
            print(
                f"✅ Successfully inserted {len(vectors_to_upsert)} contradictions in Pinecone."
            )

        return len(vectors_to_upsert), skipped_chunks

    except Exception as e:
        print(f"🔹 Error in store_contradictions_in_pinecone: {e}")
        return 0, []


def store_admissions_in_pinecone(depoIQ_ID, category, admissions_data):
    vectors_to_upsert = []
    skipped_chunks = []
    max_chunk_upsert = 30

    try:
        print(
            f"🔹 Storing admissions for depoIQ_ID: {depoIQ_ID} {len(admissions_data)}"
        )

        for index in range(0, len(admissions_data)):
            admission = admissions_data[index]
            chunk_index = admission.get("admission_id")

            if chunk_index == None:
                print(
                    f"⚠️ Skipping admission: {admission} because chunk_index is None \n\n\n\n"
                )
                continue

            print(f"\n\n 🔹 Storing admission: {chunk_index} \n\n\n\n")

            # check if already exists in pinecone
            if check_existing_entry(
                depoIQ_ID,
                category,
                chunk_index,
            ):
                skipped_chunks.append(chunk_index)
                print(
                    f"⚠️ Admission already exists for pages {chunk_index}, skipping..."
                )
                continue

            # Extract admission text, ensuring each line is properly formatted
            reason = admission.get("reason")
            question = admission.get("question")
            answer = admission.get("answer")
            admission_text = f"{question} {answer} {reason}"

            print(f"\n\n\n\n🔹 Admission text: {admission_text}\n\n\n\n\n")

            # Generate embedding
            embedding = generate_embedding(admission_text)

            if not any(embedding):
                print(f"⚠️ Skipping empty embedding for admission: {admission_text}")
                continue

            # Add the actual transcript text in metadata for retrieval
            metadata = {
                "depoIQ_ID": depoIQ_ID,
                "category": category,
                "chunk_index": chunk_index,
            }

            # Add to batch
            vectors_to_upsert.append(
                {
                    "id": str(uuid.uuid4()),  # Unique ID
                    "values": embedding,  # Embedding vector
                    "metadata": metadata,  # Metadata including the text
                }
            )

            if len(vectors_to_upsert) >= max_chunk_upsert:
                depoIndex.upsert(vectors=vectors_to_upsert)
                print(
                    f"✅ Successfully inserted {len(vectors_to_upsert)} admissions in Pinecone."
                )
                vectors_to_upsert = []

        if vectors_to_upsert:
            depoIndex.upsert(vectors=vectors_to_upsert)
            print(
                f"✅ Successfully inserted {len(vectors_to_upsert)} admissions in Pinecone.\n\n\n"
            )

        return len(vectors_to_upsert), skipped_chunks

    except Exception as e:
        print(f"🔹 Error in store_admissions_in_pinecone: {e}")
        return 0, []


# Function to generate add depo summaries to pinecone
def add_depo_summaries(depo_summary, depoIQ_ID):
    try:
        excluded_keys = ["visualization"]
        total_inserted = 0
        skipped_sub_categories = {}

        print(f"🔹 Adding summaries for depoIQ_ID: {depoIQ_ID}")

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

        if total_inserted > 0:
            status = "success"
        elif skipped_sub_categories:
            status = "warning"
        else:
            status = "error"

        response = {
            "status": status,
            "message": "Summaries processed.",
            "data": {
                "depoIQ_ID": depoIQ_ID,
                "total_inserted": total_inserted,
                "skipped_details": skipped_sub_categories,
                "skipped_count": sum(len(v) for v in skipped_sub_categories.values()),
            },
        }

        return response

    except Exception as e:
        return {
            "status": "error",
            "message": "Something went wrong in add_depo_summaries",
            "details": str(e),
        }


def add_depo_transcript(transcript_data, depoIQ_ID):
    try:
        print(f"🔹 Adding transcripts for depoIQ_ID: {depoIQ_ID}")
        if not transcript_data:
            return {
                "status": "warning",
                "message": "No transcript data found.",
                "data": {
                    "depoIQ_ID": depoIQ_ID,
                    "total_inserted": 0,
                    "skipped_details": [],
                    "skipped_count": 0,
                },
            }

        category = "transcript"  # Default category for transcripts

        # Store transcript data
        inserted_transcripts, skipped_transcripts = store_transcript_lines_in_pinecone(
            depoIQ_ID, category, transcript_data
        )

        if inserted_transcripts > 0:
            status = "success"
        elif skipped_transcripts:
            status = "warning"
        else:
            status = "error"

        response = {
            "status": status,
            "message": "Transcripts processed.",
            "data": {
                "depoIQ_ID": depoIQ_ID,
                "total_inserted": inserted_transcripts,
                "skipped_details": skipped_transcripts,
                "skipped_count": len(skipped_transcripts),
            },
        }

        return response

    except Exception as e:
        return {
            "status": "error",
            "message": "Something went wrong in add_depo_transcript",
            "details": str(e),
        }


def add_depo_contradictions(contradictions_data, depoIQ_ID):
    try:
        print(f"🔹 Adding contradictions for depoIQ_ID: {depoIQ_ID}")
        if not contradictions_data:
            return {
                "status": "warning",
                "message": "No contradictions data found.",
                "data": {
                    "depoIQ_ID": depoIQ_ID,
                    "total_inserted": 0,
                    "skipped_details": [],
                    "skipped_count": 0,
                },
            }

        category = "contradiction"

        # Store contradictions data
        inserted_contradictions, skipped_contradictions = (
            store_contradictions_in_pinecone(depoIQ_ID, category, contradictions_data)
        )

        print()

        if inserted_contradictions > 0:
            status = "success"
        elif skipped_contradictions:
            status = "warning"
        else:
            status = "error"

        response = {
            "status": status,
            "message": "Contradictions processed.",
            "data": {
                "depoIQ_ID": depoIQ_ID,
                "total_inserted": inserted_contradictions,
                "skipped_details": skipped_contradictions,
                "skipped_count": len(skipped_contradictions),
            },
        }

        return response

    except Exception as e:
        return {
            "status": "error",
            "message": "Something went wrong in add_depo_contradictions",
            "details": str(e),
        }


def add_depo_admissions(admissions_data, depoIQ_ID):
    try:
        print(f"🔹 Adding admissions for depoIQ_ID: {depoIQ_ID}\n\n")

        if not admissions_data:
            return {
                "status": "warning",
                "message": "No admissions data found.",
                "data": {
                    "depoIQ_ID": depoIQ_ID,
                    "total_inserted": 0,
                    "skipped_details": [],
                    "skipped_count": 0,
                },
            }

        category = "admission"

        # Store admissions data
        inserted_admissions, skipped_admissions = store_admissions_in_pinecone(
            depoIQ_ID, category, admissions_data=admissions_data["text"]
        )

        if inserted_admissions > 0:
            status = "success"
        elif skipped_admissions:
            status = "warning"
        else:
            status = "error"

        response = {
            "status": status,
            "message": "Admissions processed.",
            "data": {
                "depoIQ_ID": depoIQ_ID,
                "total_inserted": inserted_admissions,
                "skipped_details": skipped_admissions,
                "skipped_count": len(skipped_admissions),
            },
        }

        return response

    except Exception as e:
        return {
            "status": "error",
            "message": "Something went wrong in add_depo_admissions",
            "details": str(e),
        }


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
        depo = getDepo(
            depoIQ_ID,
            isSummary=True,
            isTranscript=True,
            isContradictions=True,
            isAdmission=True,
        )
        return jsonify(depo), 200

    except Exception as e:
        return (
            jsonify(
                {"error": "Something went wrong in get_depo_by_Id", "details": str(e)}
            ),
            500,
        )


@app.route("/depo/add/<string:depoIQ_ID>", methods=["GET"])
def add_depo(depoIQ_ID):
    """
    Get Summaries & Transcript & Contradictions & Admissions from Depo and store into Pinecone
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
        depo = getDepo(
            depoIQ_ID,
            isSummary=True,
            isTranscript=True,
            isContradictions=True,
            isAdmission=True,
        )

        # Extract summary & transcript data
        summary_data = depo.get("summary", {})
        transcript_data = depo.get("transcript", [])
        contradictions_data = depo.get("contradictions", [])
        admissions_data = depo.get("admissions", [])

        # Process & store summaries
        summmary_response = add_depo_summaries(summary_data, depoIQ_ID)

        # Process & store transcript
        transcript_response = add_depo_transcript(transcript_data, depoIQ_ID)

        # Process & store contradictions
        # contradictions_response = add_depo_contradictions(
        #     contradictions_data, depoIQ_ID
        # )

        # Process & store admissions
        admissions_response = add_depo_admissions(admissions_data, depoIQ_ID)

        # Determine status
        if (
            summmary_response["status"] == "success"
            and transcript_response["status"] == "success"
            # and contradictions_response["status"] == "success"
            and admissions_response["status"] == "success"
        ):
            status = "success"
        elif (
            summmary_response["status"] == "warning"
            or transcript_response["status"] == "warning"
            # or contradictions_response["status"] == "warning"
            or admissions_response["status"] == "warning"
        ):
            status = "warning"
        else:
            status = "error"

        # Merge responses
        merged_response = {
            "status": status,
            "depoIQ_ID": depoIQ_ID,
            # "message": f"Stored {summmary_response['data']['total_inserted']} summaries and {transcript_response['data']['total_inserted']} transcript chunks and {contradictions_response['data']['total_inserted']} contradictions and {admissions_response['data']['total_inserted']} admissions in Pinecone for depoIQ_ID {depoIQ_ID}.",
            "message": f"Stored {summmary_response['data']['total_inserted']} summaries and {transcript_response['data']['total_inserted']} transcript chunks and {admissions_response['data']['total_inserted']} admissions in Pinecone for depoIQ_ID {depoIQ_ID}.",
            "summary": summmary_response["data"],
            "transcript": transcript_response["data"],
            # "contradictions": contradictions_response["data"],
            "admissions": admissions_response["data"],
        }

        return jsonify(merged_response), 200

    except Exception as e:
        print(f"🔹 Error in add_depo: {e}")
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

        if query_pinecone_response["status"] == "Not Found":
            print(f"\n\n🔍 No matches found for the query and return error \n\n\n\n")
            return (
                jsonify(
                    {
                        "error": "No matches found for the query",
                        "details": "No matches found",
                        "status": "Not Found",
                    }
                ),
                200,
            )
        else:
            ai_resposne = get_answer_from_AI(query_pinecone_response)

            print(f"\n\n✅ AI Response: {ai_resposne} \n\n\n\n")

            query_pinecone_response["answer"] = str(
                ai_resposne
            )  # Add AI response to the pinecone response

        return jsonify(query_pinecone_response), 200

    except Exception as e:
        return (
            jsonify(
                {"error": "Something went wrong in talk_summary", "details": str(e)}
            ),
            500,
        )


if __name__ == "__main__":
    app.run(debug=True)
