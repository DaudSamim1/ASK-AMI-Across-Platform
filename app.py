from flask import Flask, jsonify, request, Response
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
from datetime import datetime
import csv
import io

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
        if not text or not isinstance(text, str) or text.strip() == "":
            print("⚠️ Invalid or empty text for embedding generation. Text:", text)
            return False

        text = text.replace("\n", " ").replace("\r", " ").strip()
        response = openai_client.embeddings.create(
            input=[text], model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embeddings for text: {text[:200]}... Error: {e}")
        return False


def generate_batch_embeddings(text_list):
    try:
        response = openai_client.embeddings.create(
            input=text_list, model="text-embedding-3-small"
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        print(f"Batch embedding error: {e}")
        return [None] * len(text_list)


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


# Function to group by depoiq_id for Pinecone results
def group_by_depoIQ_ID(
    data, isSummary=False, isTranscript=False, isContradictions=False, isAdmission=False
):
    grouped_data = {}
    for entry in data:
        depoiq_id = entry["depoiq_id"]
        if depoiq_id not in grouped_data:
            print(f"🔍 Fetching Depo: {depoiq_id} fro DB\n\n\n")

            if isTranscript and not isSummary:
                grouped_data[depoiq_id] = getDepoTranscript(depoiq_id)
                return grouped_data

            if isSummary and not isTranscript:
                grouped_data[depoiq_id] = getDepoSummary(depoiq_id)
                return grouped_data

            if isContradictions and not isSummary and not isTranscript:
                grouped_data[depoiq_id] = getDepoContradictions(depoiq_id)
                return grouped_data

            if (
                isAdmission
                and not isSummary
                and not isTranscript
                and not isContradictions
            ):
                grouped_data[depoiq_id] = getDepoAdmissions(depoiq_id)
                return grouped_data

            grouped_data[depoiq_id] = getDepo(
                depoiq_id,
                isSummary=True,
                isTranscript=True,
                isContradictions=True,
                isAdmission=True,
            )
    print("Grouped Data fetched from DB \n\n")
    return grouped_data


def extract_keywords_and_synonyms(text):
    try:
        prompt = f"""
        You are an AI-specialised interactive bot working for a major US law firm. Your task is to perform two steps on the provided text:

        1. Extract all keywords from the text which may be relevant to a legal case. Only include keywords that are present in the text.
        2. For each keyword, generate 2 synonyms or closely related terms that could help in legal query searches.

        Format your output exactly like this:
        Keywords: keyword1, keyword2, keyword3, ...
        Synonyms:
        keyword1: synonym1, synonym2
        keyword2: synonym1, synonym2
        keyword3: synonym1, synonym2

        Text:
        {text}
        """

        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.3,
        )

        full_text = response.choices[0].message.content.strip()
        lines = full_text.splitlines()

        # Extract keywords list
        keywords_line = next(
            (line for line in lines if line.startswith("Keywords:")), ""
        )
        keywords = [
            kw.strip()
            for kw in keywords_line.replace("Keywords:", "").split(",")
            if kw.strip()
        ]

        # Extract synonyms into a flat list
        synonyms = []
        for line in lines:
            if (
                ":" in line
                and not line.startswith("Keywords:")
                and not line.startswith("Synonyms:")
            ):
                _, syns = line.split(":", 1)
                synonyms.extend([s.strip() for s in syns.split(",") if s.strip()])

        # convert keywords and synonyms into a string
        keywords = ", ".join(keywords)
        synonyms = ", ".join(synonyms)

        return keywords, synonyms

    except Exception as e:
        print(f"Error extracting keywords and synonyms: {e}")
        return [], []


def detect_category(user_query):
    query_lower = user_query.lower()

    # Mapping keywords to specific categories
    category_keywords = {
        "overview_summary": ["overview summary", "general overview", "case summary"],
        "topical_summary": ["topical summary", "specific topic", "subject summary"],
        "high_level_summary": [
            "high-level summary",
            "executive summary",
            "brief summary",
        ],
        "detailed_summary": [
            "detailed summary",
            "in-depth summary",
            "comprehensive analysis",
        ],
        "transcript": ["transcript", "conversation", "dialogue", "verbatim"],
        "admissions": ["admission", "confession", "agreement", "acknowledgment"],
        "contradictions": ["contradiction", "dispute", "conflict", "misstatement"],
    }

    # Check for matching keywords
    for category, keywords in category_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            print(f"\n\n\nDetected category: {category}\n\n\n")
            return category

    print("No specific category detected. Using all categories.")
    return "all"


# Function to query Pinecone and match return top k results
def query_pinecone(
    query_text,
    depo_ids=[],
    top_k=5,
    is_unique=False,
    category=None,
    is_detect_category=False,
):
    try:
        print(
            f"\n🔍 Querying Pinecone for : query_text-> '{query_text}' (depo_id->: {depo_ids})\n\n\n"
        )

        # Generate query embedding
        query_vector = generate_embedding(query_text)

        category_detect = detect_category(query_text) if is_detect_category else "all"

        if not query_vector:
            return json.dumps(
                {"status": "error", "message": "Failed to generate query embedding."}
            )

        # Define filter criteria (search within `depo_id`)
        transcript_category = "transcript"
        contradiction_category = "contradictions"
        admission_category = "admissions"
        filter_criteria = {}
        if depo_ids and len(depo_ids) > 0:
            # Add depo_id filter
            filter_criteria["depoiq_id"] = {"$in": depo_ids}
            if category:
                if category == "text":
                    filter_criteria["is_keywords"] = {"$ne": True}
                    filter_criteria["is_synonyms"] = {"$ne": True}
                elif category == "keywords":
                    filter_criteria["is_keywords"] = {"$eq": True}
                elif category == "synonyms":
                    filter_criteria["is_synonyms"] = {"$eq": True}
            if is_detect_category and category_detect != "all":
                filter_criteria["category"] = category_detect

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
                "depoiq_id": match["metadata"].get("depoiq_id"),
                "category": match["metadata"]["category"],
                "chunk_index": match["metadata"].get("chunk_index", None),
                "created_at": match["metadata"].get("created_at", None),
                "is_keywords": match["metadata"].get("is_keywords", False),
                "is_synonyms": match["metadata"].get("is_synonyms", False),
                "keywords": match["metadata"].get("keywords", None),
                "synonyms_keywords": match["metadata"].get("synonyms_keywords", None),
            }
            for match in results.get("matches", [])
        ]

        print(f"Matched Results: {matched_results} \n\n\n")

        # Group results by depoiq_id
        grouped_result = group_by_depoIQ_ID(
            matched_results, isSummary=True, isTranscript=True, isContradictions=True
        )

        # print(f"Grouped Results: {grouped_result} \n\n\n")

        custom_response = []  # Custom response

        for match in matched_results:
            depoiq_id = match["depoiq_id"]
            category = match["category"]
            isSummary = (
                category != transcript_category
                and category != contradiction_category
                and category != admission_category
            )
            isTranscript = category == transcript_category
            isContradictions = category == contradiction_category
            isAdmission = category == admission_category

            depo_data = grouped_result.get(depoiq_id, {})

            if isSummary:
                chunk_index = int(match["chunk_index"])
                db_category = snake_to_camel(category, isSummary=False)
                db_category_2 = snake_to_camel(category)
                summaries = depo_data.get("summary", {})
                is_keywords = match["is_keywords"]
                is_synonyms = match["is_synonyms"]
                keywords = match["keywords"]
                synonyms_keywords = match["synonyms_keywords"]

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
                        "depoiq_id": depoiq_id,
                        "category": category,
                        "chunk_index": chunk_index,
                        "text": text,
                        "is_keywords": is_keywords,
                        "is_synonyms": is_synonyms,
                        "keywords": keywords,
                        "synonyms_keywords": synonyms_keywords,
                    }
                )

            if isTranscript:
                chunk_index = match["chunk_index"]
                start_page, end_page = map(int, chunk_index.split("-"))
                transcripts = depo_data.get("transcript", {})
                is_keywords = match["is_keywords"]
                is_synonyms = match["is_synonyms"]
                keywords = match["keywords"]
                synonyms_keywords = match["synonyms_keywords"]
                pages = [
                    page
                    for page in transcripts
                    if start_page <= page["pageNumber"] <= end_page
                ]

                custom_response.append(
                    {
                        "depoiq_id": depoiq_id,
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
                        "is_keywords": is_keywords,
                        "is_synonyms": is_synonyms,
                        "keywords": keywords,
                        "synonyms_keywords": synonyms_keywords,
                    }
                )

            if isContradictions:
                chunk_index = match["chunk_index"]
                contradictions = depo_data.get("contradictions", {})
                is_keywords = match["is_keywords"]
                is_synonyms = match["is_synonyms"]
                keywords = match["keywords"]
                synonyms_keywords = match["synonyms_keywords"]

                # Extract the specific contradiction based on chunk_index
                contradiction = next(
                    (
                        item
                        for item in contradictions
                        if item["contradiction_id"] == chunk_index
                    ),
                    None,
                )

                if contradiction == None:
                    print(
                        f"Contradiction with ID {chunk_index} not found in depo_id {depoiq_id}"
                    )
                    continue

                reason = contradiction.get("reason")
                initial_question = contradiction.get("initial_question")
                initial_answer = contradiction.get("initial_answer")
                contradictory_responses = contradiction.get("contradictory_responses")
                contradictory_responses_string = ""
                for contradictory_response in contradictory_responses:
                    contradictory_responses_string += f"{contradictory_response['contradictory_question']} {contradictory_response['contradictory_answer']} "

                text = f"{initial_question} {initial_answer} {contradictory_responses_string} {reason}"

                print(f"\n\n\n\n\nText: {text} \n\n\n\n\n")
                custom_response.append(
                    {
                        "depoiq_id": depoiq_id,
                        "category": category,
                        "chunk_index": chunk_index,
                        "text": text,
                    }
                )

            if isAdmission:
                chunk_index = match["chunk_index"]
                admissions = depo_data.get("admissions", {})
                is_keywords = match["is_keywords"]
                is_synonyms = match["is_synonyms"]
                keywords = match["keywords"]
                synonyms_keywords = match["synonyms_keywords"]

                [admissions] = [
                    admissions
                    for admissions in admissions["text"]
                    if admissions["admission_id"] == chunk_index
                ]
                reason = admissions.get("reason")
                question = admissions.get("question")
                answer = admissions.get("answer")
                text = f"Question: {question} Answer: {answer} Reason:  {reason}"

                custom_response.append(
                    {
                        "depoiq_id": depoiq_id,
                        "category": category,
                        "chunk_index": chunk_index,
                        "text": text,
                        "is_keywords": is_keywords,
                        "is_synonyms": is_synonyms,
                        "keywords": keywords,
                        "synonyms_keywords": synonyms_keywords,
                    }
                )

        # text_list = [entry["text"] for entry in custom_response]
        # print(f"\n\n\n\n\nText List: {text_list} \n\n\n\n\n")

        # Instantiate the Pinecone client
        # pc = Pinecone(api_key=PINECONE_API_KEY)

        # from pinecone import RerankModel

        # # Perform reranking to get top_n results based on the query
        # reranked_results = pc.inference.rerank(
        #     model=RerankModel.Bge_Reranker_V2_M3,
        #     query=query_text,
        #     documents=text_list,
        #     top_n=top_k if is_unique else 8,
        #     return_documents=True,
        # )

        # rerank_response = []
        # for rank_match in reranked_results.data:
        #     index = rank_match.index
        #     response = custom_response[index]
        #     rerank_response.append(response)

        if is_unique:
            unique_set_response = {}
            for entry in custom_response:
                depoiq_id = entry["depoiq_id"]
                if depoiq_id not in unique_set_response:
                    unique_set_response[depoiq_id] = []
                unique_set_response[depoiq_id].append(entry)

            unique_set_response_keys = list(unique_set_response.keys())

            final_response = []

            # Ensure we have exactly 8 entries in final_response
            while len(final_response) < 8:
                for depoiq_id in unique_set_response_keys:
                    if len(final_response) >= 8:
                        break
                    try:
                        final_response.append(unique_set_response[depoiq_id].pop(0))
                        # if not unique_set_response[depoiq_id]:
                        #     unique_set_response_keys.remove(depoiq_id)
                    except IndexError:
                        continue

            return {
                "status": "success",
                "user_query": query_text,
                "depoiq_id": unique_set_response_keys,
                "metadata": final_response,
                "is_unique": is_unique,
            }
        else:

            grouped_result_keys = list(grouped_result.keys())

            response = {
                "status": "success",
                "user_query": query_text,
                "depoiq_id": grouped_result_keys,
                "metadata": custom_response,
                "is_unique": is_unique,
            }

        return response

    except Exception as e:
        print(f"Error querying Pinecone in query_pinecone: {e}")
        return {"status": "error", "message": str(e)}


def get_answer_from_AI(response):
    try:
        text_list = [entry["text"] for entry in response["metadata"]]
        # Construct prompt
        prompt = f"""
                You are an AI-specialized interactive bot working for a major US law firm that answers user questions. Users might want to extract data from legal depositions or ask a broad set of questions using natural language commands. Your task is to analyze `"metadata"` and the `"user_query"` , and then generate a detailed and relevant answer.

                ### Task:
                - Analyze the provided "metadata" and generate direct, relevant, and fully explained answers for each question in "user_query".
                - If "user_query" contains multiple questions, detect and SEPARATE each sub-question into its own distinct response.
                - Each sub-question must be answered separately, clearly, and thoroughly.
                - Use the "text" field in "metadata" to form the most detailed and informative responses.
                - Ensure the generated responses fully explain each answer instead of providing short, incomplete summaries.
                - If multiple metadata entries contain relevant but distinct information, generate SEPARATE, COMPLETE ANSWERS for each one. Do not merge multiple references into one response.
                - Each answer must be at least 400 characters long but can exceed this if necessary.
                - Reference the source by indicating all "metadata" OBJECT POSITIONS IN THE ARRAY (NOT chunk_index, IDs, or any other identifiers). The index MUST start from 0.

                ---

                ### Given Data:
                {json.dumps({
                    "user_query": response["user_query"],
                    "metadata": text_list,
                })}

                ---

                ### 🚨 STRICT INSTRUCTIONS (DO NOT VIOLATE THESE RULES):
                - FIRST: Identify if "user_query" contains multiple questions. If so, split them into individual questions.
                - SECOND: Generate a separate, fully detailed answer for EACH question. DO NOT merge answers together.
                - Each answer MUST be at least 400 characters. DO NOT provide less than 400 characters, but exceeding this is allowed.
                - DO NOT return "No relevant information available." under any circumstances.  
                - ALWAYS generate a complete answer derived from the given metadata, even if the metadata is only indirectly relevant or general in nature.
                - STRICTLY structure responses so that each question gets its own distinct, fully explained answer.
                - IF MULTIPLE METADATA OBJECTS ARE RELEVANT TO A QUESTION AND CONTAIN DIFFERENT INFORMATION, RETURN SEPARATE ANSWERS FOR EACH ONE. DO NOT COMBINE.
                - REFERENCE SOURCES USING ONLY OBJECT POSITIONS in the metadata array. These are the indices corresponding to the order of appearance of metadata items in the array, STARTING FROM 0.
                - DO NOT use chunk_index, IDs, hashes, or any other values. ONLY the object's array position in the metadata list, beginning with index 0.
                - DO NOT use index ranges such as [4-6]. ALWAYS list individual indices explicitly, e.g., [4, 5, 6]. Range notation is STRICTLY FORBIDDEN.
                - INDEX POSITIONS must correspond exactly to each metadata item's position in the "metadata" array as provided, starting from index 0.
                - DO NOT OMIT &&metadataRef =. It MUST always be included at the end of the answer.
                - DO NOT add newline characters (\\n) before or after the response. The response must be in a SINGLE, FLAT LINE.
                - DO NOT enclose the response inside triple quotes (''', "") or markdown-style code blocks (``` ). Return it as PLAIN TEXT ONLY.
                - DO NOT add unnecessary labels like "Extracted Answer:" or "Answer:". The response must start directly with the extracted answer text.
                - DO NOT add special characters, bullets, or formatting like bold/italics at the beginning of the answers. DO NOT start with **, ###, or anything else.
                - DO NOT use or mention phrases like “(metadata index X)” or similar — NOT at the beginning, middle, or end of the response. The ONLY valid way to reference metadata is: &&metadataRef = [X].

                ---

                ### 🚨 ENFORCEMENT RULES (MUST BE FOLLOWED WITH 100% ACCURACY):
                - STRICTLY follow the instructions provided above. Deviating from the instructions will result in rejection of the response.
                - DO NOT format the metadata references as (metadata index X) or similar — NEVER use that format in any part of the answer. The ONLY acceptable format is &&metadataRef = [X].
                - FORCE SEPARATE ANSWERS FOR EACH QUESTION IN THE QUERY. DO NOT COMBINE MULTIPLE QUESTIONS INTO A SINGLE RESPONSE UNDER ANY CIRCUMSTANCES.  
                - STRICTLY CHECK IF THE USER QUERY CONTAINS MULTIPLE QUESTIONS. IF YES, EACH SUB-QUESTION MUST BE IDENTIFIED AND ANSWERED SEPARATELY.  
                - FOR EVERY SUB-QUESTION, IF MULTIPLE METADATA OBJECTS CONTAIN RELEVANT INFORMATION, GENERATE A SEPARATE DETAILED ANSWER FOR EACH OBJECT.
                - DO NOT COMBINE MULTIPLE OBJECTS INTO ONE RESPONSE IF THEY CONTAIN DISTINCT INFORMATION. RETURN MULTIPLE ANSWERS ACCORDINGLY.
                - ENSURE THAT ALL metadataRef INDEX (start from 0) VALUES USED IN RESPONSES ARE LESS THAN THE TOTAL LENGTH (0-{len(text_list)-1}) OF THE METADATA ARRAY. DO NOT GENERATE INDEXES THAT EXCEED THE ARRAY'S ACTUAL LENGTH.
                - ENSURE THAT YOU DO NOT FORMAT THE METADATA REFERENCES INCORRECTLY. THE ONLY VALID FORMAT IS: &&metadataRef = [0, 3, 7]
                - EVERY SINGLE ANSWER MUST BE A MINIMUM OF 400 CHARACTERS. IF NEEDED, EXPAND THE EXPLANATION TO REACH THIS THRESHOLD.
                - DO NOT START ANY ANSWER WITH SPECIAL CHARACTERS OR FORMATTING LIKE **, ##, --, :, or bullet points. ANSWERS MUST START DIRECTLY WITH THE TEXT.
                - DO NOT USE OR APPEND ANY VARIATION OF (metadata index X) AT THE END OR IN ANY PART OF THE RESPONSE. ONLY &&metadataRef = [X] IS ALLOWED.

                ---

                ### 🚨 FINAL OUTPUT FORMAT (NO EXCEPTIONS, FOLLOW THIS EXACTLY):

                <Detailed Answer for Question 1> &&metadataRef = [0]  
                <Detailed Answer for Question 2> &&metadataRef = [4, 8]  
                <Detailed Answer for Question 3> &&metadataRef = [1, 3, 5, 7]  
                <Detailed Answer for Question 4> &&metadataRef = [8, 9]  

                - Example of Correct Output for Multiple Questions (DETAILED RESPONSES, NO NEWLINES, MINIMUM 400 CHARACTERS PER ANSWER):

                The expert witness deposed in this case was Mark Strassberg, M.D. He has significant experience in forensic psychiatry and has been involved in multiple legal cases, providing expert testimony. &&metadataRef = [0, 1]  

                Dr. Strassberg specializes in forensic and clinical practice. His expertise includes forensic psychiatry, medical evaluations, and expert testimony, with 85% of his forensic practice being for defendants. &&metadataRef = [2]  

                Dr. Strassberg was retained by Mr. Wilson for this case. Mr. Wilson has worked with Dr. Strassberg on multiple cases due to his specialization in forensic evaluations. &&metadataRef = [3]  

                Dr. Strassberg has worked with Mr. Wilson approximately 5 or 6 times before this case. However, he did not keep records of previous engagements and could not recall specific details of past collaborations. &&metadataRef = [4]
            """

        # Call GPT-3.5 API with deterministic parameters for consistent responses
        ai_response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI assistant that extracts precise answers from multiple transcript sources efficiently and consistently. Your answers must be deterministic and identical when given the same input.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=1000,  # Increase if needed to avoid truncation
            temperature=0,  # 🔐 Fully deterministic
            top_p=1,  # 🔐 Disable nucleus sampling
            frequency_penalty=0,  # 🔐 Avoid penalizing common phrases
            presence_penalty=0,  # 🔐 Avoid bias toward novelty
        )

        # Extract plain text response
        extracted_text = ai_response.choices[0].message.content.strip()

        print(f"AI Response: {extracted_text} \n\n\n\n")

        return extracted_text

    except Exception as e:
        print(f"Error querying Pinecone in get_answer_from_AI: {e}")
        return str(e)


def get_answer_from_AI_without_ref(response):
    try:
        text_list = [entry["text"] for entry in response["metadata"]]

        # Construct prompt
        prompt = f"""
                You are an AI-specialized interactive bot working for a major US law firm that answers user questions. Users might want to extract data from legal depositions or ask a broad set of questions using natural language commands. Your task is to analyze `\"metadata\"` and the `\"user_query\"` , and then generate a detailed and relevant answer.

                ### Task:
                - Analyze the provided "metadata" and generate a direct, relevant, and fully explained answer for the given "user_query".
                - Use the "text" field in "metadata" to form the most detailed and informative response.
                - Ensure the generated response fully explains the answer instead of providing a short, incomplete summary.
                - Each answer must be at least 400 characters long but can exceed this if necessary.

                ---

                ### Given Data:
                {json.dumps({
                    "user_query": response["user_query"],
                    "metadata": text_list,
                })}

                ---

                ### 🚨 STRICT INSTRUCTIONS (DO NOT VIOLATE THESE RULES):
                - Generate a single, fully detailed answer for the user query.
                - The answer MUST be at least 400 characters. DO NOT provide less than 400 characters, but exceeding this is allowed.
                - DO NOT return "No relevant information available." under any circumstances.  
                - ALWAYS generate a complete answer derived from the given metadata, even if the metadata is only indirectly relevant or general in nature.
                - DO NOT reference metadata positions, object indices, or any other identifiers.
                - DO NOT add newline characters (\n) before or after the response. The response must be in a SINGLE, FLAT LINE.
                - DO NOT enclose the response inside triple quotes (''', "") or markdown-style code blocks (``` ). Return it as PLAIN TEXT ONLY.
                - DO NOT add unnecessary labels like "Extracted Answer:" or "Answer:". The response must start directly with the extracted answer text.

                ---

                ### 🚨 FINAL OUTPUT FORMAT (NO EXCEPTIONS, FOLLOW THIS EXACTLY):

                <Detailed Answer for the User Query>  
            """

        # Call GPT-3.5 API with deterministic parameters for consistent responses
        ai_response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI assistant that extracts precise answers from multiple transcript sources efficiently and consistently. Your answers must be deterministic and identical when given the same input.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=1000,  # Increase if needed to avoid truncation
            temperature=0,  # 🔐 Fully deterministic
            top_p=1,  # 🔐 Disable nucleus sampling
            frequency_penalty=0,  # 🔐 Avoid penalizing common phrases
            presence_penalty=0,  # 🔐 Avoid bias toward novelty
        )

        # Extract plain text response without metadata references
        extracted_text = ai_response.choices[0].message.content.strip()

        print(f"AI Response: {extracted_text} \n\n\n\n")

        return extracted_text

    except Exception as e:
        print(f"Error querying Pinecone in get_answer_from_AI_without_ref: {e}")
        return str(e)


def get_answer_score_from_AI(question, answer):
    try:
        # Construct prompt
        prompt = f"""
                You are an AI language model tasked with evaluating how relevant a given answer is to a specific question. Please carefully compare the QUESTION and ANSWER below. Use logical, semantic, and contextual understanding to decide how well the answer addresses the question.

                Your response must be a SINGLE INTEGER between 1 and 10, based on the following criteria:

                - 10 = Perfectly relevant and fully answers the question.
                - 8–9 = Mostly relevant, may miss very minor details.
                - 6–7 = Moderately relevant, addresses part of the question but lacks key context.
                - 4–5 = Minimally relevant, some surface-level connection but largely incomplete.
                - 2–3 = Barely relevant, vague or off-topic.
                - 1 = Completely irrelevant.

                DO NOT EXPLAIN. DO NOT ADD ANY TEXT. RETURN ONLY A SINGLE INTEGER (1–10) ON A NEW LINE.

                QUESTION: {question}

                ANSWER: {answer}
                """

        # Call GPT-3.5 API with deterministic parameters for consistent responses
        ai_response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI assistant that extracts precise answers from multiple transcript sources efficiently and consistently. Your answers must be deterministic and identical when given the same input.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=10,
            temperature=0,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )

        # Extract plain text response
        extracted_text = ai_response.choices[0].message.content.strip()

        print(f"AI Response: {extracted_text} \n\n\n\n")

        return extracted_text

    except Exception as e:
        print(f"Error querying Pinecone get_answer_score_from_AI: {e}")
        return str(e)


def generate_legal_prompt_from_AI(user_query, count=6):
    try:
        # - Contradictions (e.g., contradiction, dispute, conflict, misstatement)
        prompt = f"""
        Acting as a strategic lawyer, analyze the following legal scenario or query:

        \"{user_query}\"

        Generate {count} strategically relevant questions in bullet points. Each question should specifically target only the following categories and include relevant keywords:

        - Admissions (e.g., admission, confession, agreement, acknowledgment)
        - Summaries
          - Overview Summary (e.g., overview summary, general overview, case summary)
          - High-Level Summary (e.g., high-level summary, executive summary, brief summary)
          - Topical Summary (e.g., topical summary, specific topic, subject summary)
          - Detailed Summary (e.g., detailed summary, in-depth summary, comprehensive analysis)
        - Transcripts (e.g., transcript, conversation, dialogue, verbatim)

        Present the questions clearly and concisely, formatted strictly as bullet points.
        """

        # Call GPT-4o API with specific parameters for consistency
        ai_response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a highly experienced strategic lawyer specializing in legal analysis and question generation. Provide concise and relevant questions only.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=500,
            temperature=0,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )

        # Extract and return the AI-generated response as an array of strings
        extracted_text = ai_response.choices[0].message.content.strip()
        question_list = [
            q.strip("- ").strip() for q in extracted_text.split("\n") if q.strip()
        ]
        print(f"AI Response: {question_list} \n\n")
        return question_list

    except Exception as e:
        print(f"Error generating legal questions: {e}")
        return str(e)


import json
import openai


def generate_detailed_answer(user_query, answers):
    try:
        # Extract necessary data
        text_list = [
            {
                "index": idx,
                "text": meta["text"],
                "question": answer["question"],
                "answer": answer["answer"],
                "category": meta["category"],
            }
            for idx, answer in enumerate(answers)
            for meta in answer["metadata"]
        ]

        # Construct a detailed prompt for AI analysis
        prompt = f"""
        You are an AI specialized in legal analysis, acting as a legal assistant providing a comprehensive case analysis for a major US law firm. Your role is to generate a detailed and well-supported legal response based on user queries. The user has asked the following query:

        '{user_query}'

        You are provided with relevant information, including summarized answers, extracted admissions, high-level summaries, detailed summaries, topical summaries, and relevant transcripts. Your task is to analyze all the data and provide an extensive and fully detailed answer.

        ### Instructions:
        - Provide a clear, structured, and highly detailed answer that directly addresses the user's query.
        - Expand on key concepts, legal theories, and liability principles using all relevant information from the available data.
        - Provide accurate legal analysis without inserting opinions or making assumptions.
        - Ensure a professional and neutral tone, maintaining legal accuracy.
        - Reference relevant excerpts and evidence from the provided metadata using a clear reference format `&&metadataRef = [X]`, where X represents the metadata array index. 
        - Multiple references must be separated by commas, for example: `&&metadataRef = [0, 3, 7]`.
        - Avoid unnecessary repetition and maintain clarity by logically connecting facts, admissions, and legal context.

        ### Special Rules for Metadata Referencing:
        - Provide the most applicable metadata references by indicating the correct metadata index position(s).
        - NEVER use any other reference format such as `(metadata index X)` or `chunk_index`. The only valid format is `&&metadataRef = [X]`.

        ### Srict Formatting Rules:
        - Do not write answer in points or bullet points.
        - Do not add ** or any other special characters at the beginning or end of the answer.
        - Do not use any special formatting like bold, italics, or code blocks.



        ### Available Data:
        {json.dumps(text_list, indent=2)}

        ---
        
        Provide a complete, clear, and highly detailed response below, following all instructions without (**) or any other special characters:
        """

        # Generate AI response using GPT-4
        ai_response = openai.Client().chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a legal AI assistant specialized in providing detailed and well-supported legal analysis. Ensure responses are thorough, clear, and logically structured based on the provided data. Provide factual legal analysis without personal opinions or special character formatting.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=4000,
            temperature=0,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )

        # Extract the detailed response
        extracted_text = ai_response.choices[0].message.content.strip()

        print(f"AI Response: {extracted_text}\n\n\n")

        return extracted_text

    except Exception as e:
        print(f"Error generating detailed answer: {e}")
        return str(e)


# Function to get Depo from DepoIQ_ID
def getDepo(
    depoiq_id,
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
                    contradiction_id
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
                "depoId": depoiq_id,
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

        print("Response Status Code:", response.status_code)

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
def getDepoSummary(depoiq_id):
    """Get Depo Summary from DepoIQ_ID"""
    try:
        depo = getDepo(
            depoiq_id,
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
def getDepoTranscript(depoiq_id):
    """Get Depo Transcript from DepoIQ_ID"""
    try:
        depo = getDepo(
            depoiq_id,
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
def getDepoContradictions(depoiq_id):
    """Get Depo Contradictions from DepoIQ_ID"""
    try:
        depo = getDepo(
            depoiq_id,
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
def getDepoAdmissions(depoiq_id):
    """Get Depo Admissions from DepoIQ_ID"""
    try:
        depo = getDepo(
            depoiq_id,
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
    if not value:
        print("⚠️ extract_text received empty input.")
        return ""

    if value.startswith("<"):
        try:
            soup = BeautifulSoup(value, "html.parser")
            return soup.get_text().strip()
        except Exception as e:
            print(f"⚠️ Error parsing HTML: {e}")
            return ""
    return str(value).strip()


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
def check_existing_entry(depoiq_id, category, chunk_index):
    """Checks if already exists in Pinecone."""
    try:
        existing_entries = depoIndex.query(
            vector=np.random.rand(1536).tolist(),  # Use a dummy query vector
            filter={
                "depoiq_id": depoiq_id,
                "chunk_index": chunk_index,
                "category": category,
            },
            top_k=1,
        )
        return bool(existing_entries["matches"])
    except Exception as e:
        print(f"⚠️ Error querying Pinecone in check_exisiting_entry: {e}")
        return False


# Function to store summaries in Pinecone with embeddings
def store_summaries_in_pinecone(depoiq_id, category, text_chunks):
    """Stores text chunks in Pinecone with embeddings."""
    try:
        vectors_to_upsert = []
        skipped_chunks = []

        for chunk_index, chunk_value in enumerate(text_chunks):

            print(
                f"\n\n\n\n🔹 Processing chunk {chunk_index + 1} of {category}\n\n\n\n"
            )

            if not chunk_value or not isinstance(chunk_value, str):
                print(f"⚠️ Skipping invalid text chunk: {chunk_index}\n\n\n\n")
                continue

            # Check if summary already exists
            if check_existing_entry(depoiq_id, category, chunk_index):
                skipped_chunks.append(chunk_index)
                print(f"⚠️ Summary already exists: {chunk_index}, skipping...\n\n\n\n")
                continue

            # Extract keywords and synonyms
            keywords, synonyms_keywords = extract_keywords_and_synonyms(chunk_value)

            print(
                f"\n\n\n\nExtracted Keywords: for {chunk_value} ->\n\n {keywords} \n\n\n\n\n"
            )

            # Generate embedding
            embedding_text = generate_embedding(chunk_value)
            embedding_keywords = generate_embedding(keywords)
            embedding_synonyms_keywords = generate_embedding(synonyms_keywords)

            # Generate embeddings in batch
            # embeddings = generate_batch_embeddings(
            #     [chunk_value, " ".join(keywords), " ".join(synonyms_keywords)]
            # )
            # embedding_text, embedding_keywords, embedding_synonyms_keywords = embeddings

            # if not any(embedding):  # Check for all zero vectors
            #     print(f"⚠️ Skipping zero-vector embedding for: {chunk_index}\n\n\n\n")
            #     continue

            # Metadata
            # metadata = {
            #     "depoiq_id": depoiq_id,
            #     "category": category,
            #     "chunk_index": chunk_index,
            #     "created_at": datetime.now().isoformat(),
            # }

            created_at = datetime.now().isoformat()

            # Add to batch
            vectors_to_upsert.extend(
                [
                    {
                        "id": str(uuid.uuid4()),
                        "values": embedding_text,
                        "metadata": {
                            "depoiq_id": depoiq_id,
                            "category": category,
                            "chunk_index": chunk_index,
                            "created_at": created_at,
                        },
                    },
                    {
                        "id": str(uuid.uuid4()),
                        "values": embedding_keywords,
                        "metadata": {
                            "depoiq_id": depoiq_id,
                            "category": category,
                            "chunk_index": chunk_index,
                            "created_at": created_at,
                            "is_keywords": True,
                            "keywords": keywords,
                        },
                    },
                    {
                        "id": str(uuid.uuid4()),
                        "values": embedding_synonyms_keywords,
                        "metadata": {
                            "depoiq_id": depoiq_id,
                            "category": category,
                            "chunk_index": chunk_index,
                            "created_at": created_at,
                            "is_synonyms": True,
                            "synonyms_keywords": synonyms_keywords,
                        },
                    },
                ]
            )

        # Bulk upsert to Pinecone
        if vectors_to_upsert:

            depoIndex.upsert(vectors=vectors_to_upsert)
            print(
                f"✅ Successfully inserted {len(vectors_to_upsert)} summaries in Pinecone."
            )

        return len(vectors_to_upsert), skipped_chunks
    except Exception as e:
        print(f"Error storing summaries in Pinecone: {e}")
        return 0, []


def store_transcript_lines_in_pinecone(depoiq_id, category, transcript_data):
    vectors_to_upsert = []
    skipped_chunks = []
    chunk_page_range = 5
    max_chunk_upsert = 50
    total_upserted = 0

    for i in range(0, len(transcript_data), chunk_page_range):
        grouped_pages = transcript_data[i : i + chunk_page_range]

        chunk_index = (
            f"{grouped_pages[0]['pageNumber']}-{grouped_pages[-1]['pageNumber']}"
        )

        # check if already exists in pinecone
        if check_existing_entry(
            depoiq_id,
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

        # Extract keywords and synonyms
        keywords, synonyms_keywords = extract_keywords_and_synonyms(trimmed_text)

        print(
            f"\n\n\n\nExtracted Keywords: for {trimmed_text} ->\n\n {keywords} \n\n\n\n\n"
        )

        # Generate embedding
        embedding_text = generate_embedding(trimmed_text)
        embedding_keywords = generate_embedding(keywords)
        embedding_synonyms_keywords = generate_embedding(synonyms_keywords)

        # Generate embeddings in batch
        # embeddings = generate_batch_embeddings(
        #     [trimmed_text, " ".join(keywords), " ".join(synonyms_keywords)]
        # )
        # embedding_text, embedding_keywords, embedding_synonyms_keywords = embeddings

        # Debug: Check if embedding is valid
        # if not any(embedding):
        #     print(f"⚠️ Skipping empty embedding for pages {i+1}-{i+chunk_page_range}")
        #     continue

        # Add the actual transcript text in metadata for retrieval
        # metadata = {
        #     "depoiq_id": depoiq_id,
        #     "category": category,
        #     "chunk_index": chunk_index,
        #     "created_at": datetime.now().isoformat(),
        # }

        created_at = datetime.now().isoformat()

        vectors_to_upsert.extend(
            [
                {
                    "id": str(uuid.uuid4()),  # Unique ID
                    "values": embedding_text,  # Embedding vector for text
                    "metadata": {
                        "depoiq_id": depoiq_id,
                        "category": category,
                        "chunk_index": chunk_index,
                        "created_at": created_at,
                    },
                },
                {
                    "id": str(uuid.uuid4()),  # Unique ID
                    "values": embedding_keywords,  # Embedding vector for keywords
                    "metadata": {
                        "depoiq_id": depoiq_id,
                        "category": category,
                        "chunk_index": chunk_index,
                        "created_at": created_at,
                        "is_keywords": True,
                        "keywords": keywords,
                    },
                },
                {
                    "id": str(uuid.uuid4()),  # Unique ID
                    "values": embedding_synonyms_keywords,  # Embedding vector for synonyms
                    "metadata": {
                        "depoiq_id": depoiq_id,
                        "category": category,
                        "chunk_index": chunk_index,
                        "created_at": created_at,
                        "is_synonyms": True,
                        "synonyms_keywords": synonyms_keywords,
                    },
                },
            ]
        )

        if len(vectors_to_upsert) >= max_chunk_upsert:
            depoIndex.upsert(vectors=vectors_to_upsert)
            total_upserted += len(vectors_to_upsert)
            print(
                f"✅ Successfully inserted {total_upserted} transcript chunks in Pinecone.\n\n"
            )
            vectors_to_upsert = []

    if vectors_to_upsert:
        depoIndex.upsert(vectors=vectors_to_upsert)
        total_upserted += len(vectors_to_upsert)
        print(
            f"✅ Successfully inserted {total_upserted} transcript chunks in Pinecone."
        )

    return total_upserted, skipped_chunks


def store_contradictions_in_pinecone(depoiq_id, category, contradictions_data):
    vectors_to_upsert = []
    skipped_chunks = []
    max_chunk_upsert = 50
    total_upserted = 0

    try:
        print(
            f"🔹 Storing contradictions for depoiq_id: {depoiq_id} {len(contradictions_data)}"
        )

        for index in range(0, len(contradictions_data)):
            contradictions = contradictions_data[index]
            chunk_index = contradictions.get("contradiction_id", None)

            print(f"\n\n 🔹 Storing contradictions: {chunk_index} \n\n\n\n")

            if chunk_index == None:
                print(
                    f"⚠️ Skipping contradictions: because contradiction_id is None \n\n\n\n"
                )
                skipped_chunks.append(chunk_index)
                continue

            # check if already exists in pinecone
            if check_existing_entry(
                depoiq_id,
                category,
                chunk_index,
            ):
                skipped_chunks.append(chunk_index)
                print(
                    f"⚠️ Contradiction already exists for pages {chunk_index}, skipping..."
                )
                continue

            # Extract contradictions text, ensuring each line is properly formatted
            reason = contradictions.get("reason")
            contradictory_responses = contradictions.get("contradictory_responses")
            initial_question = contradictions.get("initial_question")
            initial_answer = contradictions.get("initial_answer")
            contradictory_responses_string = ""
            for contradictory_response in contradictory_responses:
                contradictory_responses_string += f"{contradictory_response['contradictory_question']} {contradictory_response['contradictory_answer']} "

            contradiction_text = f"{initial_question} {initial_answer} {contradictory_responses_string} {reason}"

            # Extract keywords and synonyms
            keywords, synonyms_keywords = extract_keywords_and_synonyms(
                contradiction_text
            )

            print(
                f"\n\n\n\nExtracted Keywords: for {contradiction_text} ->\n\n {keywords} \n\n\n\n\n"
            )

            # Generate embedding
            embedding_text = generate_embedding(contradiction_text)
            embedding_keywords = generate_embedding(keywords)
            embedding_synonyms_keywords = generate_embedding(synonyms_keywords)

            if not any(embedding_text):
                print(
                    f"⚠️ Skipping empty embedding for contradictions: {contradiction_text}"
                )
                continue

            created_at = datetime.now().isoformat()

            # Add to batch
            vectors_to_upsert.extend(
                [
                    {
                        "id": str(uuid.uuid4()),  # Unique ID
                        "values": embedding_text,  # Embedding vector for text
                        "metadata": {
                            "depoiq_id": depoiq_id,
                            "category": category,
                            "chunk_index": chunk_index,
                            "created_at": created_at,
                        },
                    },
                    {
                        "id": str(uuid.uuid4()),  # Unique ID
                        "values": embedding_keywords,  # Embedding vector for keywords
                        "metadata": {
                            "depoiq_id": depoiq_id,
                            "category": category,
                            "chunk_index": chunk_index,
                            "created_at": created_at,
                            "is_keywords": True,
                            "keywords": keywords,
                        },
                    },
                    {
                        "id": str(uuid.uuid4()),  # Unique ID
                        "values": embedding_synonyms_keywords,  # Embedding vector for synonyms
                        "metadata": {
                            "depoiq_id": depoiq_id,
                            "category": category,
                            "chunk_index": chunk_index,
                            "created_at": created_at,
                            "is_synonyms": True,
                            "synonyms_keywords": synonyms_keywords,
                        },
                    },
                ]
            )

            if len(vectors_to_upsert) >= max_chunk_upsert:
                depoIndex.upsert(vectors=vectors_to_upsert)
                total_upserted += len(vectors_to_upsert)
                print(
                    f"✅ Successfully inserted {total_upserted} contradictions in Pinecone.\n\n"
                )
                vectors_to_upsert = []

        if vectors_to_upsert:
            depoIndex.upsert(vectors=vectors_to_upsert)
            total_upserted += len(vectors_to_upsert)
            print(
                f"✅ Successfully inserted {total_upserted} contradictions in Pinecone."
            )

        return total_upserted, skipped_chunks

    except Exception as e:
        print(f"🔹 Error in store_contradictions_in_pinecone: {e}")
        return 0, []


def store_admissions_in_pinecone(depoiq_id, category, admissions_data):
    vectors_to_upsert = []
    skipped_chunks = []
    max_chunk_upsert = 50
    total_upserted = 0

    try:
        print(
            f"🔹 Storing admissions for depoiq_id: {depoiq_id} {len(admissions_data)}"
        )

        for index in range(0, len(admissions_data)):
            admissions = admissions_data[index]
            chunk_index = admissions.get("admission_id")

            if chunk_index == None:
                print(
                    f"⚠️ Skipping admissions: {admissions} because chunk_index is None \n\n\n\n"
                )
                continue

            print(f"\n\n 🔹 Storing admissions: -> {index} ->  {chunk_index} \n\n\n\n")

            # check if already exists in pinecone
            if check_existing_entry(
                depoiq_id,
                category,
                chunk_index,
            ):
                skipped_chunks.append(chunk_index)
                print(
                    f"⚠️ Admission already exists for pages {chunk_index}, skipping..."
                )
                continue

            # Extract admissions text, ensuring each line is properly formatted
            reason = admissions.get("reason")
            question = admissions.get("question")
            answer = admissions.get("answer")
            admission_text = f"{question} {answer} {reason}"

            print(f"\n\n\n\n🔹 Admission text: {admission_text}\n\n\n\n\n")

            # Extract keywords and synonyms
            keywords, synonyms_keywords = extract_keywords_and_synonyms(admission_text)

            print(
                f"\n\n\n\nExtracted Keywords: for {admission_text} ->\n\n {keywords} \n\n\n\n\n"
            )

            # Generate embedding
            embedding_text = generate_embedding(admission_text)
            embedding_keywords = generate_embedding(keywords)
            embedding_synonyms_keywords = generate_embedding(synonyms_keywords)

            # Generate embeddings in batch
            # embeddings = generate_batch_embeddings(
            #     [admission_text, " ".join(keywords), " ".join(synonyms_keywords)]
            # )
            # embedding_text, embedding_keywords, embedding_synonyms_keywords = embeddings

            # if not any(embedding):
            #     print(f"⚠️ Skipping empty embedding for admissions: {admission_text}")
            #     continue

            # Add the actual transcript text in metadata for retrieval
            # metadata = {
            #     "depoiq_id": depoiq_id,
            #     "category": category,
            #     "chunk_index": chunk_index,
            #     "created_at": datetime.now().isoformat(),
            # }

            created_at = datetime.now().isoformat()

            # Add to batch
            vectors_to_upsert.extend(
                [
                    {
                        "id": str(uuid.uuid4()),  # Unique ID
                        "values": embedding_text,  # Embedding vector for text
                        "metadata": {
                            "depoiq_id": depoiq_id,
                            "category": category,
                            "chunk_index": chunk_index,
                            "created_at": created_at,
                        },
                    },
                    {
                        "id": str(uuid.uuid4()),  # Unique ID
                        "values": embedding_keywords,  # Embedding vector for keywords
                        "metadata": {
                            "depoiq_id": depoiq_id,
                            "category": category,
                            "chunk_index": chunk_index,
                            "created_at": created_at,
                            "is_keywords": True,
                            "keywords": keywords,
                        },
                    },
                    {
                        "id": str(uuid.uuid4()),  # Unique ID
                        "values": embedding_synonyms_keywords,  # Embedding vector for synonyms
                        "metadata": {
                            "depoiq_id": depoiq_id,
                            "category": category,
                            "chunk_index": chunk_index,
                            "created_at": created_at,
                            "is_synonyms": True,
                            "synonyms_keywords": synonyms_keywords,
                        },
                    },
                ]
            )

            if len(vectors_to_upsert) >= max_chunk_upsert:
                depoIndex.upsert(vectors=vectors_to_upsert)
                total_upserted += len(vectors_to_upsert)
                print(
                    f"✅ Successfully inserted {total_upserted} admissions in Pinecone."
                )
                vectors_to_upsert = []

        if vectors_to_upsert:
            depoIndex.upsert(vectors=vectors_to_upsert)
            total_upserted += len(vectors_to_upsert)
            print(
                f"✅ Successfully inserted {total_upserted} admissions in Pinecone.\n\n\n"
            )

        return total_upserted, skipped_chunks

    except Exception as e:
        print(f"🔹 Error in store_admissions_in_pinecone: {e}")
        return 0, []


# Function to generate add depo summaries to pinecone
def add_depo_summaries(depo_summary, depoiq_id):
    try:
        excluded_keys = ["visualization"]
        total_inserted = 0
        skipped_sub_categories = {}

        print(f"🔹 Adding summaries for depoiq_id: {depoiq_id}")

        for key, value in depo_summary.items():
            if key in excluded_keys:
                continue

            category = camel_to_snake(key)  # Convert key to category format
            text = extract_text(value["text"])  # Extract clean text
            text_chunks = split_into_chunks(text)  # Split into paragraphs

            inserted_count, skipped_chunks = store_summaries_in_pinecone(
                depoiq_id, category, text_chunks
            )

            total_inserted += inserted_count
            skipped_sub_categories[category] = skipped_chunks

            print(f"✅ Successfully inserted {inserted_count} summaries for {category}")

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
                "depoiq_id": depoiq_id,
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


def add_depo_transcript(transcript_data, depoiq_id):
    try:
        print(f"🔹 Adding transcripts for depoiq_id: {depoiq_id}")
        if not transcript_data:
            return {
                "status": "warning",
                "message": "No transcript data found.",
                "data": {
                    "depoiq_id": depoiq_id,
                    "total_inserted": 0,
                    "skipped_details": [],
                    "skipped_count": 0,
                },
            }

        category = "transcript"  # Default category for transcripts

        # Store transcript data
        inserted_transcripts, skipped_transcripts = store_transcript_lines_in_pinecone(
            depoiq_id, category, transcript_data
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
                "depoiq_id": depoiq_id,
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


def add_depo_contradictions(contradictions_data, depoiq_id):
    try:
        print(f"🔹 Adding contradictions for depoiq_id: {depoiq_id}")

        if not contradictions_data:
            return {
                "status": "warning",
                "message": "No contradictions data found.",
                "data": {
                    "depoiq_id": depoiq_id,
                    "total_inserted": 0,
                    "skipped_details": [],
                    "skipped_count": 0,
                },
            }

        category = "contradictions"

        # return {
        #     "status": "warning",
        #     "message": "Waiting for backend change",
        #     "data": {
        #         "depoiq_id": depoiq_id,
        #         "total_inserted": 0,
        #         "skipped_details": [],
        #         "skipped_count": 0,
        #     },
        # }

        # Store contradictions data
        inserted_contradictions, skipped_contradictions = (
            store_contradictions_in_pinecone(depoiq_id, category, contradictions_data)
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
                "depoiq_id": depoiq_id,
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


def add_depo_admissions(admissions_data, depoiq_id):
    try:
        print(f"🔹 Adding admissions for depoiq_id: {depoiq_id}\n\n")

        if not admissions_data:
            return {
                "status": "warning",
                "message": "No admissions data found.",
                "data": {
                    "depoiq_id": depoiq_id,
                    "total_inserted": 0,
                    "skipped_details": [],
                    "skipped_count": 0,
                },
            }

        category = "admissions"

        # Store admissions data
        inserted_admissions, skipped_admissions = store_admissions_in_pinecone(
            depoiq_id, category, admissions_data=admissions_data["text"]
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
                "depoiq_id": depoiq_id,
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


@app.route("/depo/<string:depoiq_id>", methods=["GET"])
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
        description: Optional category filter (e.g., summary, transcript, contradictions, admissions)
    responses:
      200:
        description: Returns the success message
      500:
        description: Internal server error
    """
    try:

        # get category from query params
        category = request.args.get("category", None)

        if category:
            allowed_categories = [
                "summary",
                "transcript",
                "contradictions",
                "admissions",
            ]
            if category not in allowed_categories:
                return (
                    jsonify(
                        {
                            "error": "Invalid category. Should be 'summary', 'transcript', 'contradictions', or 'admissions'."
                        }
                    ),
                    400,
                )

        depo = getDepo(
            depoiq_id,
            isSummary=not category or category == "summary",
            isTranscript=not category or category == "transcript",
            isContradictions=not category or category == "contradictions",
            isAdmission=not category or category == "admissions",
        )
        print(f"🔹 Depo data: {depo}")
        return jsonify(depo), 200

    except Exception as e:
        return (
            jsonify(
                {"error": "Something went wrong in get_depo_by_Id", "details": str(e)}
            ),
            500,
        )


@app.route("/depo/add/<string:depoiq_id>", methods=["GET"])
def add_depo(depoiq_id):
    """
    Get Summaries & Transcript & Contradictions & Admissions from Depo and store into Pinecone
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
        description: Optional category filter (e.g., summary, transcript, contradictions, admissions)
    responses:
      200:
        description: Returns the success message
      500:
        description: Internal server error
    """
    try:
        category = request.args.get("category", None)
        allowed_categories = ["summary", "transcript", "contradictions", "admissions"]

        if category and category not in allowed_categories:
            return (
                jsonify(
                    {
                        "error": "Invalid category. Should be 'summary', 'transcript', 'contradictions', or 'admissions'."
                    }
                ),
                400,
            )

        # Fetch depo data
        depo = getDepo(
            depoiq_id,
            isSummary=not category or category == "summary",
            isTranscript=not category or category == "transcript",
            isContradictions=not category or category == "contradictions",
            isAdmission=not category or category == "admissions",
        )

        responses = {}
        status_list = []

        # Dynamically call processing functions based on category
        if not category or category == "summary":
            summary_data = depo.get("summary", {})
            summary_response = add_depo_summaries(summary_data, depoiq_id)
            responses["summary"] = summary_response["data"]
            status_list.append(summary_response["status"])

        if not category or category == "transcript":
            transcript_data = depo.get("transcript", [])
            transcript_response = add_depo_transcript(transcript_data, depoiq_id)
            responses["transcript"] = transcript_response["data"]
            status_list.append(transcript_response["status"])

        if not category or category == "contradictions":
            contradictions_data = depo.get("contradictions", [])
            contradiction_response = add_depo_contradictions(
                contradictions_data, depoiq_id
            )
            responses["contradictions"] = contradiction_response["data"]
            status_list.append(contradiction_response["status"])

        if not category or category == "admissions":
            admissions_data = depo.get("admissions", [])
            admissions_response = add_depo_admissions(admissions_data, depoiq_id)
            responses["admissions"] = admissions_response["data"]
            status_list.append(admissions_response["status"])

        # Determine overall status
        if all(status == "success" for status in status_list):
            overall_status = "success"
        elif any(status == "warning" for status in status_list):
            overall_status = "warning"
        else:
            overall_status = "error"

        message_parts = []
        for key, value in responses.items():
            if "total_inserted" in value:
                message_parts.append(f"{value['total_inserted']} {key}")

        message = f"Stored {' and '.join(message_parts)} in Pinecone for depoiq_id {depoiq_id}."

        return (
            jsonify(
                {
                    "status": overall_status,
                    "depoiq_id": depoiq_id,
                    "message": message,
                    **responses,
                }
            ),
            200,
        )

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
    Talk to depo summaries & transcript by depoiq_ids
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
              required: false
              value: "all"
              enum: ["text", "keywords", "synonyms", "all"]
            is_unique:
              type: boolean
              required: false
    responses:
      200:
        description: Returns the success message
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid request, JSON body required"}), 400

        depoiq_ids = data.get("depoiq_ids")
        user_query = data.get("user_query")
        is_unique = data.get("is_unique", False)
        category = data.get("category", "text")

        allowed_categories = ["text", "keywords", "synonyms", "all"]
        if category not in allowed_categories:
            return (
                jsonify(
                    {
                        "error": "Invalid category. Should be 'text' , 'keywords' , 'synonyms', 'all'."
                    }
                ),
                400,
            )

        if depoiq_ids:
            if not user_query:
                return jsonify({"error": "Missing user_query"}), 400

            if len(depoiq_ids) == 0:
                return jsonify({"error": "Missing depoiq_ids list"}), 400

            if len(depoiq_ids) > 8:
                return jsonify({"error": "Too many depoiq_ids, max 8"}), 400
            # check all ids are valid and mongo Id
            for depo_id in depoiq_ids:
                if not ObjectId.is_valid(depo_id):
                    return jsonify({"error": "Invalid depo_id " + depo_id}), 400
            depoIQ_IDs_array_length = len(depoiq_ids) * 3
        else:
            depoIQ_IDs_array_length = 24

        top_k = depoIQ_IDs_array_length if is_unique else 8

        # ✅ Query Pinecone Safely
        query_pinecone_response = query_pinecone(
            user_query,
            depoiq_ids,
            top_k=top_k,
            is_unique=is_unique,
            category=category,
            is_detect_category=False,
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

            query_pinecone_response["answer"] = str(
                ai_resposne
            )  # Add AI response to the pinecone response
            query_pinecone_response["category"] = category

        return jsonify(query_pinecone_response), 200

    except Exception as e:
        return (
            jsonify(
                {"error": "Something went wrong in talk_summary", "details": str(e)}
            ),
            500,
        )


@app.route("/depo/answer_validator", methods=["POST"])
def answer_validator():
    """
    Answer Validate for Depo
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
              value: "all"
              enum: ["text", "keywords", "synonyms", "all"]
    responses:
      200:
        description: Returns the success message
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid request, JSON body required"}), 400

        questions = data.get("questions")
        depoiq_id = data.get("depoiq_id")
        category = data.get("category", "all")
        is_download = data.get("is_download", True)
        top_k = data.get("top_k", 10)

        if not questions:
            return jsonify({"error": "Missing questions"}), 400

        if not depoiq_id:
            return jsonify({"error": "Missing depoiq_id"}), 400

        allowed_categories = ["text", "keywords", "synonyms", "all"]
        answers_response = []
        if category not in allowed_categories:
            return (
                jsonify(
                    {
                        "error": "Invalid category. Should be 'text' , 'keywords' , 'synonyms', 'all'."
                    }
                ),
                400,
            )

        for question in questions:
            print(f"\n\n🔹 Question: {question}\n\n\n\n")
            query_pinecone_response = query_pinecone(
                question,
                [depoiq_id],
                top_k=top_k,
                is_unique=False,
                category=category,
                is_detect_category=False,
            )

            print(
                f"\n\n✅ Query Pinecone Response: {json.dumps(query_pinecone_response, indent=2)}\n\n\n\n"
            )

            if query_pinecone_response["status"] == "Not Found":
                print(
                    f"\n\n🔍 No matches found for the query and return error \n\n\n\n"
                )
                answers_response.append(
                    {
                        "question": question,
                        "answer": "No matches found for the query",
                        "score": 0,
                    }
                )
                # skip the next steps in for loop
                continue

            else:
                ai_resposne = get_answer_from_AI(query_pinecone_response)

                score = get_answer_score_from_AI(
                    question=question, answer=str(ai_resposne)
                )

                answers_response.append(
                    {
                        "question": question,
                        "answer": str(ai_resposne),
                        "score": int(score),
                    }
                )

        average_score = round(
            sum([answer["score"] for answer in answers_response])
            / len(answers_response),
            2,
        )

        if is_download:
            # Create CSV in memory
            si = io.StringIO()
            writer = csv.writer(si)
            writer.writerow(
                [
                    "No",
                    "Question",
                    "Answer",
                    "Score",
                ]
            )  # Header

            for ans in answers_response:
                writer.writerow(
                    [
                        answers_response.index(ans) + 1,
                        ans["question"],
                        ans["answer"],
                        ans["score"],
                    ]
                )
            writer.writerow(["", "", "", "", ""])  # Empty row
            writer.writerow(
                ["", "Average Score", "Category", "Depoiq_id", "Searching AI"]
            )  # Footer row
            writer.writerow(
                [
                    "",
                    average_score,
                    category,
                    depoiq_id,
                    "gpt-4o",
                ]
            )  # Footer row

            output = si.getvalue()
            si.close()

            # Return response with CSV download
            return Response(
                output,
                mimetype="text/csv",
                headers={
                    "Content-Disposition": f"attachment;filename=answers_validator_{category}_{depoiq_id}.csv"
                },
            )

        else:
            # Return JSON response

            response = {
                "depoiq_id": depoiq_id,
                "category": category,
                "answers": answers_response,
                "overall_score": average_score,
            }

            return (
                jsonify(response),
                200,
            )

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


@app.route("/depo/ask-ami-agent", methods=["POST"])
def ask_ami_agent():
    """
    Talk to depo summaries & transcript by depoiq_ids
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
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid request, JSON body required"}), 400

        user_query = data.get("user_query")
        depoiq_ids = data.get("depoiq_ids")

        questions = generate_legal_prompt_from_AI(user_query, count=6)
        print(f"\n\n🔹 Question List: {questions}\n\n\n\n")

        if not depoiq_ids:
            return jsonify({"error": "Missing depoiq_id"}), 400

        answers_response = []

        for question in questions:
            print(f"\n\n🔹 Question: {question}\n\n\n\n")
            query_pinecone_response = query_pinecone(
                question,
                depoiq_ids,
                top_k=8,
                is_unique=False,
                category="text",
                is_detect_category=True,
            )

            print(
                f"\n\n✅ Query Pinecone Response: {json.dumps(query_pinecone_response, indent=2)}\n\n\n\n"
            )

            if query_pinecone_response["status"] == "Not Found":
                print(
                    f"\n\n🔍 No matches found for the query and return error \n\n\n\n"
                )
                answers_response.append(
                    {
                        "question": question,
                        "answer": "No matches found for the query",
                    }
                )
                continue

            else:
                ai_resposne = get_answer_from_AI_without_ref(query_pinecone_response)

                answers_response.append(
                    {
                        "question": question,
                        "answer": str(ai_resposne),
                        "metadata": query_pinecone_response.get("metadata"),
                        "depoiq_id": query_pinecone_response.get("depoiq_id"),
                    }
                )
        detailed_answer = generate_detailed_answer(
            user_query=user_query, answers=answers_response
        )

        response = {
            "detailed_answer": detailed_answer,
            "user_query": user_query,
            "depoiq_ids": depoiq_ids,
            "answers": answers_response,
        }

        return (
            jsonify(response),
            200,
        )

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


if __name__ == "__main__":
    app.run(debug=True)
