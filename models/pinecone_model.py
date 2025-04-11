from pinecone import Pinecone, ServerlessSpec
from datetime import datetime
from models.openAi_model import OpenAIClientModel
from helperFunctions.utils import cPrint
from helperFunctions.utils import (
    cPrint,
    detect_category,
    snake_to_camel,
    extract_text,
    split_into_chunks,
)
import numpy as np
import os
import time
import uuid
import json


class PineConeModel:
    index_name = os.getenv("DEPO_INDEX_NAME", "")

    def __init__(self, indexN=index_name):
        pinecone_api_key = os.getenv("PINECONE_API_KEY", "")

        if not pinecone_api_key:
            raise ValueError(
                "Pinecone API key or OpenAI API key is missing. Please set them in the environment variables."
            )
        self.ai_client = OpenAIClientModel()

        self.index_name = indexN
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.pc_status = self._initialize_index()

    # Function to initialize Pinecone index if it doesn't exist
    def _initialize_index(self):
        retries = 3
        delay = 180  # seconds
        for attempt in range(retries):
            try:
                """Initialize Pinecone index if it doesn't exist"""
                spec = ServerlessSpec(cloud="aws", region="us-east-1")

                if not self.index_name in self.pc.list_indexes().names():
                    self.pc.create_index(
                        name=self.index_name, dimension=1536, metric="cosine", spec=spec
                    )

                # Wait for index to be initialized
                while not self.pc.describe_index(self.index_name).status["ready"]:
                    time.sleep(1)

                self.index = self.pc.Index(self.index_name)
                return True
            except Exception as e:
                if attempt < retries - 1:
                    cPrint(
                        f"Retrying Pinecone Initialize Index in {delay} seconds",
                        "pinecone retry",
                        "purple",
                    )
                    time.sleep(delay)
                else:
                    cPrint(
                        f"All Pinecone Initialize Index retry attempts failed",
                        "Pinecone retry",
                        "purple",
                    )
                    return False

    def check_existing_entry(self, depoiq_id, category, chunk_index):
        """Checks if already exists in Pinecone."""
        try:
            existing_entries = self.index.query(
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
            cPrint(
                e,
                "⚠️ Error querying Pinecone in check_existing_entry",
                "red",
            )
            return False

    # Function to store summaries in Pinecone with embeddings
    def store_summaries_in_pinecone(self, depoiq_id, category, text_chunks):
        """Stores text chunks in Pinecone with embeddings."""
        try:
            vectors_to_upsert = []
            skipped_chunks = []

            for chunk_index, chunk_value in enumerate(text_chunks):

                cPrint(
                    f"🔹 Processing chunk {chunk_index + 1} of {category}",
                    "Info",
                    "blue",
                )

                if not chunk_value or not isinstance(chunk_value, str):
                    cPrint(
                        f"⚠️ Skipping invalid text chunk: {chunk_index}",
                        "Info",
                        "blue",
                    )
                    continue

                # Check if summary already exists
                if self.check_existing_entry(depoiq_id, category, chunk_index):
                    skipped_chunks.append(chunk_index)

                    cPrint(
                        f"⚠️ Summary already exists: {chunk_index}, skipping...",
                        "Warning",
                        "yellow",
                    )
                    continue

                # Extract keywords and synonyms
                keywords, synonyms_keywords = (
                    self.ai_client.extract_keywords_and_synonyms(chunk_value)
                )

                cPrint(keywords, "Extracted Keywords: ", "green")

                # Generate embedding
                embedding_text = self.ai_client.generate_embedding(text=chunk_value)
                embedding_keywords = self.ai_client.generate_embedding(text=keywords)
                embedding_synonyms_keywords = self.ai_client.generate_embedding(
                    text=synonyms_keywords
                )

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

                self.index.upsert(vectors=vectors_to_upsert)

                cPrint(
                    f" {len(vectors_to_upsert)} summaries in Pinecone.",
                    "✅ Successfully inserted",
                    "green",
                )

            return len(vectors_to_upsert), skipped_chunks
        except Exception as e:
            cPrint(
                e,
                "⚠️ Error storing summaries in Pinecone",
                "red",
            )
            return 0, []

    # Function to store transcript lines in Pinecone
    def store_transcript_lines_in_pinecone(self, depoiq_id, category, transcript_data):
        try:
            # Validate transcript_data
            if not transcript_data or not isinstance(transcript_data, list):
                cPrint("Invalid transcript_data - expected list", "Error", "red")
                return 0, []

            vectors_to_upsert = []
            skipped_chunks = []
            chunk_page_range = 5
            max_chunk_upsert = 50
            total_upserted = 0

            for i in range(0, len(transcript_data), chunk_page_range):
                grouped_pages = transcript_data[i : i + chunk_page_range]

                # Validate grouped_pages
                if not grouped_pages:
                    continue

                chunk_index = f"{grouped_pages[0]['pageNumber']}-{grouped_pages[-1]['pageNumber']}"

                # check if already exists in pinecone
                if self.check_existing_entry(depoiq_id, category, chunk_index):
                    skipped_chunks.append(
                        "Pages " + str(i + 1) + "-" + str(i + chunk_page_range)
                    )
                    cPrint(
                        f"⚠️ Transcript already exists for pages {i+1}-{i+chunk_page_range}, skipping...",
                        "Warning",
                        "yellow",
                    )
                    continue

                # Extract transcript lines with validation
                raw_lines = []
                for page in grouped_pages:
                    lines = page.get("lines", [])
                    if not lines or not isinstance(lines, list):
                        continue
                    for line in lines:
                        if not line or not isinstance(line, dict):
                            continue
                        line_text = line.get("lineText", "").strip()
                        if line_text:
                            raw_lines.append(line_text)

                trimmed_text = " ".join(raw_lines).strip()

                if not trimmed_text:

                    cPrint(
                        f"⚠️ Skipping empty text chunk for pages {i+1}-{i+chunk_page_range}",
                        "Warning Skipping",
                        "yellow",
                    )
                    continue

                cPrint(
                    f"(Pages {grouped_pages[0]['pageNumber']} - {grouped_pages[-1]['pageNumber']})",
                    "🔹 Storing text chunk :" "blue",
                )

                # Extract keywords and synonyms
                keywords, synonyms_keywords = (
                    self.ai_client.extract_keywords_and_synonyms(trimmed_text)
                )
                if (
                    not keywords or not synonyms_keywords
                ):  # or whatever validation makes sense
                    cPrint("Failed to extract keywords", "Warning", "yellow")
                    continue

                cPrint(
                    f"{trimmed_text} -> {keywords}",
                    "Extracted Keywords: ",
                    "green",
                )

                # Generate embedding
                embedding_text = self.ai_client.generate_embedding(text=trimmed_text)
                if not embedding_text:  # assuming embedding should be a list/array
                    cPrint("Failed to generate embedding", "Warning", "yellow")
                    continue
                embedding_keywords = self.ai_client.generate_embedding(text=keywords)
                if not embedding_keywords:
                    cPrint("Failed to generate embedding", "Warning", "yellow")
                    continue
                embedding_synonyms_keywords = self.ai_client.generate_embedding(
                    text=synonyms_keywords
                )
                if not embedding_synonyms_keywords:
                    cPrint("Failed to generate embedding", "Warning", "yellow")
                    continue

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
                    self.index.upsert(vectors=vectors_to_upsert)
                    total_upserted += len(vectors_to_upsert)

                    cPrint(
                        f" {total_upserted} transcript chunks in Pinecone.",
                        "✅ Successfully inserted",
                        "green",
                    )
                    vectors_to_upsert = []

            if vectors_to_upsert:
                self.index.upsert(vectors=vectors_to_upsert)
                total_upserted += len(vectors_to_upsert)

                cPrint(
                    f" {total_upserted} transcript chunks in Pinecone.",
                    "✅ Successfully inserted",
                    "green",
                )

            return total_upserted, skipped_chunks
        except Exception as e:
            cPrint(
                e,
                "⚠️ Error storing transcript lines in Pinecone",
                "red",
            )
            return 0, []

    # Function to store contradictions in Pinecone
    def store_contradictions_in_pinecone(
        self, depoiq_id, category, contradictions_data
    ):
        vectors_to_upsert = []
        skipped_chunks = []
        max_chunk_upsert = 50
        total_upserted = 0

        try:

            cPrint(
                f" {depoiq_id} {len(contradictions_data)}",
                "🔹 Storing contradictions for depoiq_id:",
                "blue",
            )

            for index in range(0, len(contradictions_data)):
                contradictions = contradictions_data[index]
                chunk_index = contradictions.get("contradiction_id", None)

                cPrint(
                    f"{index} -> {chunk_index}",
                    "🔹 Storing contradictions: ",
                    "blue",
                )

                if chunk_index == None:

                    cPrint(
                        f"Contradiction ID is None",
                        "⚠️ Skipping contradictions:",
                        "yellow",
                    )
                    skipped_chunks.append(chunk_index)
                    continue

                # check if already exists in pinecone
                if self.check_existing_entry(
                    depoiq_id,
                    category,
                    chunk_index,
                ):
                    skipped_chunks.append(chunk_index)

                    cPrint(
                        f"⚠️ Contradiction already exists for pages {chunk_index}",
                        "skipping",
                        "yellow",
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
                keywords, synonyms_keywords = (
                    self.ai_client.extract_keywords_and_synonyms(contradiction_text)
                )

                cPrint(
                    f"{contradiction_text} -> {keywords}",
                    "Extracted Keywords: ",
                    "green",
                )

                # Generate embedding
                embedding_text = self.ai_client.generate_embedding(
                    text=contradiction_text
                )
                embedding_keywords = self.ai_client.generate_embedding(text=keywords)
                embedding_synonyms_keywords = self.ai_client.generate_embedding(
                    text=synonyms_keywords
                )

                if not any(embedding_text):

                    cPrint(
                        contradiction_text,
                        "⚠️ Skipping empty embedding for contradictions:",
                        "yellow",
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
                    self.index.upsert(vectors=vectors_to_upsert)
                    total_upserted += len(vectors_to_upsert)

                    cPrint(
                        f" {total_upserted} contradictions in Pinecone.",
                        "✅ Successfully inserted",
                        "green",
                    )
                    vectors_to_upsert = []

            if vectors_to_upsert:
                self.index.upsert(vectors=vectors_to_upsert)
                total_upserted += len(vectors_to_upsert)

                cPrint(
                    f" {total_upserted} contradictions in Pinecone.",
                    "✅ Successfully inserted",
                    "green",
                )

            return total_upserted, skipped_chunks

        except Exception as e:
            cPrint(
                e,
                "⚠️ Error storing contradictions in Pinecone",
                "red",
            )
            return 0, []

    # Function to store admissions in Pinecone
    def store_admissions_in_pinecone(self, depoiq_id, category, admissions_data):
        vectors_to_upsert = []
        skipped_chunks = []
        max_chunk_upsert = 50
        total_upserted = 0

        try:

            cPrint(
                f"{depoiq_id} {len(admissions_data)}",
                "🔹 Storing admissions for depoiq_id:",
                "blue",
            )

            for index in range(0, len(admissions_data)):
                admissions = admissions_data[index]
                chunk_index = admissions.get("admission_id")

                if chunk_index == None:

                    cPrint(
                        f"{admissions} -> chunk_index is None",
                        "⚠️ Skipping admissions:",
                        "yellow",
                    )
                    continue

                cPrint(
                    f"{index} -> {chunk_index}",
                    "🔹 Storing admissions: ",
                    "blue",
                )

                # check if already exists in pinecone
                if self.check_existing_entry(
                    depoiq_id,
                    category,
                    chunk_index,
                ):
                    skipped_chunks.append(chunk_index)

                    cPrint(
                        f"⚠️ Admission already exists for pages {chunk_index}, ",
                        "skipping",
                        "yellow",
                    )
                    continue

                # Extract admissions text, ensuring each line is properly formatted
                reason = admissions.get("reason")
                question = admissions.get("question")
                answer = admissions.get("answer")
                admission_text = f"{question} {answer} {reason}"

                cPrint(
                    admission_text,
                    "🔹 Storing admission text:",
                    "blue",
                )

                # Extract keywords and synonyms
                keywords, synonyms_keywords = (
                    self.ai_client.extract_keywords_and_synonyms(admission_text)
                )

                cPrint(
                    f"{admission_text} -> {keywords}",
                    "Extracted Keywords: ",
                    "green",
                )

                # Generate embedding
                embedding_text = self.ai_client.generate_embedding(text=admission_text)
                embedding_keywords = self.ai_client.generate_embedding(text=keywords)
                embedding_synonyms_keywords = self.ai_client.generate_embedding(
                    text=synonyms_keywords
                )

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
                    self.index.upsert(vectors=vectors_to_upsert)
                    total_upserted += len(vectors_to_upsert)

                    cPrint(
                        f" {total_upserted} admissions in Pinecone.",
                        "✅ Successfully inserted",
                        "green",
                    )
                    vectors_to_upsert = []

            if vectors_to_upsert:
                self.index.upsert(vectors=vectors_to_upsert)
                total_upserted += len(vectors_to_upsert)

                cPrint(
                    f" {total_upserted} admissions in Pinecone.",
                    "✅ Successfully inserted",
                    "green",
                )

            return total_upserted, skipped_chunks

        except Exception as e:
            cPrint(
                e,
                "⚠️ Error storing admissions in Pinecone",
                "red",
            )
            return 0, []

    # Function to query Pinecone and match return top k results
    def query_pinecone(
        self,
        query_text,
        depo_ids=[],
        top_k=5,
        is_unique=False,
        category=None,
        is_detect_category=False,
    ):
        try:

            cPrint(
                f"Query_Text -> {query_text} (depo_id->: {depo_ids})",
                "Querying Pinecone",
                "blue",
            )

            # Generate query embedding
            query_vector = self.ai_client.generate_embedding(text=query_text)

            category_detect = (
                detect_category(query_text) if is_detect_category else "all"
            )

            if not query_vector:
                return json.dumps(
                    {
                        "status": "error",
                        "message": "Failed to generate query embedding.",
                    }
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
            results = self.index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True,
                filter=filter_criteria,
            )

            # Check if any matches found
            if not results["matches"] or not any(results["matches"]):
                cPrint(
                    f" {query_text}",
                    "No matches found",
                    "yellow",
                )
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
                    "synonyms_keywords": match["metadata"].get(
                        "synonyms_keywords", None
                    ),
                }
                for match in results.get("matches", [])
            ]

            cPrint(matched_results, "Matched Results", "cyan")

            from models.depo_model import DepoModel

            depo_instance = DepoModel()

            # Group results by depoiq_id
            grouped_result = depo_instance.group_by_depoIQ_ID(
                matched_results,
                isSummary=True,
                isTranscript=True,
                isContradictions=True,
            )

            cPrint(
                f"{len(grouped_result)}",
                "Grouped Results: ",
                "blue",
            )

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
                        text_chunks[chunk_index]
                        if chunk_index < len(text_chunks)
                        else text
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

                        cPrint(
                            f"Contradiction with ID {chunk_index} not found in depo_id {depoiq_id}",
                            "⚠️ Skipping contradictions:",
                            "yellow",
                        )
                        continue

                    reason = contradiction.get("reason")
                    initial_question = contradiction.get("initial_question")
                    initial_answer = contradiction.get("initial_answer")
                    contradictory_responses = contradiction.get(
                        "contradictory_responses"
                    )
                    contradictory_responses_string = ""
                    for contradictory_response in contradictory_responses:
                        contradictory_responses_string += f"{contradictory_response['contradictory_question']} {contradictory_response['contradictory_answer']} "

                    text = f"{initial_question} {initial_answer} {contradictory_responses_string} {reason}"

                    cPrint(
                        f"Contradiction: {text}",
                        "Contradiction Text",
                        "blue",
                    )
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
            cPrint(e, "Error querying Pinecone in query_pinecone:", "red")
            return {"status": "error", "message": str(e)}
