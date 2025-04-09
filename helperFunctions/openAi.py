import os
from openai import OpenAI
from helperFunctions.general_helpers import cPrint
from prompts import (
    extract_keywords_and_synonyms,
    get_answer_from_AI,
    get_answer_score_from_AI,
    generate_legal_prompt_from_AI,
    get_answer_from_AI_without_ref,
    generate_detailed_answer,
)


class OpenAIClient:

    def __init__(self):
        openai_api_key = os.getenv("OPENAI_API_KEY", "")
        if not openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Please set it in the environment variables."
            )

        self.client = OpenAI(api_key=openai_api_key)

    # Function to generate embeddings from OpenAI
    def generate_embedding(self, text):
        try:
            if not text or not isinstance(text, str) or text.strip() == "":
                cPrint(
                    text,
                    "⚠️ Invalid or empty text for embedding generation.",
                    "red",
                )
                return False

            text = text.replace("\n", " ").replace("\r", " ").strip()
            response = self.client.embeddings.create(
                input=[text], model="text-embedding-3-small"
            )
            return response.data[0].embedding
        except Exception as e:
            cPrint(
                e,
                f"⚠️ Error generating embeddings for text: {text[:200]}...",
                "red",
            )
            return False

    # Function to extract keywords and synonyms from text
    def extract_keywords_and_synonyms(self, text):
        try:
            prompt = extract_keywords_and_synonyms.extract_keywords_and_synonyms_prompt(
                text
            )

            response = self.client.chat.completions.create(
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
            cPrint(
                e,
                f"⚠️ Error extracting keywords and synonyms: {text[:200]}...",
                "red",
            )
            return [], []

    # Function to get the answer from AI based on metadata and user query
    def get_answer_from_AI(self, response):
        try:
            text_list = [entry["text"] for entry in response["metadata"]]
            # Construct prompt
            prompt = get_answer_from_AI.get_answer_from_AI_prompt(response, text_list)

            # Call GPT-3.5 API with deterministic parameters for consistent responses
            ai_response = self.client.chat.completions.create(
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

            cPrint(
                extracted_text,
                "AI Response: ",
                "blue",
            )

            return extracted_text

        except Exception as e:
            cPrint(
                e,
                f"⚠️ Error querying Pinecone in get_answer_from_AI:",
                "red",
            )
            return str(e)

    # Function to get the answer score from AI based on the question and answer
    def get_answer_score_from_AI(self, question, answer):
        try:
            # Construct prompt
            prompt = get_answer_score_from_AI.get_answer_score_from_AI_prompt(
                question, answer
            )

            # Call GPT-3.5 API with deterministic parameters for consistent responses
            ai_response = self.client.chat.completions.create(
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

            cPrint(
                extracted_text,
                "AI Response: ",
                "blue",
            )

            return extracted_text

        except Exception as e:
            cPrint(
                e,
                f"⚠️ Error querying Pinecone in get_answer_score_from_AI:",
                "red",
            )
            return str(e)

    # Function to generate legal questions from AI based on user query
    def generate_legal_prompt_from_AI(self, user_query, count=6):
        try:
            prompt = generate_legal_prompt_from_AI.generate_legal_prompt_from_AI_prompt(
                user_query, count
            )

            # Call GPT-4o API with specific parameters for consistency
            ai_response = self.client.chat.completions.create(
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
            cPrint(
                question_list,
                "AI Response: ",
                "blue",
            )
            return question_list

        except Exception as e:
            cPrint(
                e,
                f"⚠️ Error generating legal questions:",
                "red",
            )
            return str(e)

    # Function to get an answer from AI without referencing metadata
    def get_answer_from_AI_without_ref(self, response):
        try:
            text_list = [entry["text"] for entry in response["metadata"]]

            # Construct prompt
            prompt = (
                get_answer_from_AI_without_ref.get_answer_from_AI_without_ref_prompt(
                    response, text_list
                )
            )

            # Call GPT-3.5 API with deterministic parameters for consistent responses
            ai_response = self.client.chat.completions.create(
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

            cPrint(
                extracted_text,
                "AI Response: ",
                "blue",
            )

            return extracted_text

        except Exception as e:
            cPrint(
                e,
                f"⚠️ Error querying Pinecone in get_answer_from_AI_without_ref:",
                "red",
            )
            return str(e)

    # Function to generate a detailed answer from AI based on user query and answers
    def generate_detailed_answer(self, user_query, answers):
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
            prompt = generate_detailed_answer.generate_detailed_answer_prompt(
                user_query, text_list
            )

            # Generate AI response using GPT-4
            ai_response = self.client.chat.completions.create(
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

            cPrint(
                extracted_text,
                "AI Response: ",
                "blue",
            )

            return extracted_text

        except Exception as e:
            cPrint(
                e,
                f"⚠️ Error generating detailed answer:",
                "red",
            )
            return str(e)
