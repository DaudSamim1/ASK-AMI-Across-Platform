from flask import jsonify, Response, request
from helperFunctions.general_helpers import cPrint
from helperFunctions.pinecone_class import PineConeClass
from helperFunctions.depo import Depo
from bson import ObjectId
import csv
import io


class DepoController:

    def __init__(self):
        self.depo_model = Depo()
        self.pc = PineConeClass()
        self.ai_client = self.pc.ai_client
        self.alowed_depo_categories = [
            "summary",
            "transcript",
            "contradictions",
            "admissions",
        ]
        self.alowed_pc_categories = ["text", "keywords", "synonyms", "all"]

    # Function to get Depo from DepoIQ_ID
    def get_depo_by_Id(self, depoiq_id):
        try:
            # get category from query params
            category = request.args.get("category", None)

            if category:

                if category not in self.alowed_depo_categories:
                    return (
                        jsonify(
                            {
                                "error": "Invalid category. Should be 'summary', 'transcript', 'contradictions', or 'admissions'."
                            }
                        ),
                        400,
                    )

            depo = self.depo_model.getDepo(
                depoiq_id,
                isSummary=not category or category == "summary",
                isTranscript=not category or category == "transcript",
                isContradictions=not category or category == "contradictions",
                isAdmission=not category or category == "admissions",
            )
            cPrint(depo, "API Response", "cyan")
            return jsonify(depo), 200

        except Exception as e:
            return (
                jsonify(
                    {
                        "error": "Something went wrong in get_depo_by_Id",
                        "details": str(e),
                    }
                ),
                500,
            )

    # Function to add Depo to Pinecone
    def add_depo(self, depoiq_id):
        try:
            category = request.args.get("category", None)

            if category and category not in self.alowed_depo_categories:
                return (
                    jsonify(
                        {
                            "error": "Invalid category. Should be 'summary', 'transcript', 'contradictions', or 'admissions'."
                        }
                    ),
                    400,
                )

            # Fetch depo data
            depo = self.depo_model.getDepo(
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
                summary_response = self.depo_model.add_depo_summaries(
                    summary_data, depoiq_id
                )
                responses["summary"] = summary_response["data"]
                status_list.append(summary_response["status"])

            if not category or category == "transcript":
                transcript_data = depo.get("transcript", [])
                transcript_response = self.depo_model.add_depo_transcript(
                    transcript_data, depoiq_id
                )
                responses["transcript"] = transcript_response["data"]
                status_list.append(transcript_response["status"])

            if not category or category == "contradictions":
                contradictions_data = depo.get("contradictions", [])
                contradiction_response = self.depo_model.add_depo_contradictions(
                    contradictions_data, depoiq_id
                )
                responses["contradictions"] = contradiction_response["data"]
                status_list.append(contradiction_response["status"])

            if not category or category == "admissions":
                admissions_data = depo.get("admissions", [])
                admissions_response = self.depo_model.add_depo_admissions(
                    admissions_data, depoiq_id
                )
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
            cPrint(e, "🔹 Error in add_depo:", "red")
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

    # Function to talk to Depo
    def talk_depo(self):
        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "Invalid request, JSON body required"}), 400

            depoiq_ids = data.get("depoiq_ids")
            user_query = data.get("user_query")
            is_unique = data.get("is_unique", False)
            category = data.get("category", "text")

            if category not in self.alowed_pc_categories:
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
            query_pinecone_response = self.pc.query_pinecone(
                user_query,
                depoiq_ids,
                top_k=top_k,
                is_unique=is_unique,
                category=category,
                is_detect_category=False,
            )

            cPrint(query_pinecone_response, "✅ Query Pinecone Response", "green")

            if query_pinecone_response["status"] == "Not Found":
                cPrint(
                    f"\n\n🔍 No matches found for the query and return error \n\n\n\n",
                    "Error",
                    "red",
                )
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
                ai_resposne = self.ai_client.get_answer_from_AI(query_pinecone_response)

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

    # Function to Validate Depo Answers
    def answer_validator(self):
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

            answers_response = []
            if category not in self.alowed_pc_categories:
                return (
                    jsonify(
                        {
                            "error": "Invalid category. Should be 'text' , 'keywords' , 'synonyms', 'all'."
                        }
                    ),
                    400,
                )

            for question in questions:
                cPrint(question, "Question", "green")
                query_pinecone_response = self.pc.query_pinecone(
                    question,
                    [depoiq_id],
                    top_k=top_k,
                    is_unique=False,
                    category=category,
                    is_detect_category=False,
                )

                cPrint(
                    query_pinecone_response,
                    "✅ Query Pinecone Response",
                    "green",
                )

                if query_pinecone_response["status"] == "Not Found":
                    cPrint(
                        f"\n\n🔍 No matches found for the query and return error \n\n\n\n",
                        "Error",
                        "red",
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
                    ai_resposne = self.ai_client.get_answer_from_AI(
                        query_pinecone_response
                    )

                    score = self.ai_client.get_answer_score_from_AI(
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

    # Function to Ask AMI Agent
    def ask_ami_agent(self):
        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "Invalid request, JSON body required"}), 400

            user_query = data.get("user_query")
            depoiq_ids = data.get("depoiq_ids")

            questions = self.ai_client.generate_legal_prompt_from_AI(
                user_query, count=6
            )
            cPrint(
                questions,
                "🔹 Question List",
                "purple",
            )

            if not depoiq_ids:
                return jsonify({"error": "Missing depoiq_id"}), 400

            answers_response = []

            for question in questions:
                cPrint(question, "Question", "green")
                query_pinecone_response = self.pc.query_pinecone(
                    question,
                    depoiq_ids,
                    top_k=8,
                    is_unique=False,
                    category="text",
                    is_detect_category=True,
                )

                cPrint(
                    query_pinecone_response,
                    "✅ Query Pinecone Response",
                    "green",
                )

                if query_pinecone_response["status"] == "Not Found":
                    cPrint(
                        f"\n\n🔍 No matches found for the query and return error \n\n\n\n",
                        "Error",
                        "red",
                    )
                    answers_response.append(
                        {
                            "question": question,
                            "answer": "No matches found for the query",
                        }
                    )
                    continue

                else:
                    ai_resposne = self.ai_client.get_answer_from_AI_without_ref(
                        query_pinecone_response
                    )

                    answers_response.append(
                        {
                            "question": question,
                            "answer": str(ai_resposne),
                            "metadata": query_pinecone_response.get("metadata"),
                            "depoiq_id": query_pinecone_response.get("depoiq_id"),
                        }
                    )
            detailed_answer = self.ai_client.generate_detailed_answer(
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
