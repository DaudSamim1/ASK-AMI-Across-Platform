from helperFunctions.general_helpers import (
    generate_token,
    camel_to_snake,
    extract_text,
    split_into_chunks,
)
from helperFunctions.pinecone_class import PineConeClass
from helperFunctions.general_helpers import cPrint
import requests
import os


class Depo:

    def __init__(self):
        self.GRAPHQL_URL = os.getenv("GRAPHQL_URL", "")
        if not self.GRAPHQL_URL:
            raise ValueError(
                "GRAPHQL_URL is not set. Please set it in the environment variables."
            )
        self.pc_instance = PineConeClass()
        self.token = generate_token()

    # Function to get Depo from DepoIQ_ID
    def getDepo(
        self,
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

            # token = generate_token()

            headers = {
                "Content-Type": "application/json",
                "Authorization": self.token,
            }

            response = requests.post(self.GRAPHQL_URL, json=payload, headers=headers)

            cPrint(
                response.status_code,
                "Response Status Code",
                "blue",
            )

            if response.status_code == 200:
                return response.json()["data"]["getDepo"]
            elif response.status_code == 401:
                raise Exception("Unauthorized - Invalid Token")
            else:
                raise Exception(f"Failed to fetch depo data: {response.text}")
        except Exception as e:
            cPrint(
                e,
                "Error fetching depo data",
                "red",
            )
            return {}

    # Function to generate to Get Summary from Depo
    def getDepoSummary(self, depoiq_id):
        """Get Depo Summary from DepoIQ_ID"""
        try:
            depo = self.getDepo(
                depoiq_id,
                isSummary=True,
                isTranscript=False,
                isContradictions=False,
                isAdmission=False,
            )
            return depo["summary"]
        except Exception as e:
            cPrint(
                e,
                "Error fetching depo summary",
                "red",
            )
            return {}

    # Function to generate Get Transcript from Depo
    def getDepoTranscript(self, depoiq_id):
        """Get Depo Transcript from DepoIQ_ID"""
        try:
            depo = self.getDepo(
                depoiq_id,
                isSummary=False,
                isTranscript=True,
                isContradictions=False,
                isAdmission=False,
            )
            return depo["transcript"]
        except Exception as e:
            cPrint(
                e,
                "Error fetching depo transcript",
                "red",
            )
            return {}

    # Function to genrate Get Contradictions from Depo
    def getDepoContradictions(self, depoiq_id):
        """Get Depo Contradictions from DepoIQ_ID"""
        try:
            depo = self.getDepo(
                depoiq_id,
                isSummary=False,
                isTranscript=False,
                isContradictions=True,
                isAdmission=False,
            )
            return depo["contradictions"]
        except Exception as e:
            cPrint(
                e,
                "Error fetching depo contradictions",
                "red",
            )
            return {}

    # Function to genrate Get Admissions from Depo
    def getDepoAdmissions(self, depoiq_id):
        """Get Depo Admissions from DepoIQ_ID"""
        try:
            depo = self.getDepo(
                depoiq_id,
                isSummary=False,
                isTranscript=False,
                isContradictions=False,
                isAdmission=True,
            )
            return depo["admissions"]
        except Exception as e:
            cPrint(
                e,
                "Error fetching depo admissions",
                "red",
            )
            return {}

    # Function to generate add depo summaries to pinecone
    def add_depo_summaries(self, depo_summary, depoiq_id):
        try:
            excluded_keys = ["visualization"]
            total_inserted = 0
            skipped_sub_categories = {}

            cPrint(
                depoiq_id,
                "Adding summaries for depoiq_id",
                "blue",
            )

            for key, value in depo_summary.items():
                if key in excluded_keys:
                    continue

                category = camel_to_snake(key)  # Convert key to category format
                text = extract_text(value["text"])  # Extract clean text
                text_chunks = split_into_chunks(text)  # Split into paragraphs

                inserted_count, skipped_chunks = (
                    self.pc_instance.store_summaries_in_pinecone(
                        depoiq_id, category, text_chunks
                    )
                )

                total_inserted += inserted_count
                skipped_sub_categories[category] = skipped_chunks

                cPrint(
                    f"✅ Successfully inserted {inserted_count} summaries for {category}",
                    "Info",
                    "green",
                )

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
                    "skipped_count": sum(
                        len(v) for v in skipped_sub_categories.values()
                    ),
                },
            }

            return response

        except Exception as e:
            return {
                "status": "error",
                "message": "Something went wrong in add_depo_summaries",
                "details": str(e),
            }

    # Function to add depo transcript to Pinecone
    def add_depo_transcript(self, transcript_data, depoiq_id):
        try:
            cPrint(
                depoiq_id,
                "Adding transcripts for depoiq_id",
                "blue",
            )
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
            inserted_transcripts, skipped_transcripts = (
                self.pc_instance.store_transcript_lines_in_pinecone(
                    depoiq_id, category, transcript_data
                )
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

    # Function to add depo contradictions to Pinecone
    def add_depo_contradictions(self, contradictions_data, depoiq_id):
        try:
            cPrint(
                depoiq_id,
                "Adding contradictions for depoiq_id",
                "blue",
            )

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

            # Store contradictions data
            inserted_contradictions, skipped_contradictions = (
                self.pc_instance.store_contradictions_in_pinecone(
                    depoiq_id, category, contradictions_data
                )
            )

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

    # Function to add depo admissions to Pinecone
    def add_depo_admissions(self, admissions_data, depoiq_id):
        try:

            cPrint(
                depoiq_id,
                "Adding admissions for depoiq_id",
                "blue",
            )

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
            inserted_admissions, skipped_admissions = (
                self.pc_instance.store_admissions_in_pinecone(
                    depoiq_id, category, admissions_data=admissions_data["text"]
                )
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

    # Function to group by depoiq_id for Pinecone results
    def group_by_depoIQ_ID(
        self,
        data,
        isSummary=False,
        isTranscript=False,
        isContradictions=False,
        isAdmission=False,
    ):
        grouped_data = {}
        for entry in data:
            depoiq_id = entry["depoiq_id"]
            if depoiq_id not in grouped_data:

                cPrint(
                    depoiq_id,
                    "🔍 Fetching Depo from DB",
                    "green",
                )

                if isTranscript and not isSummary:
                    grouped_data[depoiq_id] = self.getDepoTranscript(depoiq_id)
                    return grouped_data

                if isSummary and not isTranscript:
                    grouped_data[depoiq_id] = self.getDepoSummary(depoiq_id)
                    return grouped_data

                if isContradictions and not isSummary and not isTranscript:
                    grouped_data[depoiq_id] = self.getDepoContradictions(depoiq_id)
                    return grouped_data

                if (
                    isAdmission
                    and not isSummary
                    and not isTranscript
                    and not isContradictions
                ):
                    grouped_data[depoiq_id] = self.getDepoAdmissions(depoiq_id)
                    return grouped_data

                grouped_data[depoiq_id] = self.getDepo(
                    depoiq_id,
                    isSummary=True,
                    isTranscript=True,
                    isContradictions=True,
                    isAdmission=True,
                )
        cPrint(
            "Grouped Data fetched from DB",
            "Info",
            "blue",
        )
        return grouped_data
