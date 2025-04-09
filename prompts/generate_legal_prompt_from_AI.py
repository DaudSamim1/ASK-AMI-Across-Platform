def generate_legal_prompt_from_AI_prompt(user_query, count=6):
    return f"""
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