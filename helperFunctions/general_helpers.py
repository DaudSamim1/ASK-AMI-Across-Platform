import json
import jwt
import os
import re
from bs4 import BeautifulSoup


ansi_colors = {
    "red": "1;31",  # Error in our Code
    "green": "1;32",  # File and document processing
    "yellow": "1;33",
    "blue": "1;34",  # Progress Print -> Show results from the progress of the algorithm
    "purple": "1;35",  # Debugging Print -> temporary prints that we use when we debug the code
    "cyan": "1;36",  # API Response
    "white": "1;37",
}


def cPrint(text, title="Output", title_color="purple"):
    # Get the color code, default to green if not found
    color_code = ansi_colors.get(title_color, "1;32")
    # Print title in given color and bold
    print("\n")

    if isinstance(text, dict):
        try:
            print(
                f"\n{title} → ",
                f"\033[{color_code}m{json.dumps(text, indent=4)}\033[0m",
                f"\033[1;37m ",
            )
        except:
            print(
                f"\n{title} → ",
                f"\033[{color_code}m{text}\033[0m",
                f"\033[1;37m \033[0m",
            )
    else:
        print(
            f"\n{title} → ", f"\033[{color_code}m{text}\033[0m", f"\033[1;37m \033[0m"
        )

    print("\n\n\n")


# Function to generate a JWT token
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

    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")

    # generate access token
    accessToken = "Bearer " + jwt.encode(
        access_token_payload, JWT_SECRET_KEY, algorithm="HS256"
    )

    return accessToken


# Function to convert camelCase to snake_case
def camel_to_snake(name):
    try:
        snake_case = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
        if not snake_case.endswith("_summary"):
            snake_case += "_summary"
        return snake_case
    except Exception as e:
        cPrint(
            f"⚠️ Error converting camelCase to snake_case: {e}",
            "Error",
            "red",
        )
        return name


# snake to camel case and also remove _summary
def snake_to_camel(name, isSummary=True):
    try:
        if isSummary:
            name = name.replace("_summary", "")
        parts = name.split("_")
        camel_case = parts[0] + "".join(word.capitalize() for word in parts[1:])
        return camel_case
    except Exception as e:
        cPrint(
            e,
            "⚠️ Error converting snake_case to camelCase:",
            "red",
        )
        return name


# Function to convert camelCase to snake_case
def extract_text(value):
    """Extracts plain text from HTML or returns simple text."""
    try:
        if not value:
            cPrint(
                "⚠️ extract_text received empty input.",
                "Error",
                "red",
            )
            return ""

        if value.startswith("<"):
            try:
                soup = BeautifulSoup(value, "html.parser")
                return soup.get_text().strip()
            except Exception as e:
                cPrint(
                    e,
                    "⚠️ Error parsing HTML:",
                    "red",
                )
                return ""
        return str(value).strip()
    except Exception as e:
        cPrint(
            e,
            "⚠️ Error extracting text:",
            "red",
        )
        return ""


# Function to split text into chunks while keeping sentences intact
def split_into_chunks(text, min_chunk_size=300, max_chunk_size=500):
    """Splits text into balanced chunks while keeping sentences intact."""

    try:
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
    except Exception as e:
        cPrint(
            e,
            "⚠️ Error splitting text into chunks:",
            "red",
        )
        return [text]


# Function to detect the category of the user query
def detect_category(user_query):
    try:
        query_lower = user_query.lower()

        # Mapping keywords to specific categories
        category_keywords = {
            "overview_summary": [
                "overview summary",
                "general overview",
                "case summary",
            ],
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
                cPrint(
                    category,
                    "Detected category",
                    "green",
                )
                return category

        cPrint(
            "No specific category detected. Using all categories.",
            "Detected category",
            "yellow",
        )
        return "all"
    except Exception as e:
        cPrint(
            e,
            "⚠️ Error detecting category:",
            "red",
        )
        return "all"
