from enum import Enum
from pydantic import BaseModel
from typing import Optional


class SummaryCategoriesType(Enum):
    SUMMARY = "summary"
    TRANSCRIPT = "transcript"
    CONTRADICTIONS = "contradictions"
    ADMISSIONS = "admissions"


class PineConeCategoriesType(Enum):
    ALL = "all"
    TEXT = "text"
    KEYWORDS = "keywords"
    SYNONYMS = "synonyms"


class TalkDepoRequest(BaseModel):
    depoiq_ids: list[str]
    user_query: str
    category: Optional[PineConeCategoriesType] = PineConeCategoriesType.TEXT
    is_unique: Optional[bool] = False


class AnswerValidatorRequest(BaseModel):
    questions: list[str]
    depoiq_id: str
    category: Optional[PineConeCategoriesType] = PineConeCategoriesType.ALL
    is_download: Optional[bool] = True
    top_k: Optional[int] = 10


class AskAmiAgentRequest(BaseModel):
    depoiq_ids: list[str]
    user_query: str
