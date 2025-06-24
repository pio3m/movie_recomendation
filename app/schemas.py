from pydantic import BaseModel
from typing import List, Optional

class RecommendRequest(BaseModel):
    liked_titles: List[str]
    n_recommendations: Optional[int] = 10
