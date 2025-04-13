# Copyright (C) 2025 Yelyzaveta Ivanytska

import re
import json
from fastapi.responses import JSONResponse


def convert_str_to_dict(text: str) -> JSONResponse:
    cleaned_string = re.sub(r"```json\n|```", "", text)
    parsed = json.loads(cleaned_string)

    return parsed