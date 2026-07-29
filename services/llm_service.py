import json
import logging
import re
import time
import unicodedata
from difflib import get_close_matches
import requests
from config import GEMINI_API_KEY, GEMINI_MODEL, GROQ_API_KEY, GROQ_MODEL

logger = logging.getLogger("ai-service.llm")


def _clean_text_noise(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"(?i)\b(facebook|tiktok|twitter|instagram|fanpage|website|zalo|subscribe|đăng ký|theo dõi|bản quyền|contact|email|thắc mắc|kênh youtube|fan page)\b.*", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


COOKING_ACTION_VERBS = re.compile(
    r"\b(sơ chế|rửa|cắt|thái|băm|ướp|trộn|pha|nêm|cho|đun|nấu|luộc|hấp|chiên|xào|kho|rim|quay|vớt|tắt|bếp|lửa|đảo|phi|chưng|bày|thưởng thức|chuẩn bị|làm sạch|đun nóng|đổ|rút|trút|múc|gắp|chế|rưới|rắc|vắt|thả|thêm|đậy|hầm|nướng|ngâm|dùng|sốt)\b",
    re.IGNORECASE
)


def _is_invalid_step(step: dict) -> bool:
    if not isinstance(step, dict):
        return False
    text = " ".join([
        str(step.get("instruction") or ""),
        str(step.get("action") or ""),
        " ".join([str(m) for m in step.get("main") or []]),
        " ".join([str(w) for w in step.get("with") or []]),
    ]).lower()

    if len(text.strip()) < 4:
        return True

    # 1. Check URLs & social media promos
    if re.search(r"https?://|www\.|facebook\.com|tiktok\.com|twitter\.com|instagram\.com|\.vn\b|\.com\b|fanpage|website", text):
        return True

    # 2. Check foreign non-Latin scripts (Korean, Japanese, Chinese, Thai, Hindi, Arabic)
    if re.search(r"[\u3040-\u30ff\u4e00-\u9faf\uac00-\ud7af\u0e00-\u0e7f\u0900-\u097f\u0600-\u06ff]", text):
        return True

    # 3. Check common multi-language title keywords in video description dumps
    if re.search(r"\b(resep|sayap|ayam|ailes|poulet|sauce|vietnamienne|recipe|receta)\b", text):
        return True

    # 4. Must contain at least 1 cooking action verb (filters out raw ingredient quantity list dumps)
    if not COOKING_ACTION_VERBS.search(text):
        return True

    return False


def build_recipe_context(transcript: str, description: str, max_chars: int = 9000) -> str:
    """Keep the most useful cooking instructions within a prompt budget.

    If transcript is within budget, preserve the ENTIRE transcript sequentially
    so that zero cooking details, proportions, or steps are dropped.
    """
    transcript = re.sub(r"\s+", " ", transcript or "").strip()
    description = _clean_text_noise(description)

    description_budget = min(800, max_chars // 4) if description else 0
    transcript_budget = max_chars - description_budget
    parts = []
    if description:
        parts.append(f"MÔ TẢ VIDEO: {description[:description_budget]}")

    if not transcript:
        return "\n".join(parts)

    if len(transcript) <= transcript_budget:
        parts.append(f"TRANSCRIPT: {transcript}")
        return "\n".join(parts)

    chunks = [item.strip() for item in re.split(r"(?<=[.!?…])\s+", transcript) if item.strip()]
    if len(chunks) < 3:
        chunks = [transcript[index:index + 220] for index in range(0, len(transcript), 220)]
    else:
        chunks = [
            chunk[index:index + 260]
            for chunk in chunks
            for index in range(0, len(chunk), 260)
        ]

    action_words = (
        "rửa", "cắt", "thái", "băm", "ướp", "trộn", "pha", "nêm", "cho", "đun",
        "nấu", "luộc", "hấp", "chiên", "xào", "kho", "rim", "quay", "vớt", "tắt bếp",
        "lửa", "phút", "sôi", "chín", "vàng", "mềm", "giòn", "nước mắm", "muỗng", "thìa", "gam",
    )
    ranked = sorted(
        range(len(chunks)),
        key=lambda index: (
            sum(word in chunks[index].lower() for word in action_words),
            len(chunks[index]),
        ),
        reverse=True,
    )

    selected = {0}
    used = len(chunks[0])
    for index in ranked:
        if index in selected:
            continue
        neighbor = index - 1
        extra = len(chunks[index]) + 1
        include_neighbor = neighbor >= 0 and neighbor not in selected
        if include_neighbor:
            extra += len(chunks[neighbor]) + 1
        if used + extra > transcript_budget:
            continue
        if include_neighbor:
            selected.add(neighbor)
            used += len(chunks[neighbor]) + 1
        selected.add(index)
        used += len(chunks[index]) + 1

    compact_transcript = " ".join(chunks[index] for index in sorted(selected))
    parts.append(f"TRANSCRIPT: {compact_transcript}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Grounding / normalization helpers (no extra LLM calls)
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "").lower()
    return re.sub(r"\s+", " ", text).strip()


def _appears_in_source(item: str, source_norm: str) -> bool:
    """Check that a model-claimed ingredient/word is actually backed by the
    source text, allowing a 1-word slack for morphological variants."""
    item_norm = _normalize(item)
    words = [w for w in item_norm.split() if len(w) > 1]
    if not words:
        return True
    hits = sum(1 for w in words if w in source_norm)
    return hits >= max(1, len(words) - 1)


def _instruction_grounding_ratio(instruction: str, source_norm: str) -> float:
    """Return the fraction of content words in *instruction* that appear in *source_norm*.

    Short common Vietnamese words (≤2 chars) are skipped since they are
    function words (cho, và, với, rồi …) that don't indicate hallucination.
    """
    words = [w for w in _normalize(instruction).split() if len(w) > 2]
    if not words:
        return 1.0
    hits = sum(1 for w in words if w in source_norm)
    return hits / len(words)


def verify_step_grounding(step: dict, source_text: str) -> dict:
    """Drop main/with items the model invented but the source never mentions.
    Also remove instruction if it is mostly ungrounded so render_step will
    compose a safe template-based fallback from structured fields."""
    if not isinstance(step, dict):
        return step
    source_norm = _normalize(source_text)
    for key in ("main", "with"):
        items = step.get(key) or []
        if isinstance(items, list):
            kept = [it for it in items if _appears_in_source(str(it), source_norm)]
            dropped = [it for it in items if it not in kept]
            if dropped:
                logger.info(f"Dropped ungrounded {key} items: {dropped}")
            step[key] = kept

    # Check instruction grounding — drop if <25% of content words are in source
    instruction = step.get("instruction") or ""
    if instruction:
        ratio = _instruction_grounding_ratio(instruction, source_norm)
        if ratio < 0.25:
            logger.warning(
                f"Dropped hallucinated instruction (grounding {ratio:.0%}): "
                f"{instruction[:80]}..."
            )
            del step["instruction"]

    return step


def reconcile_names(recipe: dict) -> dict:
    """Snap step ingredient mentions to the closest name already declared in
    `ingredients`, so the UI can link step text back to the ingredient list."""
    if not isinstance(recipe, dict):
        return recipe
    ing_names = [ing.get("name", "") for ing in recipe.get("ingredients", []) if isinstance(ing, dict)]
    if not ing_names:
        return recipe
    for step in recipe.get("steps", []):
        if not isinstance(step, dict):
            continue
        for key in ("main", "with"):
            items = step.get(key) or []
            if not isinstance(items, list):
                continue
            step[key] = [
                (get_close_matches(item, ing_names, n=1, cutoff=0.6) or [item])[0]
                for item in items
            ]
    return recipe


def has_cook_phase(steps: list) -> bool:
    """Sanity flag: if a recipe only has prep steps, extraction likely missed
    the actual cooking part of the video (worth flagging for review)."""
    return any(isinstance(s, dict) and s.get("phase") == "cook" for s in steps)


# ---------------------------------------------------------------------------
# Step rendering: LLM emits short fields, code composes the display sentence
# ---------------------------------------------------------------------------

def _clean_parts(value) -> list[str]:
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    return [str(item).strip(" ,.") for item in values if str(item).strip(" ,.")]


def _join_parts(parts: list[str]) -> str:
    return ", ".join(parts)


def render_step(step: dict, index: int) -> dict:
    """Render step into instruction field, prioritizing LLM's detailed instruction.
    Never auto-concatenate ingredient lists into fake generic template sentences."""
    if isinstance(step, str):
        return {"stepNumber": index, "instruction": step.strip()}

    if not isinstance(step, dict):
        return {"stepNumber": index, "instruction": f"Bước {index}."}

    instruction = str(step.get("instruction") or "").strip()
    if instruction:
        return {
            "stepNumber": step.get("stepNumber") or step.get("no") or index,
            "phase": step.get("phase", ""),
            "instruction": instruction,
        }

    # Rich fallback if model omitted the instruction field
    action = str(step.get("action") or "").strip()
    main = _join_parts(_clean_parts(step.get("main")))
    with_items = _join_parts(_clean_parts(step.get("with")))
    time_val = str(step.get("time") or "").strip()
    signal = str(step.get("signal") or "").strip()

    parts = []
    if action:
        parts.append(action.capitalize())
    if main:
        parts.append(main)
    if with_items:
        parts.append(f"cùng {with_items}")
    if time_val:
        parts.append(f"trong {time_val}")
    if signal:
        parts.append(f"cho đến khi {signal}")

    if parts:
        fallback_text = " ".join(parts) + "."
        if len(parts) == 1 and action:
            fallback_text = f"{action.capitalize()} các nguyên liệu theo hướng dẫn trong video."
    else:
        fallback_text = f"Thực hiện bước {index} theo hướng dẫn trong video."

    return {
        "stepNumber": step.get("stepNumber") or step.get("no") or index,
        "phase": step.get("phase", ""),
        "instruction": fallback_text,
    }


def render_recipe_steps(recipe: dict) -> dict:
    if not isinstance(recipe, dict):
        return recipe
    raw_steps = recipe.get("steps")
    if isinstance(raw_steps, list):
        recipe["steps"] = [render_step(step, index + 1) for index, step in enumerate(raw_steps)]
    return recipe


# ---------------------------------------------------------------------------
# LLM clients
# ---------------------------------------------------------------------------

groq_client = None
if GROQ_API_KEY and GROQ_API_KEY != "your_groq_api_key_here":
    try:
        from groq import Groq
        groq_client = Groq(api_key=GROQ_API_KEY)
        logger.info("Groq AI Client initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Groq Client: {e}")

genai_client = None
if GEMINI_API_KEY and GEMINI_API_KEY != "your_gemini_api_key_here":
    try:
        from google import genai
        genai_client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        logger.debug(f"google-genai SDK init note: {e}")


def _repair_json_fractions(raw: str) -> str:
    if not raw:
        return raw
    def replace_fraction(match):
        try:
            num = float(match.group(1))
            den = float(match.group(2))
            if den != 0:
                return f": {round(num / den, 2)}"
        except Exception:
            pass
        return ": 0"
    return re.sub(r':\s*(\d+)/(\d+)\b', replace_fraction, raw)


def call_llm(prompt: str) -> str:
    """
    Main LLM entrypoint:
    1. Try Groq (Llama 3.1 8B Instant - 14.4K RPD / 500K TPD Free).
    2. Fallback to Gemini API if Groq fails or key is missing.
    """
    if groq_client:
        groq_models = [GROQ_MODEL, "llama-3.1-8b-instant", "llama-3.3-70b-versatile"]
        groq_models = list(dict.fromkeys(groq_models))

        for model in groq_models:
            try:
                chat_completion = groq_client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": "Bạn là một đầu bếp chuyên nghiệp Việt Nam. Nhiệm vụ của bạn là trích xuất công thức nấu ăn CỰC KỲ CHI TIẾT, TỪNG BƯỚC CHẶT CHẼ từ nội dung video. CÚ PHÁP JSON BẮT BUỘC: Thuộc tính số lượng 'amount' CHỈ ĐƯỢC dùng số nguyên hoặc số thập phân (như 0.5, 0.33, 1.5, 2). TUYỆT ĐỐI KHÔNG viết dạng phân số 1/2 hay 1/3 vì sẽ gây lỗi cú pháp JSON."
                        },
                        {"role": "user", "content": prompt},
                    ],
                    model=model,
                    response_format={"type": "json_object"},
                    temperature=0.1,
                )
                content = chat_completion.choices[0].message.content
                if content:
                    logger.info(f"Groq ({model}) OK")
                    return content.strip()
            except Exception as e:
                logger.warning(f"Groq error '{model}': {e}")

    if GEMINI_API_KEY and GEMINI_API_KEY != "your_gemini_api_key_here":
        models_to_try = [GEMINI_MODEL, "gemini-2.0-flash-lite", "gemini-1.5-flash-8b"]
        models_to_try = list(dict.fromkeys(models_to_try))

        for model in models_to_try:
            if "pro" in model.lower():
                continue
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"responseMimeType": "application/json"}
            }
            try:
                res = requests.post(url, json=payload, timeout=25)
                if res.status_code == 200:
                    res_data = res.json()
                    candidates = res_data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts:
                            return parts[0].get("text", "").strip()
                elif res.status_code == 429:
                    time.sleep(1)
            except Exception as e:
                logger.error(f"Gemini error '{model}': {e}")

    raise Exception("No LLM provider available. Set GROQ_API_KEY in .env")


# ---------------------------------------------------------------------------
# Batch extraction
# ---------------------------------------------------------------------------

FEWSHOT_EXAMPLE = """Ví dụ (1 video mẫu → JSON đúng & chi tiết 100% người dùng không cần xem lại video):
VIDEO: "...thịt bò mua về rửa sạch với nước muối pha loãng, thái lát mỏng vừa ăn. Băm nhuyễn 3 tép tỏi và 1 củ hành tím. Cho thịt bò vào tô ướp với 1 thìa tỏi băm, 1 thìa hành băm, 2 thìa nước mắm, 1 thìa đường và 1 thìa tiêu trong 15 phút cho thấm. Bắc chảo lên bếp, cho 2 thìa dầu ăn đun nóng ở lửa vừa rồi phi thơm phần hành tỏi băm còn lại trong 1 phút đến khi ngả màu vàng nhạt. Tiếp theo trút thịt bò đã ướp vào xào đảo nhanh tay ở lửa lớn trong 3 phút đến khi thịt vừa chín tới và săn lại thì tắt bếp..."
→ steps: [
  {
    "no": 1,
    "phase": "prep",
    "action": "sơ chế & cắt thái",
    "main": ["thịt bò", "tỏi", "hành tím"],
    "with": ["nước muối"],
    "time": "",
    "signal": "",
    "instruction": "Sơ chế nguyên liệu: Thịt bò rửa sạch với nước muối pha loãng để khử mùi hôi, dùng khăn sạch thấm khô rồi thái lát mỏng vừa ăn. Tỏi và hành tím bóc vỏ, băm nhuyễn."
  },
  {
    "no": 2,
    "phase": "prep",
    "action": "ướp gia vị",
    "main": ["thịt bò"],
    "with": ["tỏi băm", "hành băm", "nước mắm", "đường", "tiêu"],
    "time": "15 phút",
    "signal": "thấm gia vị",
    "instruction": "Ướp thịt bò: Cho thịt bò vào tô lớn, nêm 1 thìa tỏi băm, 1 thìa hành băm, 2 thìa nước mắm, 1 thìa đường và 1 thìa tiêu xay. Trộn đều tay rồi để ướp trong 15 phút cho thịt ngấm sâu gia vị."
  },
  {
    "no": 3,
    "phase": "cook",
    "action": "phi thơm gia vị",
    "main": ["tỏi", "hành tím"],
    "with": ["dầu ăn"],
    "time": "1 phút",
    "signal": "ngả vàng thơm",
    "instruction": "Phi thơm hành tỏi: Đặt chảo lên bếp, cho 2 thìa dầu ăn vào đun nóng ở lửa vừa. Cho phần hành băm và tỏi băm còn lại vào đảo đều trong khoảng 1 phút đến khi dậy mùi thơm và ngả sang màu vàng nhạt."
  },
  {
    "no": 4,
    "phase": "cook",
    "action": "xào thịt bò",
    "main": ["thịt bò"],
    "with": [],
    "time": "3 phút",
    "signal": "thịt săn lại vừa chín tới",
    "instruction": "Xào thịt bò: Trút toàn bộ thịt bò đã ướp vào chảo, vặn lửa lớn và đảo thật nhanh tay trong 3 phút đến khi thịt chuyển màu, săn lại và vừa chín tới thì tắt bếp ngay để thịt giữ độ mềm ngọt, không bị nhảy hay ráo nước."
  }
]"""


def batch_extract_recipes(videos: list, total_context_budget: int = 24000) -> list:
    """
    Extract recipes from multiple videos in a SINGLE LLM call.
    Each video has: title, description, transcript.
    Returns a list of structured recipe JSON objects.
    """
    if not videos:
        return []

    per_video_budget = min(9000, max(2400, total_context_budget // len(videos)))
    video_blocks = []
    video_sources = []  # keep raw content per video for post-hoc grounding check
    for i, v in enumerate(videos):
        transcript = (v.get("transcript") or "").strip()
        description = (v.get("description") or "").strip()

        content = build_recipe_context(transcript, description, max_chars=per_video_budget)
        if not content:
            content = v.get("title", "Video")
        video_sources.append(content)

        video_blocks.append(
            f"=== VIDEO {i+1}: {v.get('title', '')} ===\n{content}\n=== KẾT THÚC VIDEO {i+1} ==="
        )

    all_videos_text = "\n\n".join(video_blocks)
    num_videos = len(videos)

    prompt = f"""{FEWSHOT_EXAMPLE}

Trích xuất công thức chi tiết từ {num_videos} video dưới đây. Trả về đúng {num_videos} object theo đúng thứ tự VIDEO.

ĐẶC BIỆT LƯU Ý - ĐÂY LÀ ĐIỂM NỔI BẬT CỦA ỨNG DỤNG:
Người dùng đọc công thức này KHÔNG CẦN COI LẠI VIDEO YOUTUBE vẫn tự nấu thành công 100%. 
Vì vậy, tất cả các bước chế biến phải cực kỳ CHI TIẾT, RÕ RÀNG, CHẶT CHẼ và ĐẦY ĐỦ theo đúng trình tự hướng dẫn thực tế của người nấu trong VIDEO.

Quy tắc BẮT BUỘC cho mỗi bước ("instruction"):
1. Trình tự logic từ A-Z (5-12 bước): Bắt đầu từ sơ chế làm sạch nguyên liệu -> ướp gia vị (nêu rõ tỷ lệ/muỗng nếu video nhắc tới) -> chế biến từng công đoạn (lửa lớn/nhỏ, thứ tự thả nguyên liệu/gia vị, thời gian đun xào kho) -> hoàn thành và trình bày.
2. Chi tiết từng hành động: Mỗi bước "instruction" phải viết thành 2-4 câu tiếng Việt tự nhiên, đầy đủ (làm gì, bằng dụng cụ gì, ngọn lửa ra sao, thêm gia vị gì, thời gian bao nhiêu phút, dấu hiệu thị giác/khứu giác khi nào hoàn thành).
3. Mẹo & Bí quyết: Đưa toàn bộ mẹo nhỏ của người nấu trong video vào đúng bước tương ứng (ví dụ: mẹo khử mùi, cách giữ màu xanh của rau, bí quyết nước sốt sánh mịn).
4. Chuẩn mực dữ kiện: Tuyệt đối CHỈ trích xuất thông tin có trong VIDEO (không tự bịa thiết bị hay nguyên liệu không có). Nhưng phải khai thác TỐI ĐA và TRỌN VẸN toàn bộ lời thoại và mô tả trong video.

Quy tắc ĐƠN VỊ ĐO LƯỜNG TIẾNG VIỆT cho "unit":
- BẮT BUỘC dùng đơn vị thuần Việt tự nhiên (muỗng canh, muỗng cà phê, g, kg, ml, tép, củ, quả, trái, chén, lát, miếng, nhúm). TUYỆT ĐỐI KHÔNG dùng từ viết tắt tiếng Anh như Tsp, Tbsp, cup, tablespoon, teaspoon.

Quy tắc BẮT BUỘC về Tiêu đề ("title") và Mô tả ("description"):
- "title": BẮT BUỘC phải là TÊN CỦA 1 MÓN ĂN ĐƠN LẺ CỤ THỂ (Ví dụ: "Trứng Nướng Hàn Quốc", "Trứng Hấp Vân Ngũ Sắc", "Thịt Bò Xào Cần Tây"). 
  TUYỆT ĐỐI KHÔNG dùng tên tổng hợp danh sách nhiều món như "8 Món Ngon Từ...", "Top 5 Món...", "Tổng hợp các món...", "Bài viết giới thiệu...".
  Nếu video gốc dạy nhiều món, CHỈ trích xuất món ăn chính/nổi bật nhất thành 1 công thức món ăn đơn lẻ duy nhất.
- "description": 2-3 câu ngắn gọn giới thiệu hương vị đặc sắc của món ăn cụ thể đó. KHÔNG dùng câu văn kiểu "Bài viết giới thiệu..." hay "Video tổng hợp các món...".

{all_videos_text}

Trả về CHÍNH XÁC JSON:
```json
{{
  "recipes": [
    {{
      "title": "Tên món ăn sạch đẹp",
      "description": "Tóm tắt hấp dẫn 2-3 câu giới thiệu hương vị đặc sắc của món ăn và điểm đặc biệt khi chế biến theo video",
      "ingredients": [
        {{ "name": "Tên nguyên liệu", "amount": 500, "unit": "g", "category": "thịt/rau/gia vị/khác" }}
      ],
      "steps": [
        {{
          "no": 1,
          "phase": "prep",
          "action": "hành động chính (ví dụ: sơ chế & cắt thái)",
          "main": ["nguyên liệu chính"],
          "with": ["nguyên liệu phụ/gia vị"],
          "time": "thời gian nếu có (ví dụ: 15 phút)",
          "signal": "dấu hiệu nhận biết hoàn thành (ví dụ: thấm gia vị / thơm ngả vàng / chín mềm)",
          "instruction": "Câu hướng dẫn cực kỳ CHI TIẾT và RÕ RÀNG cho bước này, mô tả trọn vẹn thao tác, ngọn lửa, thời gian và bí quyết trong video."
        }}
      ],
      "tags": ["tag1", "tag2"]
    }}
  ]
}}
```
"""

    try:
        raw = call_llm(prompt)
        raw = _repair_json_fractions(raw)
        data = json.loads(raw)
        recipes = data.get("recipes", [])
        if not isinstance(recipes, list):
            recipes = [data] if "title" in data else []

        processed = []
        for i, recipe in enumerate(recipes):
            if not isinstance(recipe, dict):
                processed.append(recipe)
                continue
            source_text = video_sources[i] if i < len(video_sources) else ""
            raw_steps = recipe.get("steps")
            if isinstance(raw_steps, list):
                raw_steps = [s for s in raw_steps if not _is_invalid_step(s)]
                recipe["steps"] = [verify_step_grounding(s, source_text) for s in raw_steps]
                if not has_cook_phase(recipe["steps"]):
                    logger.warning(f"Recipe '{recipe.get('title', '')}' has no cook-phase step — check source video")
            recipe = reconcile_names(recipe)
            recipe = render_recipe_steps(recipe)
            processed.append(recipe)

        return processed

    except Exception as e:
        logger.error(f"batch_extract_recipes error: {e}")
        return []


def extract_recipe(text_content: str, schema: dict = None) -> dict:
    """Single video extract (backward compatible). Delegates to batch."""
    results = batch_extract_recipes([{"title": "", "description": "", "transcript": text_content}])
    if results and len(results) > 0:
        return {"data": results[0]}
    return {"data": {"title": "", "description": "", "ingredients": [], "steps": [], "tags": []}}