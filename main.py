from __future__ import annotations

"""Telegram bot for data-driven SAT Math and IELTS vocabulary practice."""
import json
import logging
import math
import os
import random
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Final

# Настройка директории временных файлов matplotlib для Render
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon

from flask import Flask
from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    ReplyKeyboardMarkup,
    Update,
)
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# --- ИНИЦИАЛИЗАЦИЯ FLASK ДЛЯ RENDER ---
app = Flask("")

@app.route("/")
def home():
    return "Bot is active and running 24/7!"

def run_flask():
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

# --- ЛОГИРОВАНИЕ ---
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# --- КОНСТАНТЫ ---
MENU_SAT: Final = "Practice SAT Math"
MENU_IELTS: Final = "Practice IELTS Vocab"
MENU_SAT_EXAM: Final = "📝 Take SAT Mock Exam"
MENU_IELTS_EXAM: Final = "📝 Take IELTS Mock Exam"
SAT_MODULE_QUESTIONS: Final = 22
IELTS_EXAM_QUESTIONS: Final = 40
MENU_BUTTONS: Final = [
    [MENU_SAT, MENU_IELTS],
    [MENU_SAT_EXAM, MENU_IELTS_EXAM],
]
MENU_TO_CATEGORY: Final = {
    MENU_SAT: "sat_math",
    MENU_IELTS: "ielts_vocab",
}
CATEGORY_TO_MENU: Final = {value: key for key, value in MENU_TO_CATEGORY.items()}
QUESTIONS_FILE: Final = Path(__file__).with_name("questions.json")
USERS_FILE: Final = Path(__file__).with_name("users.json")
ERRORS_FILE: Final = Path(__file__).with_name("errors.json")
ADMIN_USER_ID: Final = 5311931120
SEEN_IDS_KEY: Final = "seen_question_ids"


@dataclass(frozen=True)
class Question:
    question_id: str
    category: str
    prompt: str
    options: tuple[str, ...]
    correct_index: int
    explanation: str
    tags: tuple[str, ...] = ()
    diagram: dict[str, object] | None = None


class CategoryCompleteError(Exception):
    """Raised when a category has no unseen questions in the current deck."""

    def __init__(self, category: str) -> None:
        self.category = category


def _diagram_point(value: object, field_name: str) -> tuple[float, float]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or not all(isinstance(item, (int, float)) for item in value)
    ):
        raise ValueError(f"Diagram field {field_name!r} must be a two-number point.")
    return float(value[0]), float(value[1])


def _set_geometry_axes(ax, x_values: list[float], y_values: list[float]) -> None:
    if not x_values or not y_values:
        return
    x_span = max(max(x_values) - min(x_values), 1.0)
    y_span = max(max(y_values) - min(y_values), 1.0)
    x_pad = x_span * 0.2
    y_pad = y_span * 0.2
    ax.set_xlim(min(x_values) - x_pad, max(x_values) + x_pad)
    ax.set_ylim(min(y_values) - y_pad, max(y_values) + y_pad)


def draw_geometry_figure(question: Question) -> Path:
    """Render a tagged question's diagram to a temporary PNG file."""
    if question.diagram is None:
        raise ValueError(f"Question {question.question_id} has no diagram data.")

    diagram_type = question.diagram.get("type")
    figure, ax = plt.subplots(figsize=(5.4, 4.2), dpi=150)
    figure.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.axhline(0, color="#9ca3af", linewidth=0.8)
    ax.axvline(0, color="#9ca3af", linewidth=0.8)
    ax.set_aspect("equal", adjustable="box")

    try:
        if diagram_type == "triangle":
            raw_points = question.diagram.get("points")
            if not isinstance(raw_points, list) or len(raw_points) != 3:
                raise ValueError("Triangle diagrams require exactly three points.")
            points = [_diagram_point(point, "points") for point in raw_points]
            polygon = Polygon(
                points,
                closed=True,
                fill=False,
                edgecolor="#2563eb",
                linewidth=2.2,
            )
            ax.add_patch(polygon)
            labels = question.diagram.get("labels", [])
            if isinstance(labels, list):
                for point, label in zip(points, labels):
                    ax.annotate(
                        str(label),
                        point,
                        xytext=(5, 5),
                        textcoords="offset points",
                        color="#111827",
                        fontsize=10,
                    )
            _set_geometry_axes(
                ax,
                [point[0] for point in points],
                [point[1] for point in points],
            )

        elif diagram_type == "circle":
            center = _diagram_point(question.diagram.get("center"), "center")
            radius = question.diagram.get("radius")
            if not isinstance(radius, (int, float)) or radius <= 0:
                raise ValueError("Circle diagrams require a positive radius.")
            ax.add_patch(
                Circle(
                    center,
                    float(radius),
                    fill=False,
                    edgecolor="#2563eb",
                    linewidth=2.2,
                )
            )
            ax.plot(center[0], center[1], "o", color="#dc2626", markersize=4)
            ax.annotate(
                "center",
                center,
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
            )
            ax.plot(
                [center[0], center[0] + float(radius)],
                [center[1], center[1]],
                color="#dc2626",
                linewidth=1.4,
            )
            _set_geometry_axes(
                ax,
                [center[0] - float(radius), center[0] + float(radius)],
                [center[1] - float(radius), center[1] + float(radius)],
            )

        elif diagram_type == "coordinate_lines":
            raw_points = question.diagram.get("points")
            if not isinstance(raw_points, list) or len(raw_points) != 2:
                raise ValueError("Coordinate-line diagrams require exactly two points.")
            first, second = (
                _diagram_point(raw_points[0], "points"),
                _diagram_point(raw_points[1], "points"),
            )
            ax.plot(
                [first[0], second[0]],
                [first[1], second[1]],
                color="#2563eb",
                linewidth=2.2,
            )
            ax.scatter(
                [first[0], second[0]],
                [first[1], second[1]],
                color="#dc2626",
                zorder=3,
            )
            labels = question.diagram.get("labels", ["A", "B"])
            if isinstance(labels, list):
                for point, label in zip((first, second), labels):
                    ax.annotate(
                        str(label),
                        point,
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=10,
                    )
            _set_geometry_axes(
                ax,
                [first[0], second[0]],
                [first[1], second[1]],
            )
            ax.set_xlabel("x")
            ax.set_ylabel("y")

        elif diagram_type == "intersecting_lines":
            raw_lines = question.diagram.get("lines")
            x_limits = question.diagram.get("xlim", [-5, 5])
            if not isinstance(raw_lines, list) or not isinstance(x_limits, list):
                raise ValueError(
                    "Intersecting-line diagrams require lines and xlim arrays."
                )
            if len(x_limits) != 2 or not all(
                isinstance(value, (int, float)) for value in x_limits
            ):
                raise ValueError("Diagram xlim must contain two numbers.")
            x_start, x_end = float(x_limits[0]), float(x_limits[1])
            x_values = [x_start + (x_end - x_start) * step / 100 for step in range(101)]
            y_values: list[float] = []
            for line in raw_lines:
                if not isinstance(line, dict):
                    raise ValueError("Each intersecting line must be an object.")
                slope = line.get("slope")
                intercept = line.get("intercept")
                if not isinstance(slope, (int, float)) or not isinstance(
                    intercept, (int, float)
                ):
                    raise ValueError("Each line requires numeric slope and intercept.")
                current_y = [float(slope) * x + float(intercept) for x in x_values]
                y_values.extend(current_y)
                ax.plot(x_values, current_y, linewidth=2.0, label=line.get("label"))
            _set_geometry_axes(ax, x_values, y_values)
            if raw_lines and all(
                isinstance(line, dict) and line.get("label") for line in raw_lines
            ):
                ax.legend(frameon=False)
            ax.set_xlabel("x")
            ax.set_ylabel("y")

        else:
            raise ValueError(f"Unsupported geometry diagram type: {diagram_type!r}")

        ax.set_title(str(question.diagram.get("title", "")))
        figure.tight_layout()
        temporary_file = tempfile.NamedTemporaryFile(
            prefix="study_sprint_geometry_",
            suffix=".png",
            delete=False,
        )
        temporary_path = Path(temporary_file.name)
        temporary_file.close()
        figure.savefig(temporary_path, format="png", bbox_inches="tight")
        return temporary_path
    finally:
        plt.close(figure)


async def send_question_message(
    bot,
    chat_id: int,
    question: Question,
    text: str,
    reply_markup: InlineKeyboardMarkup,
) -> None:
    """Send a question as text or a temporary geometry image."""
    if "geometry" not in question.tags or question.diagram is None:
        await bot.send_message(
            chat_id=chat_id,
            text=text,
            reply_markup=reply_markup,
        )
        return

    temporary_path = draw_geometry_figure(question)
    try:
        with temporary_path.open("rb") as image_file:
            await bot.send_photo(
                chat_id=chat_id,
                photo=image_file,
                caption=text,
                reply_markup=reply_markup,
            )
    finally:
        temporary_path.unlink(missing_ok=True)


def load_registered_user_ids() -> set[int]:
    """Safely load user IDs, creating users.json if missing."""
    if not USERS_FILE.exists():
        USERS_FILE.write_text("[]\n", encoding="utf-8")
        return set()
    try:
        raw_users = json.loads(USERS_FILE.read_text(encoding="utf-8"))
        if isinstance(raw_users, list):
            return set(int(uid) for uid in raw_users if isinstance(uid, (int, str)))
    except Exception as error:
        logger.warning("Could not read users.json: %s", error)
    return set()


def save_registered_user_ids() -> None:
    """Persist the user ID set as a stable JSON array."""
    try:
        USERS_FILE.write_text(
            json.dumps(sorted(REGISTERED_USER_IDS), indent=2) + "\n",
            encoding="utf-8",
        )
    except Exception as error:
        logger.error("Failed to save users.json: %s", error)


def register_user(user_id: int) -> None:
    """Add a user to the registry once and persist new registrations."""
    if user_id not in REGISTERED_USER_IDS:
        REGISTERED_USER_IDS.add(user_id)
        save_registered_user_ids()


def load_error_bank() -> dict[str, list[dict[str, object]]]:
    """Safely load incorrect-answer records, creating errors.json if missing."""
    if not ERRORS_FILE.exists():
        ERRORS_FILE.write_text("{}\n", encoding="utf-8")
        return {}
    try:
        raw_errors = json.loads(ERRORS_FILE.read_text(encoding="utf-8"))
        if isinstance(raw_errors, dict):
            return raw_errors
    except Exception as error:
        logger.warning("Could not read errors.json: %s", error)
    return {}


def save_error_bank() -> None:
    """Persist incorrect-answer records."""
    try:
        ERRORS_FILE.write_text(
            json.dumps(ERROR_BANK, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except Exception as error:
        logger.error("Failed to save errors.json: %s", error)


def record_user_error(
    update: Update,
    question: Question,
    selected_index: int,
    source: str,
) -> dict[str, object] | None:
    """Persist one incorrect answer and return its review-ready record."""
    user = getattr(update, "effective_user", None)
    if user is None:
        return None

    record: dict[str, object] = {
        "source": source,
        "question_id": question.question_id,
        "prompt": question.prompt,
        "your_answer": question.options[selected_index],
        "correct_answer": question.options[question.correct_index],
        "explanation": question.explanation,
    }
    ERROR_BANK.setdefault(str(user.id), []).append(record)
    save_error_bank()
    return record


def load_questions() -> dict[str, tuple[Question, ...]]:
    """Load and validate the question bank from questions.json."""
    if not QUESTIONS_FILE.exists():
        raise RuntimeError(f"Missing required file: {QUESTIONS_FILE}")

    try:
        raw_questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    except Exception as error:
        raise RuntimeError(f"Invalid JSON in {QUESTIONS_FILE}: {error}") from error

    if not isinstance(raw_questions, list) or not raw_questions:
        raise RuntimeError("questions.json must contain a non-empty JSON array.")

    questions_by_category: dict[str, list[Question]] = {
        category: [] for category in CATEGORY_TO_MENU
    }
    question_ids: set[str] = set()

    for position, raw_question in enumerate(raw_questions, start=1):
        if not isinstance(raw_question, dict):
            raise RuntimeError(f"Question {position} must be a JSON object.")

        required_fields = {
            "id",
            "category",
            "prompt",
            "options",
            "correct_index",
            "explanation",
        }
        missing_fields = required_fields.difference(raw_question)
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise RuntimeError(f"Question {position} is missing: {missing}.")

        question_id = raw_question["id"]
        category = raw_question["category"]
        prompt = raw_question["prompt"]
        options = raw_question["options"]
        correct_index = raw_question["correct_index"]
        explanation = raw_question["explanation"]
        tags = raw_question.get("tags", [])
        diagram = raw_question.get("diagram")

        if not isinstance(question_id, str) or not question_id.strip():
            raise RuntimeError(f"Question {position} has an invalid id.")
        if question_id in question_ids:
            raise RuntimeError(f"Duplicate question id: {question_id}.")
        if category not in questions_by_category:
            raise RuntimeError(f"Question {question_id} has an unknown category.")
        if not isinstance(prompt, str) or not prompt.strip():
            raise RuntimeError(f"Question {question_id} has an empty prompt.")
        if (
            not isinstance(options, list)
            or len(options) != 4
            or not all(isinstance(option, str) and option.strip() for option in options)
            or len(set(options)) != len(options)
        ):
            raise RuntimeError(
                f"Question {question_id} must have four unique, non-empty options."
            )
        if not isinstance(correct_index, int) or correct_index not in range(4):
            raise RuntimeError(f"Question {question_id} has an invalid correct_index.")
        if not isinstance(explanation, str) or not explanation.strip():
            raise RuntimeError(f"Question {question_id} has an empty explanation.")

        option_pairs = list(enumerate(options))
        random.shuffle(option_pairs)
        shuffled_options = tuple(option for _, option in option_pairs)
        shuffled_correct_index = next(
            index
            for index, (original_index, _) in enumerate(option_pairs)
            if original_index == correct_index
        )

        question_ids.add(question_id)
        questions_by_category[category].append(
            Question(
                question_id=question_id,
                category=category,
                prompt=prompt,
                options=shuffled_options,
                correct_index=shuffled_correct_index,
                explanation=explanation,
                tags=tuple(tags),
                diagram=diagram,
            )
        )

    return {
        category: tuple(questions)
        for category, questions in questions_by_category.items()
    }


# Инициализация глобальных хранилищ
QUESTION_BANK: Final = load_questions()
QUESTION_BY_ID: Final = {
    question.question_id: question
    for questions in QUESTION_BANK.values()
    for question in questions
}
TOTAL_QUESTIONS: Final = len(QUESTION_BY_ID)
REGISTERED_USER_IDS: set[int] = load_registered_user_ids()
ERROR_BANK: dict[str, list[dict[str, object]]] = load_error_bank()


def main_menu() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        MENU_BUTTONS,
        resize_keyboard=True,
        is_persistent=True,
    )


def question_keyboard(
    question: Question,
    callback_prefix: str = "answer",
    include_stop: bool = False,
) -> InlineKeyboardMarkup:
    keyboard = [
        [
            InlineKeyboardButton(
                option,
                callback_data=f"{callback_prefix}:{question.question_id}:{index}",
            )
        ]
        for index, option in enumerate(question.options)
    ]
    if include_stop:
        keyboard.append(
            [InlineKeyboardButton("🛑 Stop Exam", callback_data="exam_stop")]
        )
    return InlineKeyboardMarkup(keyboard)


def seen_question_ids(context: ContextTypes.DEFAULT_TYPE) -> set[str]:
    seen = context.user_data.setdefault(SEEN_IDS_KEY, set())
    if not isinstance(seen, set):
        seen = set(seen) if isinstance(seen, (list, tuple)) else set()
        context.user_data[SEEN_IDS_KEY] = seen
    return seen


def next_question(
    category: str, context: ContextTypes.DEFAULT_TYPE
) -> tuple[Question, bool]:
    """Choose a random unseen question."""
    seen = seen_question_ids(context)
    available = [
        question
        for question in QUESTION_BANK[category]
        if question.question_id not in seen
    ]
    started_new_round = False

    if not available:
        if len(seen) < TOTAL_QUESTIONS:
            raise CategoryCompleteError(category)
        seen.clear()
        context.user_data["answered_questions"] = 0
        context.user_data["correct_answers"] = 0
        context.user_data["round_number"] = context.user_data.get("round_number", 1) + 1
        available = list(QUESTION_BANK[category])
        started_new_round = True

    question = random.choice(available)
    seen.add(question.question_id)
    context.user_data["active_question_id"] = question.question_id
    context.user_data["active_category"] = category
    return question, started_new_round


def score_text(context: ContextTypes.DEFAULT_TYPE) -> str:
    correct = context.user_data.get("correct_answers", 0)
    answered = context.user_data.get("answered_questions", 0)
    return f"Score: {correct}/{answered}" if answered else "Score: 0/0"


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_user is not None:
        register_user(update.effective_user.id)
    context.user_data.clear()
    await update.message.reply_text(
        f"Welcome to Study Sprint.\n\n"
        f"I have {TOTAL_QUESTIONS} questions ready. Choose a practice area:",
        reply_markup=main_menu(),
    )


async def show_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Choose a practice area:",
        reply_markup=main_menu(),
    )


async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_user is None or update.effective_user.id != ADMIN_USER_ID:
        await update.effective_message.reply_text("Unauthorized")
        return

    await update.effective_message.reply_text(
        f"Total questions in questions.json: {TOTAL_QUESTIONS}\n"
        f"Total unique users registered: {len(REGISTERED_USER_IDS)}"
    )


async def send_question(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    menu_label: str,
) -> None:
    active_question_id = context.user_data.get("active_question_id")
    if active_question_id:
        active_question = QUESTION_BY_ID.get(active_question_id)
        if active_question:
            await send_question_message(
                context.bot,
                update.effective_chat.id,
                active_question,
                "Please answer the current question before starting another one:",
                question_keyboard(active_question),
            )
            return

    category = MENU_TO_CATEGORY[menu_label]
    try:
        question, started_new_round = next_question(category, context)
    except CategoryCompleteError:
        seen = seen_question_ids(context)
        remaining = TOTAL_QUESTIONS - len(seen)
        await update.message.reply_text(
            f"You've answered every {menu_label.removeprefix('Practice ')} question "
            f"in this round.\n\n"
            f"{remaining} questions remain in the full deck. Choose the other "
            "practice area to continue without repeats.",
            reply_markup=main_menu(),
        )
        return

    round_notice = (
        "\n\nYou completed the full question bank. Starting a fresh round!"
        if started_new_round
        else ""
    )
    await send_question_message(
        context.bot,
        update.effective_chat.id,
        question,
        f"{menu_label}\n\n{question.prompt}\n\n{score_text(context)}{round_notice}",
        question_keyboard(question),
    )


async def menu_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    menu_label = update.message.text
    if menu_label not in MENU_TO_CATEGORY:
        await update.message.reply_text(
            "Please choose one of the practice areas below.",
            reply_markup=main_menu(),
        )
        return
    await send_question(update, context, menu_label)


async def answer_question(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    if query.data == "exam_stop":
        await stop_exam(update, context)
        return
    if query.data == "review_mistakes":
        await start_mistake_review(update, context)
        return
    if query.data == "review_next":
        context.user_data["review_index"] = context.user_data.get("review_index", 0) + 1
        await show_mistake_review(update, context)
        return
    if query.data == "review_done":
        await finish_mistake_review(update, context)
        return

    try:
        callback_prefix, question_id, selected_index_text = query.data.split(
            ":", maxsplit=2
        )
        selected_index = int(selected_index_text)
    except (AttributeError, TypeError, ValueError):
        await query.edit_message_text("That answer is no longer available.")
        return

    if callback_prefix == "exam_answer":
        await answer_exam_question(update, context, question_id, selected_index)
        return

    if callback_prefix != "answer" or question_id not in QUESTION_BY_ID:
        await query.edit_message_text("That answer is no longer available.")
        return

    question = QUESTION_BY_ID[question_id]
    if context.user_data.get("active_question_id") != question_id:
        await query.edit_message_text(
            "This question has expired. Choose a practice area for a new one.",
            reply_markup=None,
        )
        return

    if selected_index not in range(len(question.options)):
        await query.edit_message_text("That answer is not available.")
        return

    context.user_data["active_question_id"] = None
    context.user_data["answered_questions"] = (
        context.user_data.get("answered_questions", 0) + 1
    )
    is_correct = selected_index == question.correct_index
    if is_correct:
        context.user_data["correct_answers"] = (
            context.user_data.get("correct_answers", 0) + 1
        )
    else:
        record_user_error(update, question, selected_index, source="practice")

    result = "Correct!" if is_correct else "Not quite."
    correct_option = question.options[question.correct_index]
    menu_label = CATEGORY_TO_MENU[question.category]
    await query.edit_message_text(
        f"{result}\n\n"
        f"Correct answer: {correct_option}\n"
        f"Step-by-step explanation:\n{question.explanation}\n\n"
        f"{score_text(context)}\n\n"
        f"Choose {menu_label} or the other practice area for another question.",
        reply_markup=None,
    )
    if query.message:
        await query.message.reply_text("Keep practicing:", reply_markup=main_menu())


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Unhandled Telegram update error", exc_info=context.error)


def choose_exam_question_ids(
    context: ContextTypes.DEFAULT_TYPE,
    category: str,
    count: int,
    route: str | None = None,
) -> list[str]:
    used_ids = context.user_data.setdefault("exam_used_question_ids", set())
    if not isinstance(used_ids, set):
        used_ids = set(used_ids)
        context.user_data["exam_used_question_ids"] = used_ids

    remaining = [
        question
        for question in QUESTION_BANK[category]
        if question.question_id not in used_ids
    ]
    if route in {"hard", "standard"} and category == "sat_math":
        midpoint = len(QUESTION_BANK[category]) // 2
        preferred_pool = (
            QUESTION_BANK[category][midpoint:]
            if route == "hard"
            else QUESTION_BANK[category][:midpoint]
        )
        preferred_remaining = [
            question
            for question in preferred_pool
            if question.question_id not in used_ids
        ]
        if len(preferred_remaining) >= count:
            remaining = preferred_remaining

    if len(remaining) < count:
        raise RuntimeError(
            f"Not enough unused {category} questions for this exam segment."
        )

    selected = random.sample(remaining, count)
    used_ids.update(question.question_id for question in selected)
    return [question.question_id for question in selected]


async def start_sat_exam_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    context.user_data.clear()
    context.user_data["exam_type"] = "sat"
    context.user_data["current_module"] = 1
    context.user_data["m1_score"] = 0
    context.user_data["m2_score"] = 0
    context.user_data["correct_answers"] = 0
    context.user_data["exam_answered_questions"] = 0
    context.user_data["exam_incorrect_answers"] = []
    context.user_data["question_index"] = 0
    context.user_data["total_questions"] = SAT_MODULE_QUESTIONS
    context.user_data["module_total_questions"] = SAT_MODULE_QUESTIONS
    context.user_data["exam_used_question_ids"] = set()
    context.user_data["exam_question_ids"] = choose_exam_question_ids(
        context, "sat_math", SAT_MODULE_QUESTIONS
    )
    if update.message:
        await update.message.reply_text(
            "🚀 *Starting Digital SAT Math Mock Exam*\n\n"
            "• *Module 1:* 22 Questions\n"
            "• *Module 2:* Adaptive based on M1 score\n\n"
            "Good luck!",
            parse_mode="Markdown",
        )
        await send_next_exam_question(update.message, context)


async def start_ielts_exam_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    context.user_data.clear()
    context.user_data["exam_type"] = "ielts"
    context.user_data["current_module"] = 1
    context.user_data["m1_score"] = 0
    context.user_data["correct_answers"] = 0
    context.user_data["exam_answered_questions"] = 0
    context.user_data["exam_incorrect_answers"] = []
    context.user_data["question_index"] = 0
    context.user_data["total_questions"] = IELTS_EXAM_QUESTIONS
    context.user_data["module_total_questions"] = IELTS_EXAM_QUESTIONS
    context.user_data["exam_used_question_ids"] = set()
    context.user_data["exam_question_ids"] = choose_exam_question_ids(
        context, "ielts_vocab", IELTS_EXAM_QUESTIONS
    )
    if update.message:
        await update.message.reply_text(
            "🚀 *Starting IELTS Reading Mock Exam*\n\n"
            "• *Total:* 40 Questions\n\n"
            "Good luck!",
            parse_mode="Markdown",
        )
        await send_next_exam_question(update.message, context)


def current_exam_question(context: ContextTypes.DEFAULT_TYPE) -> Question | None:
    exam_type = context.user_data.get("exam_type")
    question_ids = context.user_data.get("exam_question_ids")
    question_index = context.user_data.get("question_index")

    if (
        exam_type not in {"sat", "ielts"}
        or not isinstance(question_ids, list)
        or not isinstance(question_index, int)
    ):
        return None

    if question_index < 0 or question_index >= len(question_ids):
        return None

    return QUESTION_BY_ID.get(question_ids[question_index])


async def send_next_exam_question(message, context: ContextTypes.DEFAULT_TYPE) -> None:
    question = current_exam_question(context)
    if question is None:
        return

    exam_type = context.user_data["exam_type"]
    question_index = context.user_data["question_index"]
    total_questions = context.user_data["module_total_questions"]
    if exam_type == "sat":
        module = context.user_data["current_module"]
        title = f"Digital SAT Math — Module {module}"
    else:
        title = "IELTS Reading Mock Exam"

    await send_question_message(
        context.bot,
        message.chat_id,
        question,
        f"{title}\n\n"
        f"Question {question_index + 1} of {total_questions}\n\n"
        f"{question.prompt}",
        question_keyboard(
            question,
            callback_prefix="exam_answer",
            include_stop=True,
        ),
    )


async def answer_exam_question(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    question_id: str,
    selected_index: int,
) -> None:
    query = update.callback_query
    question = QUESTION_BY_ID.get(question_id)
    expected_question = current_exam_question(context)

    if (
        question is None
        or expected_question is None
        or expected_question.question_id != question_id
    ):
        await query.edit_message_text(
            "This mock-exam question has expired. Start a new mock exam from the menu."
        )
        return

    if selected_index not in range(len(question.options)):
        await query.edit_message_text("That answer is not available.")
        return

    exam_type = context.user_data["exam_type"]
    module = context.user_data.get("current_module", 1)
    is_correct = selected_index == question.correct_index
    context.user_data["exam_answered_questions"] += 1
    if is_correct:
        context.user_data["correct_answers"] += 1
        score_key = "m1_score" if exam_type == "ielts" or module == 1 else "m2_score"
        context.user_data[score_key] += 1
    else:
        error_record = record_user_error(
            update, question, selected_index, source=f"mock_{exam_type}"
        )
        if error_record is not None:
            context.user_data["exam_incorrect_answers"].append(error_record)

    context.user_data["question_index"] += 1
    question_index = context.user_data["question_index"]
    module_total = context.user_data["module_total_questions"]

    await query.edit_message_text(
        f"Answer recorded.\n\nProgress: {question_index}/{module_total}",
        reply_markup=None,
    )

    if not query.message:
        return

    if exam_type == "sat" and question_index >= SAT_MODULE_QUESTIONS:
        if module == 1:
            route = (
                "Hard Module 2"
                if context.user_data["m1_score"] >= 15
                else "Standard Module 2"
            )
            context.user_data["current_module"] = 2
            context.user_data["question_index"] = 0
            route_key = "hard" if context.user_data["m1_score"] >= 15 else "standard"
            context.user_data["exam_route"] = route_key
            context.user_data["exam_question_ids"] = choose_exam_question_ids(
                context, "sat_math", SAT_MODULE_QUESTIONS, route=route_key
            )
            await query.message.reply_text(f"Module 1 complete. Starting {route}.")
            await send_next_exam_question(query.message, context)
        else:
            await send_exam_results(update, context, "sat")
        return

    if exam_type == "ielts" and question_index >= IELTS_EXAM_QUESTIONS:
        await send_exam_results(update, context, "ielts")
        return

    await send_next_exam_question(query.message, context)


async def stop_exam(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    exam_type = context.user_data.get("exam_type")
    if exam_type not in {"sat", "ielts"}:
        await query.edit_message_text("There is no active mock exam to stop.")
        return

    answered = context.user_data.get("exam_answered_questions", 0)
    correct = context.user_data.get("correct_answers", 0)
    if exam_type == "sat":
        m1_score = context.user_data.get("m1_score", 0)
        m2_score = context.user_data.get("m2_score", 0)
        summary = (
            "🛑 *SAT Mock Exam Stopped*\n\n"
            f"Questions answered: {answered}\n"
            f"Module 1 score: {m1_score}/{SAT_MODULE_QUESTIONS}\n"
            f"Module 2 score: {m2_score}/{SAT_MODULE_QUESTIONS}\n"
            f"Total correct: {correct}"
        )
    else:
        summary = (
            "🛑 *IELTS Mock Exam Stopped*\n\n"
            f"Questions answered: {answered}/{IELTS_EXAM_QUESTIONS}\n"
            f"Correct answers: {correct}"
        )

    context.user_data.clear()
    await query.edit_message_text(summary, parse_mode="Markdown")
    if query.message:
        await query.message.reply_text("Main menu:", reply_markup=main_menu())


def mistake_review_keyboard(index: int, total: int) -> InlineKeyboardMarkup:
    if index + 1 < total:
        label, callback_data = "➡️ Next Mistake", "review_next"
    else:
        label, callback_data = "✅ Finish Review", "review_done"
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton(label, callback_data=callback_data)]]
    )


async def show_mistake_review(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    query = update.callback_query
    mistakes = context.user_data.get("exam_incorrect_answers", [])
    index = context.user_data.get("review_index", 0)
    if not isinstance(mistakes, list) or not mistakes:
        await query.edit_message_text("There are no mistakes to review.")
        return
    if not isinstance(index, int) or index < 0 or index >= len(mistakes):
        await query.edit_message_text("The mistake review has ended.")
        return

    mistake = mistakes[index]
    await query.edit_message_text(
        f"📝 Mistake {index + 1} of {len(mistakes)}\n\n"
        f"Question:\n{mistake['prompt']}\n\n"
        f"Your answer: {mistake['your_answer']}\n"
        f"Correct answer: {mistake['correct_answer']}\n\n"
        f"Step-by-step explanation:\n{mistake['explanation']}",
        reply_markup=mistake_review_keyboard(index, len(mistakes)),
    )


async def start_mistake_review(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    mistakes = context.user_data.get("exam_incorrect_answers", [])
    if not isinstance(mistakes, list) or not mistakes:
        await update.callback_query.edit_message_text(
            "You made no mistakes in this mock exam."
        )
        context.user_data.clear()
        if update.callback_query.message:
            await update.callback_query.message.reply_text(
                "Main menu:", reply_markup=main_menu()
            )
        return

    context.user_data["review_index"] = 0
    await show_mistake_review(update, context)


async def finish_mistake_review(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    query = update.callback_query
    context.user_data.clear()
    await query.edit_message_text("Mistake review complete.")
    if query.message:
        await query.message.reply_text("Main menu:", reply_markup=main_menu())


def get_sat_math_score(raw_score, total_questions=44):
    percentage = raw_score / total_questions
    if percentage >= 0.95:
        return 800
    if percentage >= 0.85:
        return int(700 + (percentage - 0.85) * 1000)
    if percentage >= 0.70:
        return int(600 + (percentage - 0.70) * 666)
    if percentage >= 0.50:
        return int(500 + (percentage - 0.50) * 500)
    return max(200, int(200 + percentage * 600))


def get_ielts_reading_band(raw_score):
    if raw_score >= 39:
        return 9.0
    if raw_score >= 37:
        return 8.5
    if raw_score >= 35:
        return 8.0
    if raw_score >= 33:
        return 7.5
    if raw_score >= 30:
        return 7.0
    if raw_score >= 27:
        return 6.5
    if raw_score >= 23:
        return 6.0
    if raw_score >= 19:
        return 5.5
    if raw_score >= 15:
        return 5.0
    return 4.0


def evaluate_sat_exam(module_1_score, module_2_score):
    total_raw = module_1_score + module_2_score
    if module_1_score >= 15:
        scaled_score = get_sat_math_score(total_raw, total_questions=44)
        route_taken = "Hard Module 2 (Max Scale: 800)"
    else:
        scaled_score = min(620, get_sat_math_score(total_raw, total_questions=44))
        route_taken = "Standard Module 2 (Capped Range: 200–620)"
    return scaled_score, route_taken


async def send_exam_results(
    update: Update, context: ContextTypes.DEFAULT_TYPE, exam_type: str
):
    raw_m1 = context.user_data.get("m1_score", 0)
    raw_m2 = context.user_data.get("m2_score", 0)

    if exam_type == "sat":
        sat_score, route_taken = evaluate_sat_exam(raw_m1, raw_m2)
        report = (
            "📊 *DIGITAL SAT EXAM RESULT*\n"
            "-------------------------------------\n"
            f"• *Module 1 Score:* {raw_m1} / 22\n"
            f"• *Module 2 Route:* {route_taken}\n"
            f"• *Module 2 Score:* {raw_m2} / 22\n"
            f"• *Total Raw Correct:* {raw_m1 + raw_m2} / 44\n\n"
            f"🎯 *Official SAT Scale:* *{sat_score} / 800*\n"
        )
    else:
        band = get_ielts_reading_band(raw_m1)
        report = (
            "📊 *IELTS READING MOCK RESULT*\n"
            "-------------------------------------\n"
            f"• *Raw Score:* {raw_m1} / 40\n"
            f"🎯 *Official IELTS Band:* *{band} / 9.0*\n"
        )

    context.user_data["exam_completed"] = True
    review_markup = InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "📝 Review My Mistakes", callback_data="review_mistakes"
                )
            ]
        ]
    )
    if update.callback_query and update.callback_query.message:
        await update.callback_query.message.reply_text(
            report,
            parse_mode="Markdown",
            reply_markup=review_markup,
        )
    elif update.message:
        await update.message.reply_text(
            report,
            parse_mode="Markdown",
            reply_markup=review_markup,
        )


def build_application(token: str) -> Application:
    application = Application.builder().token(token).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("menu", show_menu))
    application.add_handler(
        MessageHandler(filters.Regex("^📝 Take SAT Mock Exam$"), start_sat_exam_handler)
    )
    application.add_handler(
        MessageHandler(
            filters.Regex("^📝 Take IELTS Mock Exam$"), start_ielts_exam_handler
        )
    )
    application.add_handler(CommandHandler("stats", stats))
    application.add_handler(
        CallbackQueryHandler(
            answer_question,
            pattern=r"^(answer|exam_answer):|^(exam_stop|review_mistakes|review_next|review_done)$",
        )
    )
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, menu_selection)
    )
    application.add_error_handler(error_handler)
    return application


def main() -> None:
    token = os.getenv("BOT_TOKEN")
    if not token:
        logger.critical("BOT_TOKEN environment variable is not set!")
        raise RuntimeError(
            "BOT_TOKEN environment variable is missing. "
            "Please configure BOT_TOKEN in Render Environment tab."
        )

    # Фоновый запуск Flask для проверки работоспособности (Health Check)
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    logger.info("Starting Telegram bot with %d questions loaded", TOTAL_QUESTIONS)
    bot_app = build_application(token)
    bot_app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
