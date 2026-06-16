"""Additional ViGOS evaluation benchmarks.

The implementations in this module are self-contained. Third-party benchmark
repositories are used only as references for prompt and metric semantics.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .eval_utils import (
    EvalSample,
    extract_images,
    image_value_to_metadata,
)


DEFAULT_EVAL_BENCHMARKS = ["vlind", "vilp-f", "vilp-p", "cv-bench"]

_BENCHMARK_ALIASES = {
    "vlind": "vlind",
    "vlind-bench": "vlind",
    "vlind_bench": "vlind",
    "vilp-f": "vilp-f",
    "vilp_f": "vilp-f",
    "vilpf": "vilp-f",
    "vilp-p": "vilp-p",
    "vilp_p": "vilp-p",
    "vilpp": "vilp-p",
    "cv-bench": "cv-bench",
    "cv_bench": "cv-bench",
    "cvbench": "cv-bench",
}

_BENCHMARK_RESPONSE_STEMS = {
    "vlind": "vlind_bench",
    "vilp-f": "vilp_f",
    "vilp-p": "vilp_p",
    "cv-bench": "cv_bench",
}


@dataclass(frozen=True)
class BenchmarkTask:
    name: str
    source: str
    response_stem: str
    total: int
    load_sample: Callable[[int], EvalSample]


def parse_benchmark_specs(raw: str | None) -> list[str]:
    if raw is None:
        parts = DEFAULT_EVAL_BENCHMARKS
    else:
        parts = [part.strip() for part in raw.split(",") if part.strip()]
    specs = []
    for part in parts:
        normalized = _BENCHMARK_ALIASES.get(part.lower())
        if normalized is None:
            raise ValueError(
                f"Unknown evaluation benchmark {part!r}. "
                f"Expected one of {sorted(set(_BENCHMARK_ALIASES.values()))}."
            )
        specs.append(normalized)
    return specs


def benchmark_response_stem(name: str) -> str:
    normalized = _BENCHMARK_ALIASES.get(name.lower(), name)
    try:
        return _BENCHMARK_RESPONSE_STEMS[normalized]
    except KeyError as exc:
        raise ValueError(f"Unknown evaluation benchmark: {name}") from exc


def load_benchmark_tasks(raw: str | None) -> list[BenchmarkTask]:
    tasks = []
    for name in parse_benchmark_specs(raw):
        if name == "vlind":
            tasks.append(load_vlind_task())
        elif name == "vilp-f":
            tasks.append(load_vilp_task("f"))
        elif name == "vilp-p":
            tasks.append(load_vilp_task("p"))
        elif name == "cv-bench":
            tasks.append(load_cv_bench_task())
        else:  # pragma: no cover - parse_benchmark_specs validates this.
            raise ValueError(f"Unhandled benchmark: {name}")
    return tasks


def score_benchmark(
    name: str,
    response_file: Path,
    judgment_file: Path,
    output_dir: Path,
) -> dict[str, Any]:
    normalized = _BENCHMARK_ALIASES.get(name.lower(), name)
    if normalized == "vlind":
        result = score_vlind(judgment_file)
    elif normalized in {"vilp-f", "vilp-p"}:
        result = score_vilp(normalized, judgment_file)
    elif normalized == "cv-bench":
        result = score_cv_bench(judgment_file)
    else:
        raise ValueError(f"Unknown evaluation benchmark: {name}")

    result.update(avg_at_k_fields(read_jsonl(judgment_file)))
    result["response_file"] = str(response_file)
    result["judgment_file"] = str(judgment_file)
    score_dir = output_dir / "benchmark_scores"
    score_dir.mkdir(parents=True, exist_ok=True)
    score_path = score_dir / f"{benchmark_response_stem(normalized)}.json"
    result["score_file"] = str(score_path)
    score_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def load_vilp_task(mode: str) -> BenchmarkTask:
    from datasets import load_dataset

    dataset = load_dataset("ViLP/ViLP", split="train")
    name = f"vilp-{mode}"
    total = len(dataset) * 3

    def load_sample(index: int) -> EvalSample:
        row_index = index // 3
        image_position = index % 3 + 1
        row = dataset[row_index]
        question = str(row["question"])
        if mode == "p":
            question = question.split(".", 1)[1].strip() if "." in question else question
        problem = f"Please answer with one word: {question}"
        image_key = f"image{image_position}"
        answer_key = f"answer{image_position}"
        image = row[image_key]
        ground_truth = str(row[answer_key])
        return EvalSample(
            dataset=f"benchmark/{name}",
            sample_id=f"{row_index}:{image_position}",
            problem=problem,
            ground_truth=ground_truth,
            images=extract_images({"image": image}),
            image_metadata=image_value_to_metadata(image),
            raw={
                "benchmark_meta": {
                    "benchmark": name,
                    "row_index": row_index,
                    "image_position": image_position,
                    "mode": mode,
                }
            },
        )

    return BenchmarkTask(
        name=name,
        source="ViLP/ViLP@train",
        response_stem=benchmark_response_stem(name),
        total=total,
        load_sample=load_sample,
    )


def load_cv_bench_task() -> BenchmarkTask:
    from datasets import load_dataset

    dataset = load_dataset("nyu-visionx/CV-Bench", "default", split="test")

    def load_sample(index: int) -> EvalSample:
        row = dataset[index]
        problem = str(row.get("prompt") or format_cv_prompt(row))
        answer = str(row.get("answer") or "")
        ground_truth = format_cv_ground_truth(answer, row.get("choices") or [])
        image = row["image"]
        sample_id = str(row.get("idx", index))
        return EvalSample(
            dataset="benchmark/cv-bench",
            sample_id=sample_id,
            problem=problem,
            ground_truth=ground_truth,
            images=extract_images({"image": image}),
            image_metadata=image_value_to_metadata(image),
            raw={
                "benchmark_meta": {
                    "benchmark": "cv-bench",
                    "idx": sample_id,
                    "type": row.get("type"),
                    "task": row.get("task"),
                    "answer": answer,
                    "choices": row.get("choices") or [],
                }
            },
        )

    return BenchmarkTask(
        name="cv-bench",
        source="nyu-visionx/CV-Bench@default/test",
        response_stem=benchmark_response_stem("cv-bench"),
        total=len(dataset),
        load_sample=load_sample,
    )


def load_vlind_task() -> BenchmarkTask:
    data_path = vlind_hf_file("VLind-Bench Dataset/data.json")
    contexts = json.loads(data_path.read_text(encoding="utf-8"))
    items: list[dict[str, Any]] = []
    for instance_index, instance in enumerate(contexts):
        good_image_ids = vlind_good_image_ids(instance)
        if not good_image_ids:
            continue
        items.extend(vlind_items_for_instance(instance_index, instance, good_image_ids))

    def load_sample(index: int) -> EvalSample:
        item = items[index]
        image_file = vlind_image_file(item["instance"], item["image_kind"], item["image_id"])
        image_path = vlind_hf_file(image_file)
        instance = item["instance"]
        meta = {
            "benchmark": "vlind",
            "instance_index": item["instance_index"],
            "concept": instance.get("concept"),
            "stage": item["stage"],
            "subtask": item["subtask"],
            "image_id": item["image_id"],
            "expected": item["ground_truth"],
            "good_image_ids": item["good_image_ids"],
        }
        return EvalSample(
            dataset="benchmark/vlind",
            sample_id=item["sample_id"],
            problem=item["problem"],
            ground_truth=item["ground_truth"],
            images=extract_images({"image": image_path}),
            image_metadata=[image_file],
            raw={"benchmark_meta": meta},
        )

    return BenchmarkTask(
        name="vlind",
        source="klee972/VLind-Bench",
        response_stem=benchmark_response_stem("vlind"),
        total=len(items),
        load_sample=load_sample,
    )


def vlind_hf_file(filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id="klee972/VLind-Bench",
            repo_type="dataset",
            filename=filename,
        )
    )


def format_cv_prompt(row: dict[str, Any]) -> str:
    question = str(row.get("question") or "").strip()
    choices = row.get("choices") or []
    if choices:
        labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        options = " ".join(f"({labels[index]}) {choice}" for index, choice in enumerate(choices))
        return f"{question} Select from the following choices. {options}"
    return question


def format_cv_ground_truth(answer: str, choices: list[Any]) -> str:
    label = normalize_choice_label(answer)
    if label:
        index = ord(label) - ord("A")
        if 0 <= index < len(choices):
            return f"({label}) {choices[index]}"
        return f"({label})"
    return str(answer)


def normalize_choice_label(value: Any) -> str:
    match = re.search(r"\b([A-Z])\b", str(value).upper())
    return match.group(1) if match else ""


def vlind_good_image_ids(instance: dict[str, Any], vote_threshold: int = 2) -> list[str]:
    labels = instance.get("aggregated_human_label_good_images") or {}
    return [
        str(image_id)
        for image_id, vote in labels.items()
        if int(vote) >= vote_threshold
    ]


def vlind_items_for_instance(
    instance_index: int,
    instance: dict[str, Any],
    good_image_ids: list[str],
) -> list[dict[str, Any]]:
    items = []
    false_statement = str(instance["false_statement"])
    true_statement = str(instance["true_statement"])
    context = str(instance["context"])
    existent = str(instance["existent_noun"])
    nonexistent = str(instance["non-existent_noun"])
    best_image_id = str(instance["best_img_id"])

    def add(subtask: str, stage: str, image_kind: str, image_id: str, problem: str, answer: str) -> None:
        items.append(
            {
                "instance": instance,
                "instance_index": instance_index,
                "sample_id": f"{instance_index}:{subtask}:{image_id}",
                "stage": stage,
                "subtask": subtask,
                "image_kind": image_kind,
                "image_id": image_id,
                "problem": problem,
                "ground_truth": answer,
                "good_image_ids": good_image_ids,
            }
        )

    add("a1", "a", "factual", "0", vlind_prompt(false_statement, "commonsense_TF"), "True")
    add("a2", "a", "factual", "0", vlind_prompt(true_statement, "commonsense_TF"), "False")
    add("b1", "b", "counterfactual", best_image_id, vlind_prompt(vlind_object_statement(existent), "simple_TF"), "True")
    add("b2", "b", "counterfactual", best_image_id, vlind_prompt(vlind_object_statement(nonexistent), "simple_TF"), "False")
    add("c1", "c", "counterfactual", best_image_id, vlind_context_prompt(context, true_statement, "detailed_TF"), "True")
    add("c2", "c", "counterfactual", best_image_id, vlind_context_prompt(context, false_statement, "detailed_TF"), "False")
    for image_id in good_image_ids:
        add("d1", "d", "counterfactual", image_id, vlind_prompt(true_statement, "detailed_TF"), "True")
        add("d2", "d", "counterfactual", image_id, vlind_prompt(false_statement, "detailed_TF"), "False")
    return items


def vlind_prompt(statement: str, prompt_type: str) -> str:
    if prompt_type == "detailed_TF":
        return (
            f"Statement: {statement}\n"
            "Based on the image, is the given statement true or false? "
            "Forget real-world common sense and just follow the information provided in the image. "
            "Only respond in True or False."
        )
    if prompt_type == "simple_TF":
        return (
            f"Statement: {statement}\n"
            "Based on the image, is the given statement true or false? Only respond in True or False."
        )
    if prompt_type == "commonsense_TF":
        return (
            f"Statement: {statement}\n"
            "Based on the common sense, is the given statement true or false? Only respond in True or False."
        )
    raise ValueError(f"Invalid VLind prompt type: {prompt_type}")


def vlind_context_prompt(context: str, statement: str, prompt_type: str) -> str:
    if prompt_type == "detailed_TF":
        return (
            f"Context: {context}\nStatement: {statement}\n"
            "Based on the context, is the given statement true or false? "
            "Forget real-world common sense and just follow the information provided in the context. "
            "Only respond in True or False."
        )
    if prompt_type == "simple_TF":
        return (
            f"Context: {context}\nStatement: {statement}\n"
            "Based on the context, is the given statement true or false? Only respond in True or False."
        )
    raise ValueError(f"Invalid VLind context prompt type: {prompt_type}")


def vlind_object_statement(obj: str) -> str:
    return f"There is {obj} in the given image."


def vlind_image_file(instance: dict[str, Any], image_kind: str, image_id: str) -> str:
    concept = str(instance["concept"])
    context_id = str(instance["context_id"])
    if image_kind == "factual":
        context = str(instance["factual_context"])
        group = "factual"
    elif image_kind == "counterfactual":
        context = str(instance["context"])
        group = "counterfactual"
    else:
        raise ValueError(f"Invalid VLind image kind: {image_kind}")
    return (
        "VLind-Bench Dataset/images/"
        f"{group}/{concept}/{context_id}_{context}/{image_id}.jpg"
    )


def score_vilp(name: str, judgment_file: Path) -> dict[str, Any]:
    judgments = read_jsonl(judgment_file)
    correct_by_position: dict[int, int] = defaultdict(int)
    total_by_position: dict[int, int] = defaultdict(int)
    attempt_correct_by_position: dict[int, int] = defaultdict(int)
    attempt_total_by_position: dict[int, int] = defaultdict(int)
    correct_by_row: dict[str, int] = defaultdict(int)
    total_by_row: dict[str, int] = defaultdict(int)

    for judgment in judgments:
        meta = judgment.get("benchmark_meta") or {}
        position = int(meta.get("image_position", 0))
        row_index = str(meta.get("row_index", ""))
        correct = is_correct_judgment(judgment)
        attempt_correct, attempt_total = attempt_stats(judgment)
        if position:
            total_by_position[position] += 1
            correct_by_position[position] += int(correct)
            attempt_correct_by_position[position] += attempt_correct
            attempt_total_by_position[position] += attempt_total
        if row_index:
            total_by_row[row_index] += 1
            correct_by_row[row_index] += int(correct)

    position_accuracy = [
        ratio(correct_by_position[position], total_by_position[position])
        for position in (1, 2, 3)
    ]
    position_avg_at_k = [
        optional_ratio(attempt_correct_by_position[position], attempt_total_by_position[position])
        for position in (1, 2, 3)
    ]
    score = (position_accuracy[1] + position_accuracy[2]) / 2
    prior = position_accuracy[0]
    counts = {f"{idx}_correct": 0 for idx in range(4)}
    for row_index, total in total_by_row.items():
        if total == 3:
            counts[f"{correct_by_row[row_index]}_correct"] += 1

    score_name = "ViLP-F" if name == "vilp-f" else "ViLP-P"
    return {
        "benchmark": name,
        "total": len(judgments),
        "correct": sum(1 for item in judgments if is_correct_judgment(item)),
        "accuracy": ratio(sum(1 for item in judgments if is_correct_judgment(item)), len(judgments)),
        "metrics": {
            f"{score_name} Score": score,
            f"{score_name} Prior": prior,
            "position_accuracy": position_accuracy,
            "position_avg_at_k": position_avg_at_k,
            "count_of_correct_by_question": counts,
        },
    }


def score_cv_bench(judgment_file: Path) -> dict[str, Any]:
    judgments = read_jsonl(judgment_file)
    groups: dict[str, dict[str, dict[str, int]]] = {"type": defaultdict(make_counter), "task": defaultdict(make_counter)}
    total = 0
    correct = 0
    for judgment in judgments:
        meta = judgment.get("benchmark_meta") or {}
        item_correct = int(is_correct_judgment(judgment))
        total += 1
        correct += item_correct
        for group_name in ("type", "task"):
            key = str(meta.get(group_name) or "unknown")
            groups[group_name][key]["total"] += 1
            groups[group_name][key]["correct"] += item_correct
            attempt_correct, attempt_total = attempt_stats(judgment)
            groups[group_name][key]["attempt_correct"] += attempt_correct
            groups[group_name][key]["attempt_total"] += attempt_total

    return {
        "benchmark": "cv-bench",
        "total": total,
        "correct": correct,
        "accuracy": ratio(correct, total),
        "metrics": {
            "overall_accuracy": ratio(correct, total),
            "type_accuracy": summarize_group(groups["type"]),
            "task_accuracy": summarize_group(groups["task"]),
        },
    }


def score_vlind(judgment_file: Path) -> dict[str, Any]:
    judgments = read_jsonl(judgment_file)
    grouped: dict[str, dict[str, Any]] = defaultdict(lambda: {"subtasks": {}, "d": defaultdict(dict)})
    for judgment in judgments:
        meta = judgment.get("benchmark_meta") or {}
        instance_key = str(meta.get("instance_index", ""))
        subtask = str(meta.get("subtask", ""))
        image_id = str(meta.get("image_id", ""))
        if not instance_key or not subtask:
            continue
        if meta.get("stage") == "d":
            grouped[instance_key]["d"][image_id][subtask] = is_correct_judgment(judgment)
        else:
            grouped[instance_key]["subtasks"][subtask] = is_correct_judgment(judgment)

    totals = defaultdict(float)
    total_instances = len(grouped)
    total_d_images = 0
    for data in grouped.values():
        subtasks = data["subtasks"]
        a_raw = int(bool(subtasks.get("a1")) and bool(subtasks.get("a2")))
        b_raw = int(bool(subtasks.get("b1")) and bool(subtasks.get("b2")))
        c_raw = int(bool(subtasks.get("c1")) and bool(subtasks.get("c2")))
        c_pass = int(c_raw and a_raw)
        bc_pass = int(b_raw and c_pass)

        d_values = []
        d_pass_values = []
        for image_subtasks in data["d"].values():
            d_raw = int(bool(image_subtasks.get("d1")) and bool(image_subtasks.get("d2")))
            d_values.append(d_raw)
            d_pass_values.append(int(d_raw and bc_pass))
        total_d_images += len(d_values)
        d_macro = sum(d_values) / len(d_values) if d_values else 0.0
        d_pass_macro = sum(d_pass_values) / len(d_pass_values) if d_pass_values else 0.0

        totals["a_pass"] += a_raw
        totals["b_pass"] += b_raw
        totals["c"] += c_raw
        totals["c_pass"] += c_pass
        totals["bc_pass"] += bc_pass
        totals["d_macro"] += d_macro
        totals["d_micro"] += sum(d_values)
        totals["d_pass_macro"] += d_pass_macro
        totals["d_pass_micro"] += sum(d_pass_values)
        totals["bc_pass_micro"] += bc_pass * len(d_values)

    metrics = {
        "a_pass_ratio": ratio(totals["a_pass"], total_instances),
        "b_pass_ratio": ratio(totals["b_pass"], total_instances),
        "c_pass_ratio": ratio(totals["c_pass"], totals["a_pass"]),
        "d_pass_ratio_macro": ratio(totals["d_pass_macro"], totals["bc_pass"]),
        "d_pass_ratio_micro": ratio(totals["d_pass_micro"], totals["bc_pass_micro"]),
        "c": ratio(totals["c"], total_instances),
        "d_macro": ratio(totals["d_macro"], total_instances),
        "d_micro": ratio(totals["d_micro"], total_d_images),
    }
    return {
        "benchmark": "vlind",
        "total": len(judgments),
        "correct": sum(1 for item in judgments if is_correct_judgment(item)),
        "accuracy": ratio(sum(1 for item in judgments if is_correct_judgment(item)), len(judgments)),
        "metrics": metrics,
    }


def is_correct_judgment(judgment: dict[str, Any]) -> bool:
    return str(judgment.get("judge_verdict") or "").strip().lower() == "correct"


def attempt_verdicts(judgment: dict[str, Any]) -> list[str]:
    raw = judgment.get("judge_attempt_verdicts")
    if not isinstance(raw, list):
        return []
    verdicts = []
    for value in raw:
        if isinstance(value, bool):
            verdicts.append("correct" if value else "incorrect")
            continue
        normalized = str(value).strip().lower()
        verdicts.append("correct" if normalized in {"correct", "true", "yes", "1"} else "incorrect")
    return verdicts


def attempt_stats(judgment: dict[str, Any]) -> tuple[int, int]:
    verdicts = attempt_verdicts(judgment)
    return sum(1 for verdict in verdicts if verdict == "correct"), len(verdicts)


def avg_at_k_fields(judgments: list[dict[str, Any]]) -> dict[str, Any]:
    attempt_totals = [attempt_stats(judgment) for judgment in judgments]
    attempt_correct = sum(correct for correct, _ in attempt_totals)
    attempt_total = sum(total for _, total in attempt_totals)
    return {
        "pass_at_k": ratio(
            sum(1 for item in judgments if is_correct_judgment(item)),
            len(judgments),
        ),
        "attempt_correct": attempt_correct,
        "attempt_total": attempt_total,
        "avg_at_k": optional_ratio(attempt_correct, attempt_total),
    }


def summarize_group(group: dict[str, dict[str, int]]) -> dict[str, dict[str, float | int]]:
    return {
        key: {
            "total": values["total"],
            "correct": values["correct"],
            "accuracy": ratio(values["correct"], values["total"]),
            "pass_at_k": ratio(values["correct"], values["total"]),
            "attempt_correct": values.get("attempt_correct", 0),
            "attempt_total": values.get("attempt_total", 0),
            "avg_at_k": optional_ratio(
                values.get("attempt_correct", 0),
                values.get("attempt_total", 0),
            ),
        }
        for key, values in sorted(group.items())
    }


def make_counter() -> dict[str, int]:
    return {"total": 0, "correct": 0, "attempt_correct": 0, "attempt_total": 0}


def ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def optional_ratio(numerator: float, denominator: float) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
