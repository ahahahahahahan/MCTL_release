"""
Exp4: + MPRE (Multi-Perspective Reasoning Ensemble)

Built on Exp3 (CAMR + MCP + ATR): instead of using the last tool output,
an LLM integrator synthesizes all tool reasoning traces for a fused judgment.

Pipeline: Input -> CAMR -> MCP -> ATR -> MPRE evidence fusion -> Output

Design choices:
  - Tool evidence with confidence < CONFIDENCE_FILTER_THRESHOLD is excluded
    from the MPRE prompt to reduce noise.
  - A confidence-weighted majority vote is injected as an anchor signal.
  - Short-circuit threshold raised to 0.95 (requires >= 2 tools) to avoid
    premature termination.
  - When all high-confidence tools agree with a clear margin, MPRE is
    bypassed (consensus direct output).
"""
import asyncio
import aiohttp
import time
import json
import os
from typing import List, Dict, Any, Optional

from tqdm import tqdm

from config import (
    TEMPERATURE, MAX_TOKENS,
    BATCH_SIZE, MAX_CONCURRENCY, BATCH_DELAY, REQUEST_DELAY,
    DATASET_CONFIGS, API_TIMEOUT,
    MCP_PLANNING_PROMPT_EN,
    MPRE_PROMPT_EN,
)
from models.camr import CAMRRetriever
from models.mcp import parse_planning_output, get_dag_execution_order
from models.atr import (
    TOOL_DEFINITIONS, DIMENSION_TO_TOOL,
    TOOL_PROMPT_EN,
    parse_tool_output,
)
from utils import preprocess_data, fetch_api, extract_answer, normalize_prediction, calculate_metrics

CONFIDENCE_FILTER_THRESHOLD = 0.5
SHORT_CIRCUIT_THRESHOLD = 0.95


def format_tool_evidence_filtered(
    tool_results: Dict[str, Dict],
    min_confidence: float = CONFIDENCE_FILTER_THRESHOLD,
) -> tuple:
    """
    Serialize tool results as an MPRE evidence string, omitting
    low-confidence tools.

    Returns:
        (evidence_str, included_count, filtered_count)
    """
    lines = []
    filtered = 0
    for i, (tool_name, result) in enumerate(tool_results.items(), 1):
        if result["confidence_score"] < min_confidence:
            filtered += 1
            continue
        tool_def = TOOL_DEFINITIONS.get(tool_name, {})
        display_name = tool_def.get("name_en", tool_name)
        lines.append(
            f"[Tool {i}: {display_name}]\n"
            f"Analysis: {result['reasoning_trace'][:400]}\n"
            f"Confidence: {result['confidence_score']:.2f}\n"
            f"Judgment: {result['prediction']}"
        )
    return "\n\n".join(lines), len(lines), filtered


def compute_weighted_vote(
    tool_results: Dict[str, Dict],
    min_confidence: float = CONFIDENCE_FILTER_THRESHOLD,
) -> dict:
    """
    Confidence-weighted voting over tool predictions.

    Returns:
        {"fake_score", "real_score", "majority", "margin"}
    """
    fake_score = 0.0
    real_score = 0.0
    for result in tool_results.values():
        conf = result["confidence_score"]
        if conf < min_confidence:
            continue
        pred = normalize_prediction(result["prediction"])
        if pred == 1:
            fake_score += conf
        elif pred == 0:
            real_score += conf

    total = fake_score + real_score
    if total == 0:
        return {"fake_score": 0, "real_score": 0, "majority": "uncertain", "margin": 0}

    majority = "fake" if fake_score > real_score else "real"
    margin = abs(fake_score - real_score) / total
    return {
        "fake_score": round(fake_score, 3),
        "real_score": round(real_score, 3),
        "majority": majority,
        "margin": round(margin, 3),
    }


class MPREDetector:
    """Exp4: CAMR + MCP + ATR + MPRE detector."""

    def __init__(self, top_k: int = 5, clip_model: str = "ViT-B/32"):
        self.temperature = TEMPERATURE
        self.max_tokens = MAX_TOKENS
        self.timeout = API_TIMEOUT
        self.batch_size = BATCH_SIZE
        self.max_concurrency = MAX_CONCURRENCY
        self.request_delay = REQUEST_DELAY
        self.top_k = top_k
        self.clip_model = clip_model

    def _get_result_path(self, dataset_type: str) -> str:
        results_dir = DATASET_CONFIGS[dataset_type]["results_dir"]
        os.makedirs(results_dir, exist_ok=True)
        return os.path.join(results_dir, f"exp4_MPRE_k{self.top_k}.jsonl")

    def _load_existing_results(self, result_path: str) -> List[Dict]:
        if not os.path.exists(result_path):
            return []
        results = []
        with open(result_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        if "index" in item:
                            results.append(item)
                    except json.JSONDecodeError:
                        continue
        return results

    async def _call_llm(
        self,
        session: aiohttp.ClientSession,
        prompt: str,
        image_path: Optional[str] = None,
    ) -> str:
        return await fetch_api(
            session=session,
            prompt=prompt,
            image_path=image_path,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            request_delay=self.request_delay,
        )

    def _map_dimensions_to_tools(self, execution_order: List[str]) -> List[str]:
        tools = []
        seen = set()
        for dim in execution_order:
            tool_name = DIMENSION_TO_TOOL.get(dim, "semantic_dissector")
            if tool_name not in seen:
                tools.append(tool_name)
                seen.add(tool_name)
        return tools

    def _build_tool_prompt(
        self,
        tool_name: str,
        text: str,
        retrieved_context: str,
        prev_results: Dict[str, Dict],
        tool_dag: List[Dict],
    ) -> str:
        tool_def = TOOL_DEFINITIONS.get(tool_name, TOOL_DEFINITIONS["semantic_dissector"])
        t_name = tool_def["name_en"]
        t_desc = tool_def["desc_en"]

        dep_context = ""
        deps = []
        for node in tool_dag:
            if node.get("tool") == tool_name:
                deps = node.get("deps", [])
                break
            mapped = DIMENSION_TO_TOOL.get(node.get("tool", ""), "")
            if mapped == tool_name:
                deps = [DIMENSION_TO_TOOL.get(d, d) for d in node.get("deps", [])]
                break

        if deps and prev_results:
            dep_lines = []
            for dep in deps:
                if dep in prev_results:
                    r = prev_results[dep]
                    dep_lines.append(
                        f"[Prerequisite tool - {TOOL_DEFINITIONS.get(dep, {}).get('name_en', dep)}] "
                        f"analysis result:\n{r['reasoning_trace'][:300]}\n"
                        f"(Confidence: {r['confidence_score']:.2f}, Judgment: {r['prediction']})"
                    )
            if dep_lines:
                dep_context = (
                    "Analysis results from prerequisite tools (for your reference):\n"
                    + "\n\n".join(dep_lines)
                )

        if not dep_context:
            dep_context = "(No prerequisite tool dependencies)"

        return TOOL_PROMPT_EN.format(
            tool_name=t_name,
            tool_desc=t_desc,
            retrieved_context=retrieved_context,
            dependency_context=dep_context,
            text=text,
        )

    def _build_mpre_prompt(
        self,
        text: str,
        context_str: str,
        evidence_str: str,
        num_tools: int,
        weighted_vote: dict,
    ) -> str:
        vote_signal = (
            f"\nConfidence-weighted tool voting result:\n"
            f"  Fake weighted score: {weighted_vote['fake_score']:.2f}, "
            f"Real weighted score: {weighted_vote['real_score']:.2f}\n"
            f"  Majority direction: {weighted_vote['majority']}, "
            f"Margin: {weighted_vote['margin']:.1%}\n"
            f"  Note: This vote is for reference only. Make your independent judgment "
            f"based on evidence. However, if your judgment differs from the majority, "
            f"ensure you have strong justification."
        )
        base_prompt = MPRE_PROMPT_EN.format(
            text=text,
            retrieved_context=context_str,
            tool_evidence=evidence_str,
            num_tools=num_tools,
        )
        return base_prompt + "\n" + vote_signal

    async def run_dataset(self, dataset_type: str, retriever: CAMRRetriever) -> Dict[str, Any]:
        config = DATASET_CONFIGS[dataset_type]
        self.batch_size = config.get("batch_size", BATCH_SIZE)
        self.max_concurrency = config.get("max_concurrency", MAX_CONCURRENCY)

        print(f"\n{'='*60}")
        print(f"Exp4: +MPRE (K={self.top_k}) — Dataset: {dataset_type}")
        print(f"{'='*60}")

        start_time = time.time()

        retriever.build_index(config["train_path"], config["images_dir"])

        print("--- Loading test data ---")
        test_df = preprocess_data(config["test_path"])
        test_texts = test_df["text"].tolist()
        true_labels = test_df["label"].tolist()
        test_images = (
            test_df["image"].tolist() if "image" in test_df.columns
            else [None] * len(test_texts)
        )
        images_dir = config["images_dir"]

        print(f"Test samples: {len(test_texts)}")
        print(f"Label dist: real(0)={true_labels.count(0)}, fake(1)={true_labels.count(1)}")

        print(f"--- Pre-computing Top-{self.top_k} retrievals ---")
        all_retrieved = []
        for i in tqdm(range(len(test_texts)), desc="CAMR retrieve"):
            img_path = None
            if test_images[i]:
                p = os.path.join(images_dir, test_images[i])
                if os.path.exists(p):
                    img_path = p
            all_retrieved.append(retriever.retrieve(test_texts[i], img_path, self.top_k))

        result_path = self._get_result_path(dataset_type)
        existing_results = self._load_existing_results(result_path)
        processed_indices = {r["index"] for r in existing_results}
        if processed_indices:
            print(f"Resuming: {len(processed_indices)} samples already processed.")

        semaphore = asyncio.Semaphore(self.max_concurrency)
        all_results = list(existing_results)
        all_true, all_pred = [], []
        for r in existing_results:
            if r.get("predict") is not None and r.get("label") is not None:
                all_true.append(r["label"])
                all_pred.append(r["predict"])

        f_write = open(result_path, "a" if existing_results else "w", encoding="utf-8")
        planning_template = MCP_PLANNING_PROMPT_EN

        async def process_item(session, idx, text, image_filename, label, retrieved):
            if idx in processed_indices:
                return None

            context_str = retriever.format_context(retrieved)
            image_path = None
            if image_filename:
                img_path = os.path.join(images_dir, image_filename)
                if os.path.exists(img_path):
                    image_path = img_path

            api_calls = 0

            # ---- Stage 1: MCP planning ----
            planning_prompt = planning_template.format(
                text=text, retrieved_context=context_str
            )
            async with semaphore:
                planning_response = await self._call_llm(session, planning_prompt, image_path)
            api_calls += 1

            planning_result = parse_planning_output(planning_response)
            semantic_layers = planning_result.get("semantic_layers", [])
            tool_dag = planning_result.get("tool_dag", [])
            execution_order = get_dag_execution_order(tool_dag)
            if not execution_order and semantic_layers:
                execution_order = semantic_layers

            # ---- Stage 2: ATR tool execution (moderate short-circuit at 0.95) ----
            tool_sequence = self._map_dimensions_to_tools(execution_order)
            if not tool_sequence:
                tool_sequence = ["knowledge_grounding", "rhetorical_scanner", "evidence_comparator"]

            tool_results: Dict[str, Dict] = {}
            executed_tools = []
            short_circuited = False

            for tool_name in tool_sequence:
                tool_prompt = self._build_tool_prompt(
                    tool_name, text, context_str, tool_results, tool_dag
                )
                async with semaphore:
                    tool_response = await self._call_llm(session, tool_prompt, image_path)
                api_calls += 1

                parsed = parse_tool_output(tool_response)
                tool_results[tool_name] = parsed
                executed_tools.append({
                    "tool": tool_name,
                    "confidence": parsed["confidence_score"],
                    "prediction": parsed["prediction"],
                    "reasoning_snippet": parsed["reasoning_trace"][:200],
                })

                if (
                    len(executed_tools) >= 2
                    and parsed["confidence_score"] >= SHORT_CIRCUIT_THRESHOLD
                ):
                    short_circuited = True
                    break

            # ---- Stage 3: MPRE evidence fusion ----
            weighted_vote = compute_weighted_vote(tool_results)

            # Consensus bypass: skip MPRE when all high-confidence tools agree clearly.
            consensus_bypass = False
            high_conf_preds = []
            for t in executed_tools:
                if t["confidence"] >= CONFIDENCE_FILTER_THRESHOLD:
                    p = normalize_prediction(t["prediction"])
                    if p is not None:
                        high_conf_preds.append(p)

            if (
                len(high_conf_preds) >= 2
                and len(set(high_conf_preds)) == 1
                and weighted_vote["margin"] >= 0.5
            ):
                consensus_bypass = True
                predict = high_conf_preds[0]
                answer = "fake" if predict == 1 else "real"
                mpre_response = (
                    f"[Consensus bypass] All {len(high_conf_preds)} "
                    f"high-confidence tools agree: {answer}"
                )
            else:
                evidence_str, included, _ = format_tool_evidence_filtered(
                    tool_results, min_confidence=CONFIDENCE_FILTER_THRESHOLD
                )
                if included == 0:
                    evidence_str, included, _ = format_tool_evidence_filtered(
                        tool_results, min_confidence=0.0
                    )

                mpre_prompt = self._build_mpre_prompt(
                    text, context_str, evidence_str, included, weighted_vote
                )
                async with semaphore:
                    mpre_response = await self._call_llm(session, mpre_prompt, image_path)
                api_calls += 1

                answer = extract_answer(mpre_response)
                predict = normalize_prediction(answer)

            tool_predictions = [t["prediction"] for t in executed_tools if t["prediction"]]
            fake_votes = sum(1 for p in tool_predictions if normalize_prediction(p) == 1)
            real_votes = sum(1 for p in tool_predictions if normalize_prediction(p) == 0)

            label_dist = {"real": 0, "fake": 0}
            for r in retrieved:
                label_dist[r["label_str"]] += 1

            return {
                "index": idx,
                "text": text[:200],
                "label": label,
                "predict": predict,
                "answer_raw": answer,
                "mpre_response": mpre_response[:500],
                "semantic_layers": semantic_layers,
                "tool_sequence": [t["tool"] for t in executed_tools],
                "tool_results": executed_tools,
                "tool_votes": {"fake": fake_votes, "real": real_votes},
                "weighted_vote": weighted_vote,
                "consensus_bypass": consensus_bypass,
                "short_circuited": short_circuited,
                "api_calls": api_calls,
                "retrieved_labels": label_dist,
                "top1_similarity": retrieved[0]["similarity"] if retrieved else 0,
            }

        try:
            async with aiohttp.ClientSession() as session:
                items = [
                    (
                        i, test_texts[i],
                        test_images[i] if i < len(test_images) else None,
                        true_labels[i],
                        all_retrieved[i],
                    )
                    for i in range(len(test_texts))
                    if i not in processed_indices
                ]

                with tqdm(total=len(items), desc=f"Exp4[{dataset_type}]", unit="sample") as pbar:
                    for batch_start in range(0, len(items), self.batch_size):
                        batch = items[batch_start: batch_start + self.batch_size]
                        tasks = [
                            process_item(session, idx, text, img, label, retr)
                            for idx, text, img, label, retr in batch
                        ]
                        batch_results = await asyncio.gather(*tasks)

                        for result in batch_results:
                            if result is None:
                                continue
                            all_results.append(result)
                            json.dump(result, f_write, ensure_ascii=False)
                            f_write.write("\n")
                            f_write.flush()

                            if result["predict"] is not None and result["label"] is not None:
                                all_true.append(result["label"])
                                all_pred.append(result["predict"])

                        pbar.update(len(batch))

                        if all_true:
                            acc = sum(t == p for t, p in zip(all_true, all_pred)) / len(all_true)
                            pbar.set_postfix({"acc": f"{acc:.2%}", "valid": len(all_true)})

                        await asyncio.sleep(BATCH_DELAY)
        finally:
            if all_true:
                metrics = calculate_metrics(all_true, all_pred)
                valid_results = [r for r in all_results if "tool_sequence" in r]
                total_api = sum(r.get("api_calls", 0) for r in valid_results)
                avg_tools = (
                    sum(len(r["tool_sequence"]) for r in valid_results)
                    / max(len(valid_results), 1)
                )
                sc_count = sum(1 for r in valid_results if r.get("short_circuited", False))
                summary = {
                    "summary": True,
                    "dataset": dataset_type,
                    "experiment": "exp4_MPRE",
                    "top_k": self.top_k,
                    "total_samples": len(test_texts),
                    "valid_predictions": len(all_true),
                    "total_api_calls": total_api,
                    "avg_tools_per_sample": round(avg_tools, 2),
                    "short_circuit_count": sc_count,
                    "metrics": metrics,
                }
                json.dump(summary, f_write, ensure_ascii=False)
                f_write.write("\n")
            f_write.close()

        total_time = time.time() - start_time
        metrics = calculate_metrics(all_true, all_pred) if all_true else {}

        valid_results = [r for r in all_results if "tool_sequence" in r]
        total_api = sum(r.get("api_calls", 0) for r in valid_results)
        avg_tools = (
            sum(len(r["tool_sequence"]) for r in valid_results)
            / max(len(valid_results), 1)
        )
        sc_count = sum(1 for r in valid_results if r.get("short_circuited", False))
        bypass_count = sum(1 for r in valid_results if r.get("consensus_bypass", False))

        print(f"\n--- Exp4: +MPRE [{dataset_type}] (K={self.top_k}) ---")
        print(f"Total: {len(test_texts)}, Valid predictions: {len(all_true)}")
        print(f"Total API calls: {total_api}, Avg tools/sample: {avg_tools:.2f}")
        print(f"Short-circuit: {sc_count}/{len(valid_results)}, Consensus bypass: {bypass_count}/{len(valid_results)}")
        print(f"Accuracy:  {metrics.get('accuracy', 0):.4f}")
        print(f"Precision: {metrics.get('precision', 0):.4f}")
        print(f"Recall:    {metrics.get('recall', 0):.4f}")
        print(f"F1:        {metrics.get('f1', 0):.4f}")
        print(f"Macro-F1:  {metrics.get('macro_f1', 0):.4f}")
        print(f"Time: {total_time:.1f}s")
        print(f"Results saved to: {result_path}")

        return {
            "dataset": dataset_type,
            "experiment": "exp4_MPRE",
            "top_k": self.top_k,
            "total_samples": len(test_texts),
            "valid_predictions": len(all_true),
            "total_api_calls": total_api,
            "avg_tools_per_sample": round(avg_tools, 2),
            "short_circuit_count": sc_count,
            "consensus_bypass_count": bypass_count,
            "metrics": metrics,
            "time": round(total_time, 1),
            "result_path": result_path,
        }

    def run(self, dataset_type: str = None) -> Dict[str, Any]:
        """
        Run Exp4 on one or all datasets.

        Args:
            dataset_type: Dataset name, or None to run all datasets.
        """
        datasets = [dataset_type] if dataset_type else list(DATASET_CONFIGS.keys())

        retriever = CAMRRetriever(clip_model=self.clip_model, top_k=self.top_k)
        results = {}
        for ds in datasets:
            print(f"\n>>> Processing dataset: {ds}")
            results[ds] = asyncio.run(self.run_dataset(ds, retriever))

        if len(results) > 1:
            print(f"\n{'='*60}")
            print(f"Exp4: +MPRE (K={self.top_k}) Summary")
            print(f"{'='*60}")
            print(f"{'Dataset':<12} {'Accuracy':>10} {'Macro-F1':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
            print("-" * 64)
            for ds, r in results.items():
                m = r.get("metrics", {})
                print(
                    f"{ds:<12} {m.get('accuracy', 0):>10.4f} {m.get('macro_f1', 0):>10.4f} "
                    f"{m.get('precision', 0):>10.4f} {m.get('recall', 0):>10.4f} {m.get('f1', 0):>10.4f}"
                )

        return results
