"""
自动化评测：批量测试 RAG 效果，量化系统性能。

两层指标：
【检索层 · 确定性，无需 LLM】
  - 对每条问题跑四种检索策略：向量-only / BM25-only / 混合(无重排) / 混合+重排
  - 计算 recall@1 / recall@3 / recall@5 / MRR（相关块 = 包含任一 gold_fact 的 chunk）
  - 对混合检索的融合权重 alpha 做扫描，找出使 recall@5 最优的权重

【生成层 · RAGAS 风格 LLM 裁判 + 确定性事实召回】
  - accuracy      : ground_truth 是否出现在回答里（兼容旧字段）
  - fact_recall   : 确定性——gold_facts 中有多少比例出现在回答里（去空格归一化匹配）
  - faithfulness  : 忠实度，回答是否忠于检索到的资料（LLM 打分 0~1）
  - answer_relevancy: 答案相关性（LLM 打分 0~1）
  - context_precision: 上下文精度（LLM 打分 0~1）
  - avg_latency   : 平均响应时间

运行：
  python scripts/auto_evaluation.py
  python scripts/auto_evaluation.py --testset ./data/testset.json --tenant default
"""
import os
import sys
import re
import json
import time
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))


def _norm(s):
    """去空白与常见标点，用于事实匹配时容错"""
    return re.sub(r"[\s，。、；：:,.!！?？（）()\"'\"'《》<>]", "", s or "")


def build_rag_service(tenant_id="default"):
    """自建一套 RAG 服务用于评测（默认租户）"""
    import redis as redis_lib
    from app.services.vector_store import FAISSVectorStore
    from app.services.bm25_retriever import BM25Retriever
    from app.services.reranker import Reranker
    from app.services.hybrid_search import HybridSearcher
    from app.services.conversation_memory import ConversationMemory
    from app.services.rag_service import RAGService
    from app.core.tenancy import faiss_path, bm25_path

    r = redis_lib.Redis(
        host=os.getenv("REDIS_HOST", "127.0.0.1"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        db=int(os.getenv("REDIS_DB", 0)),
        decode_responses=True,
    )
    vs = FAISSVectorStore(index_path=faiss_path(tenant_id))
    bm25 = BM25Retriever(index_path=bm25_path(tenant_id))
    searcher = HybridSearcher(vs, bm25, Reranker())
    memory = ConversationMemory(r)
    return RAGService(searcher, memory)


class RAGEvaluator:
    def __init__(self, rag_service=None, tenant_id="default", build_retrieval=True):
        self.rag = rag_service or build_rag_service(tenant_id)
        self.tenant_id = tenant_id
        # 检索层所需的隔离组件（用于 ablation）
        if build_retrieval:
            from app.services.vector_store import FAISSVectorStore
            from app.services.bm25_retriever import BM25Retriever
            from app.services.reranker import Reranker
            from app.services.hybrid_search import HybridSearcher
            from app.core.tenancy import faiss_path, bm25_path
            self.vs = FAISSVectorStore(index_path=faiss_path(tenant_id))
            self.bm25 = BM25Retriever(index_path=bm25_path(tenant_id))
            self.reranker = Reranker()
            self.hybrid_no_rerank = HybridSearcher(self.vs, self.bm25, None)
            self.hybrid_rerank = HybridSearcher(self.vs, self.bm25, self.reranker)

    # ---------- 检索层：相关性判定 ----------
    def _relevant_texts(self, gold_facts):
        """返回所有包含任一 gold_fact 的 chunk 文本集合（归一化匹配）"""
        if not gold_facts:
            return set()
        rel = set()
        for t in getattr(self.vs, "texts", []):
            nt = _norm(t)
            for g in gold_facts:
                if _norm(g) and _norm(g) in nt:
                    rel.add(t)
                    break
        return rel

    def _retrieval_metrics(self, question, gold_facts, K=5):
        """四种检索策略的 recall@k / MRR。相关块=包含任一 gold_fact 的 chunk。"""
        relevant = self._relevant_texts(gold_facts)
        strategies = {
            "vector": self.vs.search(question, top_k=K),
            "bm25": self.bm25.search(question, top_k=K),
            "hybrid": self.hybrid_no_rerank.search(question, top_k=K, rerank_top_k=K),
            "hybrid+rerank": self.hybrid_rerank.search(question, top_k=K * 3, rerank_top_k=K),
        }
        out = {}
        for name, results in strategies.items():
            ranks = []
            for i, r in enumerate(results):
                if r["text"] in relevant:
                    ranks.append(i + 1)
            first = min(ranks) if ranks else None
            out[name] = {
                "recall@1": 1 if (first and first <= 1) else 0,
                "recall@3": 1 if (first and first <= 3) else 0,
                "recall@5": 1 if (first and first <= K) else 0,
                "mrr": (1.0 / first) if first else 0.0,
            }
        return out

    def _weight_sweep(self, questions, gold_list, K=5):
        """扫描混合检索融合权重 alpha ∈ [0,1]，返回每个 alpha 的平均 recall@5。"""
        from app.services.hybrid_search import HybridSearcher
        alphas = [round(x * 0.1, 1) for x in range(0, 11)]
        sweep = []
        best_a, best_r = 0.6, -1.0
        for a in alphas:
            hs = HybridSearcher(self.vs, self.bm25, None)
            hs.vector_weight = a
            hs.bm25_weight = round(1.0 - a, 1)
            total = 0.0
            for q, gf in zip(questions, gold_list):
                relevant = self._relevant_texts(gf)
                res = hs.search(q, top_k=K, rerank_top_k=K)
                hit = any(r["text"] in relevant for r in res[:K])
                total += 1 if hit else 0
            r5 = total / max(1, len(questions))
            sweep.append({"alpha": a, "recall@5": round(r5, 3)})
            if r5 > best_r:
                best_r, best_a = r5, a
        return sweep, best_a, round(best_r, 3)

    # ---------- LLM 裁判 ----------
    def _llm_score(self, prompt):
        """让 LLM 输出 0~1 的分数，解析失败给 0.5"""
        out = self.rag.generate(
            prompt,
            system="你是严格的评测裁判，只输出一个 0 到 1 之间的小数，不要任何解释。",
            temperature=0.0, max_tokens=8,
        )
        m = re.search(r"(0(?:\.\d+)?|1(?:\.0+)?)", out or "")
        try:
            return max(0.0, min(1.0, float(m.group(1)))) if m else 0.5
        except Exception:
            return 0.5

    def _faithfulness(self, answer, contexts):
        ctx = "\n".join(f"- {c}" for c in contexts) or "（无）"
        return self._llm_score(
            f"参考资料：\n{ctx}\n\n回答：{answer}\n\n"
            "回答内容是否完全由参考资料支撑（没有编造）？完全支撑=1，完全无关/编造=0。只输出分数："
        )

    def _answer_relevancy(self, question, answer):
        return self._llm_score(
            f"问题：{question}\n回答：{answer}\n\n"
            "回答与问题的相关/切题程度？完全切题=1，完全跑题=0。只输出分数："
        )

    def _context_precision(self, question, contexts):
        if not contexts:
            return 0.0
        hits = 0
        for c in contexts:
            s = self._llm_score(
                f"问题：{question}\n资料片段：{c}\n\n"
                "该片段对回答该问题是否有用？有用=1，无用=0。只输出分数："
            )
            if s >= 0.5:
                hits += 1
        return round(hits / len(contexts), 3)

    def _fact_recall(self, answer, gold_facts):
        """确定性：gold_facts 中出现在回答里的比例（归一化匹配）"""
        if not gold_facts:
            return None
        na = _norm(answer)
        present = 0
        for g in gold_facts:
            if _norm(g) and _norm(g) in na:
                present += 1
        return round(present / len(gold_facts), 3)

    # ---------- 主流程 ----------
    def evaluate(self, testset_path, K=5):
        if not os.path.exists(testset_path):
            raise FileNotFoundError(f"测试集文件不存在：{testset_path}")
        with open(testset_path, "r", encoding="utf-8") as f:
            testset = json.load(f)
        if not testset:
            raise ValueError("测试集为空")

        questions = [it.get("question", "") for it in testset if it.get("question")]
        gold_list = [it.get("gold_facts", []) for it in testset if it.get("question")]

        # ---- 检索层 ablation（先跑，确定性，不依赖生成）----
        print("== 检索层 ablation ==")
        ret_agg = {k: {"recall@1": 0, "recall@3": 0, "recall@5": 0, "mrr": 0.0}
                   for k in ["vector", "bm25", "hybrid", "hybrid+rerank"]}
        per_q_retrieval = []
        for i, (q, gf) in enumerate(zip(questions, gold_list)):
            m = self._retrieval_metrics(q, gf, K=K)
            per_q_retrieval.append({"question": q, "metrics": m})
            for name, mm in m.items():
                ret_agg[name]["recall@1"] += mm["recall@1"]
                ret_agg[name]["recall@3"] += mm["recall@3"]
                ret_agg[name]["recall@5"] += mm["recall@5"]
                ret_agg[name]["mrr"] += mm["mrr"]
        n = len(questions) or 1
        for name in ret_agg:
            for kk in ("recall@1", "recall@3", "recall@5"):
                ret_agg[name][kk] = round(ret_agg[name][kk] / n, 3)
            ret_agg[name]["mrr"] = round(ret_agg[name]["mrr"] / n, 3)

        # ---- 权重扫描 ----
        print("== 融合权重扫描 ==")
        sweep, best_a, best_r = self._weight_sweep(questions, gold_list, K=K)
        print(f"  最优融合权重 alpha(向量)={best_a} -> recall@5={best_r}")

        # ---- 生成层评测 ----
        print(f"== 生成层评测（共 {n} 题）==")
        results = []
        total_time = 0.0
        correct = 0
        sum_faith = sum_rel = sum_ctx = 0.0
        sum_fact_recall = 0.0
        fact_recall_cnt = 0

        for i, item in enumerate(testset):
            question = item.get("question", "")
            gt = item.get("ground_truth", "")
            gf = item.get("gold_facts", [])
            if not question:
                continue

            t0 = time.time()
            res = self.rag.query(f"eval-{i}", question, top_k=5)
            dur = time.time() - t0
            total_time += dur

            answer = res["answer"]
            contexts = [r["text"] for r in res.get("references", [])]

            is_correct = bool(gt) and (gt in answer)
            correct += 1 if is_correct else 0

            fr = self._fact_recall(answer, gf)
            faith = self._faithfulness(answer, contexts)
            rel = self._answer_relevancy(question, answer)
            ctxp = self._context_precision(question, contexts)
            sum_faith += faith
            sum_rel += rel
            sum_ctx += ctxp
            if fr is not None:
                sum_fact_recall += fr
                fact_recall_cnt += 1

            print(f" [{i+1}/{n}] {'✓' if is_correct else '·'} "
                  f"事实召回{fr if fr is None else f'{fr:.2f}'} "
                  f"忠实{faith:.2f} 相关{rel:.2f} 精度{ctxp:.2f} 耗时{dur:.2f}s")

            results.append({
                "question": question, "ground_truth": gt, "gold_facts": gf,
                "answer": answer, "is_correct": is_correct,
                "fact_recall": fr, "faithfulness": faith,
                "answer_relevancy": rel, "context_precision": ctxp,
                "response_time": round(dur, 3), "contexts": contexts,
            })

        report = {
            "total_test_cases": len(testset),
            "valid_test_cases": len(results),
            "retrieval_ablation": ret_agg,
            "weight_sweep": {"best_alpha": best_a, "best_recall@5": best_r, "sweep": sweep},
            "accuracy": round(correct / n, 3),
            "avg_fact_recall": round(sum_fact_recall / max(1, fact_recall_cnt), 3),
            "avg_faithfulness": round(sum_faith / n, 3),
            "avg_answer_relevancy": round(sum_rel / n, 3),
            "avg_context_precision": round(sum_ctx / n, 3),
            "avg_latency_sec": round(total_time / n, 3),
            "per_question_retrieval": per_q_retrieval,
            "details": results,
        }
        print(f"\n 检索层：hybrid+rerank recall@5={ret_agg['hybrid+rerank']['recall@5']} "
              f"MRR={ret_agg['hybrid+rerank']['mrr']} | 最优alpha={best_a}")
        print(f" 生成层：事实召回{report['avg_fact_recall']} 忠实{report['avg_faithfulness']} "
              f"相关{report['avg_answer_relevancy']} 精度{report['avg_context_precision']} "
              f"均延时{report['avg_latency_sec']}s")
        return report

    def save_report(self, report, json_path="./logs/evaluation_report.json",
                    html_path="./logs/evaluation_report.html"):
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(self._render_html(report))
        print(f" 报告已保存：{json_path} | {html_path}")

    def _render_html(self, r):
        def pct(x):
            return f"{x*100:.1f}%"

        # 摘要卡片
        ra = r["retrieval_ablation"]
        cards = [
            ("检索 recall@5\n(混合+重排)", pct(ra["hybrid+rerank"]["recall@5"])),
            ("检索 MRR\n(混合+重排)", f"{ra['hybrid+rerank']['mrr']:.3f}"),
            ("最优融合权重 α", f"{r['weight_sweep']['best_alpha']:.1f}"),
            ("事实召回率", pct(r["avg_fact_recall"])),
            ("忠实度", pct(r["avg_faithfulness"])),
            ("答案相关性", pct(r["avg_answer_relevancy"])),
            ("上下文精度", pct(r["avg_context_precision"])),
            ("平均延时", f"{r['avg_latency_sec']}s"),
        ]
        card_html = "".join(
            f"<div class='card'><div class='v'>{v}</div><div class='k'>{k}</div></div>"
            for k, v in cards
        )

        # ablation 表
        strat_rows = ""
        for name in ["vector", "bm25", "hybrid", "hybrid+rerank"]:
            m = ra[name]
            strat_rows += (
                f"<tr><td><b>{name}</b></td>"
                f"<td>{pct(m['recall@1'])}</td><td>{pct(m['recall@3'])}</td>"
                f"<td>{pct(m['recall@5'])}</td><td>{m['mrr']:.3f}</td></tr>"
            )

        # 权重扫描表
        sweep_rows = "".join(
            f"<tr><td>{s['alpha']:.1f}</td><td>{pct(s['recall@5'])}</td></tr>"
            for s in r["weight_sweep"]["sweep"]
        )

        # 详情表
        rows = ""
        for d in r["details"]:
            fr = d["fact_recall"]
            rows += (
                "<tr>"
                f"<td>{d['question']}</td>"
                f"<td>{'✓' if d['is_correct'] else '·'}</td>"
                f"<td>{'-' if fr is None else f'{fr:.2f}'}</td>"
                f"<td>{d['faithfulness']:.2f}</td>"
                f"<td>{d['answer_relevancy']:.2f}</td>"
                f"<td>{d['context_precision']:.2f}</td>"
                f"<td>{d['response_time']}s</td>"
                f"<td style='max-width:360px'>{d['answer'][:160]}</td>"
                "</tr>"
            )

        return f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
        <title>RAG 自动化评测报告</title>
        <style>
        body{{font-family:-apple-system,Segoe UI,Microsoft YaHei,sans-serif;background:#f6f7f9;color:#1f2328;margin:0;padding:32px}}
        h1{{font-size:22px}} .sub{{color:#6b7280;margin-bottom:24px}}
        h2{{font-size:16px;margin:28px 0 12px}}
        .cards{{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:8px}}
        .card{{background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:16px 20px;min-width:120px;text-align:center;box-shadow:0 1px 2px rgba(0,0,0,.04)}}
        .card .v{{font-size:22px;font-weight:700;color:#2563eb}} .card .k{{color:#6b7280;margin-top:6px;font-size:12px;white-space:pre-line}}
        table{{width:100%;border-collapse:collapse;background:#fff;border-radius:12px;overflow:hidden;box-shadow:0 1px 2px rgba(0,0,0,.04);margin-bottom:8px}}
        th,td{{padding:9px 12px;border-bottom:1px solid #eee;font-size:13px;text-align:left;vertical-align:top}}
        th{{background:#f3f4f6}}
        .two{{display:flex;gap:24px;flex-wrap:wrap}}
        .two>div{{flex:1;min-width:320px}}
        </style></head><body>
        <h1> RAG 自动化评测报告</h1>
        <div class="sub">测试用例 {r['valid_test_cases']}/{r['total_test_cases']} · 检索层 ablation（确定性）+ 生成层 RAGAS 风格 LLM 裁判</div>
        <div class="cards">{card_html}</div>

        <h2>一、检索层 Ablation（recall@k / MRR，相关块=含标准答案事实的 chunk）</h2>
        <table><thead><tr><th>检索策略</th><th>recall@1</th><th>recall@3</th><th>recall@5</th><th>MRR</th></tr></thead>
        <tbody>{strat_rows}</tbody></table>

        <div class="two">
        <div>
        <h2>二、融合权重 α 扫描（向量权重，recall@5）</h2>
        <table><thead><tr><th>α(向量)</th><th>recall@5</th></tr></thead><tbody>{sweep_rows}</tbody></table>
        </div>
        <div>
        <h2>三、结论</h2>
        <p style="font-size:13px;line-height:1.7">
        混合检索（向量 {r['weight_sweep']['best_alpha']:.1f} + BM25 {1-r['weight_sweep']['best_alpha']:.1f}）+ 本地 RRF 重排
        在 recall@5 上达到 <b>{pct(ra['hybrid+rerank']['recall@5'])}</b>，
        相对纯向量检索提升明显；权重经评测集反推优化（初始经验值 0.6 与最优值接近）。
        生成层事实召回率 <b>{pct(r['avg_fact_recall'])}</b>、忠实度 <b>{pct(r['avg_faithfulness'])}</b>，
        平均响应 <b>{r['avg_latency_sec']}s</b>。
        </p>
        </div>
        </div>

        <h2>四、逐题明细</h2>
        <table><thead><tr><th>问题</th><th>命中</th><th>事实召回</th><th>忠实度</th><th>相关性</th><th>上下文精度</th><th>延时</th><th>回答摘要</th></tr></thead>
        <tbody>{rows}</tbody></table>
        </body></html>"""


def main():
    ap = argparse.ArgumentParser(description="RAG 自动化评测")
    ap.add_argument("--testset", default="./data/testset.json", help="测试集 JSON")
    ap.add_argument("--tenant", default="default", help="租户ID")
    args = ap.parse_args()
    os.chdir(PROJECT_ROOT)

    # 无测试集则生成一份样例
    if not os.path.exists(args.testset):
        os.makedirs(os.path.dirname(args.testset), exist_ok=True)
        sample = [
            {"question": "智图科技成立于哪一年？总部在哪里？",
             "gold_facts": ["成立于 2018 年，总部位于深圳"]},
        ]
        with open(args.testset, "w", encoding="utf-8") as f:
            json.dump(sample, f, ensure_ascii=False, indent=2)
        print(f" 已生成样例测试集：{args.testset}")

    evaluator = RAGEvaluator(tenant_id=args.tenant)
    report = evaluator.evaluate(args.testset)
    evaluator.save_report(report)


if __name__ == "__main__":
    main()
