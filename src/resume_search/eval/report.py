"""Plain-text tables for `eval run` and `eval show`."""

from __future__ import annotations

from resume_search.eval.ranking_eval import VARIANTS


def _table(header: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(str(r[i])) for r in [header, *rows]) for i in range(len(header))]
    line = lambda r: "  ".join(str(c).ljust(w) for c, w in zip(r, widths, strict=True)).rstrip()
    return "\n".join([line(header), line(["-" * w for w in widths]), *(line(r) for r in rows)])


def _pct(x: float) -> str:
    return f"{x:.0%}"


def format_parsing(body: dict) -> str:
    m = body["metrics"]
    rows = [
        [k, int(v["total"]), int(v["correct"]), int(v["partial"]), int(v["wrong"]),
         int(v["hallucinated"]), int(v["missing"]), _pct(v["accuracy"]), _pct(v["soft_accuracy"])]
        for k, v in sorted(m["per_field"].items(), key=lambda kv: kv[0] == "overall")
    ]
    out = [
        (
            f"Parsing eval, judge {body['judge_model']}, prompt {body['prompt_version']}, "
            f"{m['candidates']} candidates, cost ${body['cost_usd']:.4f}"
        ),
        _table(["field", "n", "correct", "partial", "wrong", "halluc", "missing", "acc", "soft"], rows),
    ]
    ag = m["agreement"]
    if ag["overall"]:
        o = ag["overall"]
        out.append(
            f"\nJudge vs human ({'/'.join(ag['gold_status'])} gold), {o['n']} fields: "
            f"exact {_pct(o['exact'])}, within one level {_pct(o['within_one_level'])}"
        )
        out.append(_table(
            ["candidate", "n", "exact", "within_one"],
            [[n, a["n"], _pct(a["exact"]), _pct(a["within_one_level"])]
             for n, a in ag["per_candidate"].items()],
        ))
        for n, a in ag["per_candidate"].items():
            for d in a["disagreements"]:
                out.append(f"  {n} {d['field']}: human {d['human']}, judge {d['judge']}")
    return "\n".join(out)


def format_ranking(body: dict) -> str:
    m = body["metrics"]
    cols = [(v, v) for v in VARIANTS]
    header = ["requisition", "pool"] + [f"{v} {k}" for v, _ in cols for k in ("ndcg5", "p3")]
    rows = [
        [i["requisition_id"], i["pool_size"]]
        + [f"{i[f'{v}_{k}']:.3f}" for v, _ in cols for k in ("ndcg5", "p3")]
        for i in body["items"]
    ]
    rows.append(
        ["MEAN", ""] + [f"{m['mean'][v][k]:.3f}" for v, _ in cols for k in ("ndcg5", "p3")]
    )
    return (
        f"Ranking eval, judge {body['judge_model']}, prompt {body['prompt_version']}, "
        f"judge calls this run {m['judge_calls_this_run']}, cost ${body['cost_usd']:.4f}\n"
        + _table(header, rows)
    )


FORMATTERS = {"parsing": format_parsing, "ranking": format_ranking}
