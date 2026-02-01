import os
import json
from mdutils.mdutils import MdUtils
from typing import List
from dataclasses import asdict
from .common import CookbookResult

def generate_combined_report(results: List[CookbookResult], output_dir: str) -> str:
    md = MdUtils(file_name=f"{output_dir}/combined", title="Cookbook Performance Report")
    
    md.new_header(level=1, title="Executive Summary")
    
    # Table Results
    headers = ["Recipe", "Model", "Batch", "Encrypted P50 (ms)", "Plaintext P50 (ms)", "Slowdown", "Correctness"]
    rows = []
    
    for r in results:
        slowdown = r.p50_latency_ms / r.plaintext_p50_ms if r.plaintext_p50_ms > 0 else 0
        rows.extend([
            r.recipe_name,
            r.model_type,
            str(r.batch_size),
            f"{r.p50_latency_ms:.2f}",
            f"{r.plaintext_p50_ms:.4f}",
            f"{slowdown:.1f}x",
            "✅ PASS" if r.correctness_passed else "❌ FAIL"
        ])
    
    md.new_table(columns=len(headers), rows=len(rows)//len(headers) + 1, text=headers + rows, text_align='center')
    
    md.new_header(level=1, title="Crypto Counters")
    
    c_headers = ["Model", "Rotations", "Switches"]
    c_rows = []
    for r in results:
        c_rows.extend([
            r.model_type,
            str(r.server_counters.get("rotations", 0)),
            str(r.server_counters.get("switches", 0))
        ])
        
    md.new_table(columns=len(c_headers), rows=len(c_rows)//len(c_headers) + 1, text=c_headers + c_rows, text_align='center')
    
    md.create_md_file()
    
    # Also save combined JSON
    with open(f"{output_dir}/combined.json", "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
        
    return f"{output_dir}/combined.md"
