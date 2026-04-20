import sys
import re

def clean_benchmark_output(text):
    lines = text.splitlines()
    cleaned = []
    current_bench = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Extract benchmark name (works for both "Benchmarking ..." and raw paths)
        name_match = re.search(r'([A-Za-z0-9_]+_Performances/[^\s:]+)', stripped)
        if name_match:
            bench_name = name_match.group(1)
            if bench_name != current_bench:
                if cleaned:
                    cleaned.append("")
                cleaned.append(f"Benchmarking {bench_name}")
                current_bench = bench_name

        # Extract time line
        if 'time:' in stripped:
            time_match = re.search(r'time:\s*\[.*?\]', stripped)
            if time_match:
                cleaned.append(f"  {time_match.group(0)}")

        # Extract thrpt line
        if 'thrpt:' in stripped:
            thrpt_match = re.search(r'thrpt:\s*\[.*?\]', stripped)
            if thrpt_match:
                cleaned.append(f"  {thrpt_match.group(0)}")

    return "\n".join(cleaned)


if __name__ == "__main__":
    print("🚀 Benchmark Output Cleaner")
    print("Paste your messy benchmark text below (Ctrl+D / Ctrl+Z+Enter to finish):")
    print("-" * 70)

    messy_text = sys.stdin.read()

    if not messy_text.strip():
        print("No input received.")
        sys.exit(1)

    result = clean_benchmark_output(messy_text)

    print("\n" + "=" * 70)
    print("✅ CLEANED OUTPUT (ready to copy)")
    print("=" * 70)
    print(result)
    print("=" * 70)
