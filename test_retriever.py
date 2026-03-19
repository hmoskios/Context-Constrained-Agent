
from drift_agent.retrieval.retriever import retrieve_for_query, retrieve_for_build_output
from drift_agent.tools.toolkit import run_cmd


REPO = "/home/hmoskios/json"


print("=== Understanding query ===")
snips = retrieve_for_query(REPO, "What does json_pointer do and where is it defined?", max_snippets=3)
for s in snips:
    print(f"\n[{s.reason}] {s.path}:{s.start_line}-{s.end_line}\n{s.text[:800]}")
print("\n=== Build output-driven query ===")
# Use configure output as a quick demo; later you'll feed real error output here
res = run_cmd(REPO, ["cmake", "-S", ".", "-B", "build"], cwd_rel=".")
snips2 = retrieve_for_build_output(REPO, res.stdout + "\n" + res.stderr, max_snippets=3)
print("error-locs snippets:", len(snips2))


fake_error = "/home/hmoskios/json/include/nlohmann/detail/json_pointer.hpp:35: error: fake compiler error"
snips = retrieve_for_build_output("/home/hmoskios/json", fake_error, max_snippets=3)
print("\n=== Fake build-error retrieval ===")
for s in snips:
    print(f"\n[{s.reason}] {s.path}:{s.start_line}-{s.end_line}\n{s.text[:800]}")
