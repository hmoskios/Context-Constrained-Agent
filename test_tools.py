
from drift_agent.tools.toolkit import list_dir, rg_search, read_file, run_cmd

REPO = "/home/hmoskios/json"

print("Top-level:")
print(list_dir(REPO, "."))

print("\nSearch for json_pointer:")
hits = rg_search(REPO, "json_pointer", glob="*.hpp")
print("hits:", len(hits))
for h in hits[:5]:
    print(h)

if hits:
    h0 = hits[0]
    text, s, e = read_file(REPO, h0.path, h0.line - 5, h0.line + 25)
    print(f"\nSnippet from {h0.path}:{s}-{e}\n")
    print(text)

print("\nCMake configure (may take a moment):")
res = run_cmd(REPO, ["cmake", "-S", ".", "-B", "build"], cwd_rel=".")
print("returncode:", res.returncode)
print(res.stdout[-1000:])
print(res.stderr[-1000:])
