<a name="sentinel-scope"></a>

# ![Sentinel scope](../assets/docs/headings/h1-sentinel-scope.svg)

Use `/sentinel` → `/SCOPE — Edit scope restrictions` to create or edit
`SCOPE/SCOPE.MD` inside the harness. **Save / Ctrl+S** saves; **Cancel / Esc**
discards edits. Editing is available between tasks and model operations; saved
rules apply to the next task. Saving does not turn Sentinel ON.

Alternatively, from the repository root:

```bash
cp SCOPE/SCOPE.example.md SCOPE/SCOPE.MD
```

Replace the example with your actual written authorization, permitted targets,
allowed actions and restrictions before enabling `/sentinel`. Use a regular
UTF-8 text file with **1–6,000 bytes**. The current observer reads this single
`SCOPE/SCOPE.MD` file; finish or stop the current task before editing it.
Personal scope files are ignored by Git.

The menu offers Qwen3.5-35B-A3B IQK and Granite Guardian 4.1 8B Q4 as local
observers. Pair either with an API or Lambda coding model; the harness prevents
concurrent Pi coding and sentinel models. If a Pi coding model is loaded, turning
Sentinel ON or selecting its model offers a swap: confirm to unload Pi coding and
start the observer. The confirmation names the coding provider used afterward;
queued prompts pause and saved conversations are kept.

The sentinel reviews executable tool actions asynchronously and can stop the
current task after detecting a possible violation. It cannot guarantee prevention
or undo actions that have already executed. See [the user guide](../docs/usage.md#data-and-sentinel)
for collection/upload controls and the limits of sentinel observations as training data.
