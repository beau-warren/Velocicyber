<a name="sentinel-scope"></a>

# ![Sentinel scope](../assets/docs/headings/h1-sentinel-scope.svg)

From the repository root, create your scope file:

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
concurrent Pi coding and sentinel models.

The sentinel reviews executable tool actions asynchronously and can stop the
current task after detecting a possible violation. It cannot guarantee prevention
or undo actions that have already executed. See [the user guide](../docs/usage.md#data-and-sentinel)
for collection/upload controls and the limits of sentinel observations as training data.
