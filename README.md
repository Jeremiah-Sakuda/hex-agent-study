# Hex agent study copies

Both files are annotated copies of the PA2 agents. The comments explain the code without changing how it runs. Each file declares the same Java class, so compile them separately.

| File | What to study |
| --- | --- |
| [Regular UCT](regular/UCTAgent.java) | Selection, expansion, random rollouts, and backup. The agent returns the root move with the highest sampled average reward. |
| [Extra credit: UCT with RAVE](extra-credit/UCTAgent.java) | The same search loop with bridge replies, immediate tactics, and AMAF statistics. The normal search result is the most visited root move. |

Start with `treeSearch`, then follow one iteration through `selectArtificialLeaf`, `chooseChild`, `getRolloutOutcome`, and `backupStatistics`. Read `bestUCB` to see how the resulting statistics guide later iterations.

## Trace one simulation

Suppose X is our agent. X plays A from the root, then the rollout starts with O playing B. X plays C later and wins. Rewards are stored from X's perspective throughout this example.

| Position | Ordinary Q update in both versions | Extra AMAF update with RAVE |
| --- | --- | --- |
| Root, X to move | A gets one sample of +100 | A and C each get one sample of +100 |
| Child after A, O to move | B gets one sample of +100 as the first rollout action | B gets one sample of +100 |

C still has zero direct trials at the root. RAVE records C's result separately and can use it after C has a sampled child. A later placement is only indirect evidence that playing the same cell earlier would help. As direct trials accumulate, the blend gives them more weight. At opponent nodes, the search negates the blended reward term so the opponent prefers moves that are worse for X.

## Compile against the course library

Run from this repository with Java 21 and the JARs from your PA2 setup. Set `HEX_LIB_DIR` to your course library directory:

```sh
HEX_LIB_DIR="$HOME/cs440/lib"
mkdir -p build/regular build/extra-credit
javac -cp "$HEX_LIB_DIR/*" -d build/regular regular/UCTAgent.java
javac -cp "$HEX_LIB_DIR/*" -d build/extra-credit extra-credit/UCTAgent.java
```

To open the regular agent in the course GUI, run `java -cp "build/regular:$HEX_LIB_DIR/*" edu.bu.pas.hex.DebugMain` and load `src.pas.hex.agents.UCTAgent`. Use `build/extra-credit` in the classpath to try the other version.
