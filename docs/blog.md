# Deep-Dive: Hybrid RAG & Referential Guard Mechanics

## The Challenge in Local SLM Agent Deployments

Small Language Models (SLMs) deployed locally (vLLM, Ollama, SGLang) suffer from two major context failure modes:

1. **Referential Dangling**: Standard unconstrained prompt compression (e.g. LLMLingua) evaluates text fragments independently. In multi-turn agent transcripts, it prunes variable declarations or schema antecedents while retaining downstream calls, causing runtime execution errors (`NameError`).
2. **Prefix Cache Invalidation**: Modifying 1 token in the middle of a context invalidates PagedAttention block hashes, forcing the local GPU engine to recompute key-value tensors for thousands of tokens.

---

## The Nexus-Context Solution

`nexus-context` addresses these bottlenecks through a 3-pillar architecture:

1. **Submodular Graph Guard (`nexus.guard`)**: Formulates context selection as submodular optimization:
   $$\max_{S \subseteq V, c(S) \le B_T} f(S) = \text{Relevance}(S) + \beta \text{Coverage}(S) - \gamma \text{DanglingPenalty}(S)$$
   With $\gamma \to \infty$, no node $u$ can be selected without its antecedent definitions in $S$.

2. **Block-Aligned Prefix Caching (`nexus.cache`)**: Pads Zone P (System Prompt + Schemas) to exact `B_block` boundaries (16 or 32 tokens) and freezes its SHA-256 hash across session turns.

3. **WWW Memory Governance (`nexus.memory`)**: Converts pruned turns into compact state-mutation tuples $\langle \text{Who, What, When, Where} \rangle$ with exponential temporal and AST depth decay:
   $$W(t, s) = \exp(-\lambda (T - t)) \cdot (1 + \eta \cdot \text{AST\_Depth}(s))$$
