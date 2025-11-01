# Multi-Value Feature Aggregation Problem

## Executive Summary

Multi-value features successfully solve the **training double-counting problem** (parent nodes see correct counts), but introduce a fundamental **prediction aggregation challenge** when trying to combine predictions from peer nodes that represent co-occurring feature values.

## Current Implementation

### Training Behavior (Working Correctly)

When training with `[country=USA, device=android, format=[banner, video]]`:

```
Tree traversal:
1. Root receives training input (1 request)
2. USA node receives training input (1 request)
3. Android node receives training input (1 request)
4. BOTH banner AND video nodes receive training input (1 request each)
```

**Result**:
- USA node: `requests = 1` ✓ Correct
- Android node: `requests = 1` ✓ Correct
- Banner node: `requests = 1` ✓ (from its perspective)
- Video node: `requests = 1` ✓ (from its perspective)

**Parent nodes do NOT double-count** - this was the original problem we solved.

### Prediction Behavior (Current Implementation)

Located in `src/node.rs:49-75`:

```rust
pub fn predict(&self, stack: &[Feature]) -> Option<O> {
    if stack.is_empty() {
        return self.handler.predict();
    }

    let feat = &stack[0];

    // Collect predictions from all children that return Some
    let mut predictions = Vec::new();
    for value in &feat.values {
        if let Some(child) = self.children.get(value)
            && let Some(prediction) = child.predict(&stack[1..]) {
                predictions.push(prediction);
            }
    }

    match predictions.len() {
        0 => self.handler.predict(),           // No children predicted: use parent
        1 => predictions.into_iter().next(),    // Single child: return directly (no fold)
        _ => {                                  // Multiple children: fold them
            if let Some(folded) = self.handler.fold(predictions) {
                Some(folded)
            } else {
                self.handler.predict()          // Fold returned None: use parent
            }
        }
    }
}
```

**Key behaviors**:
1. Only collects predictions that return `Some` (allows partial predictions)
2. Single prediction returned directly without calling fold
3. Multiple predictions passed to `fold()` for aggregation
4. Falls back to parent if no predictions or fold returns None

## The Core Problem

### Scenario

Historical training data:
- 100 requests with banner format → Banner node: `{auctions: 100, revenue: $100}`
- 10 requests with video format → Video node: `{auctions: 10, revenue: $1}`

Some of these requests had BOTH banner and video (co-occurring), but we don't track which ones.

### Prediction Query

User predicts with `format=[banner, video]` (representing a request supporting both formats).

**Banner node predicts**:
```rust
{
    rpm: $100 / 100 * 1000 = $1000,
    fill_rate: 10%,
    auctions: 100  // sample size
}
```

**Video node predicts**:
```rust
{
    rpm: $1 / 10 * 1000 = $100,
    fill_rate: 10%,
    auctions: 10  // sample size
}
```

### The Aggregation Challenge

**fold()** receives both predictions. How should it combine them?

#### Option A: Naive Average
```rust
rpm = ($1000 + $100) / 2 = $550
```
**Problem**: Ignores that banner has 10x more data than video.

#### Option B: Weighted Average
```rust
rpm = ($1000 * 100 + $100 * 10) / (100 + 10)
    = $100,100 / 110
    = $910
```

**Problem**: Denominat or is wrong! The 100 banner requests and 10 video requests are NOT independent:
- Some requests had BOTH formats (we trained them on both paths)
- True denominator is unknown: could be 100-110 depending on overlap
- Using 110 **under-reports** RPM if there was significant co-occurrence

#### Option C: Max Weighting
```rust
rpm = ($1000 * 100 + $100 * 10) / max(100, 10)
    = $100,100 / 100
    = $1001
```

**Problem**: Assumes ALL video requests co-occurred with banner (not necessarily true).

### The Fundamental Issue

**When training with multi-value features, we create separate statistical views (banner stats, video stats) that are not independent samples.**

The banner node's 100 requests and video node's 10 requests have an unknown overlap. Without tracking co-occurrence, we cannot correctly aggregate their derived metrics (rates, ratios, RPM).

## Options Considered

### Option 1: Require ALL Children to Predict (Strict Mode)

**Implementation**:
```rust
if predictions.len() == feat.values.len() {
    // All children returned Some, safe to fold
    self.handler.fold(predictions)
} else {
    // Partial data, fall back to parent
    self.handler.predict()
}
```

**Pros**:
- Clear semantics: either use all children or use parent
- Avoids partial/biased aggregations

**Cons**:
- Breaks use cases with large multi-value sets (e.g., 50 deal IDs)
- Would rarely leverage child-level knowledge
- Too restrictive for sparse features

**Verdict**: Rejected - too restrictive for real-world use cases like deal_id features.

---

### Option 2: Allow Partial Predictions (Current Implementation)

**Implementation**: Already implemented (see "Prediction Behavior" above)

**Pros**:
- Flexible: uses whatever child data is available
- Works well for sparse features (deal IDs)
- Each child's `predict()` returning Some/None acts as "am I useful?" signal

**Cons**:
- fold() doesn't know if predictions are partial or complete
- fold() still faces aggregation challenge regardless of completeness

**Verdict**: Currently implemented, but doesn't solve aggregation challenge.

---

### Option 3: Pass Complete Context to fold()

**Implementation**:
```rust
// fold receives ALL children's results, including None
fn fold(&self, results: Vec<(&Value, Option<O>)>) -> Option<O> {
    // Can see which children exist and which predicted
}
```

**Pros**:
- Handler sees complete picture
- Can make informed decisions about partial data
- Maximum flexibility

**Cons**:
- Breaking API change
- More complex for handler implementers
- **Still doesn't solve the aggregation math problem**

**Verdict**: Adds complexity but doesn't solve the core issue.

---

### Option 4: Remove Option from predict(), Move to fold()

**Implementation**:
```rust
trait PredictionHandler<I, O> {
    fn predict(&self) -> O;  // Always returns something
    fn fold(&self, predictions: Vec<O>) -> Option<O>;  // Decides usefulness
}
```

**Pros**:
- Cleaner separation: handlers predict, fold decides usefulness
- Can include confidence/metadata in prediction type

**Cons**:
- Breaking API change
- **Still doesn't solve the aggregation math problem**
- Handlers must always return something (even with insufficient data)

**Verdict**: Cleaner API but doesn't address the fundamental issue.

---

### Option 5: Include Sample Metadata in Predictions

**Implementation**:
```rust
struct AdPrediction {
    rpm: f32,
    fill_rate: f32,
    auctions: u64,     // Include sample size
    impressions: u64,  // Include raw counts
}

fn fold(&self, predictions: Vec<AdPrediction>) -> Option<AdPrediction> {
    // Use sample sizes for weighting
    let total_auctions = predictions.iter().map(|p| p.auctions).sum();
    let weighted_rpm = predictions.iter()
        .map(|p| p.rpm * p.auctions as f32)
        .sum::<f32>() / total_auctions as f32;
}
```

**Pros**:
- fold() can weight predictions by sample size
- Works for independent samples

**Cons**:
- **Doesn't solve co-occurrence problem**: sample sizes are not independent
- Banner's 100 requests + Video's 10 requests ≠ 110 independent samples
- Weighted average still produces incorrect results

**Verdict**: Helps with weighting but assumes independence that doesn't exist.

---

### Option 6: Track Co-occurrence with Bitsets

**Implementation**:
```rust
struct HandlerState {
    auctions: u64,
    revenue: f64,
    request_ids: SparseBitSet,  // Track which specific requests
}

fn fold(&self, predictions: Vec<Prediction>) -> Option<Prediction> {
    let union_request_ids = predictions.iter()
        .flat_map(|p| p.request_ids.iter())
        .collect();
    let actual_request_count = union_request_ids.len();
    // Use actual_request_count for correct aggregation
}
```

**Pros**:
- Mathematically correct
- Can calculate exact overlaps

**Cons**:
- Memory explosion with millions of requests
- Massive complexity
- Defeats purpose of aggregated statistics
- Not practical for online learning

**Verdict**: Theoretically correct but practically impossible.

---

### Option 7: Always Use Parent for Multi-Value Predictions

**Implementation**:
```rust
// When predicting with multi-value features, ignore fold entirely
if feat.values.len() > 1 {
    return self.handler.predict();  // Use parent node
}
// Only use children for single-value predictions
```

**Pros**:
- Parent node has correct blended statistics
- No aggregation challenges
- Simple and correct

**Cons**:
- Loses child-specific signal for multi-value queries
- **Can't leverage deeper nodes below multi-value feature**
- Example: `[USA, android, [banner, video], wifi, 300x250]`
  - Using parent at format level loses wifi and 300x250 specificity!

**Verdict**: Rejected - unacceptable because deeper children below multi-value features contain valuable signal that would be lost.

---

### Option 8: Don't Allow Multi-Value for Prediction

**Implementation**: Multi-value features only used for training (to prevent double-counting). Predictions must use single values.

**Pros**:
- Training benefits remain (no parent double-counting)
- No aggregation challenges
- Clear semantics

**Cons**:
- Can't predict for actual multi-value requests
- Defeats purpose of multi-value feature support
- User must predict each format separately and manually aggregate

**Verdict**: Defeats the purpose of multi-value features.

## The Deeper Problem: Children Below Multi-Value Features

### Why Option 7 (Parent Fallback) is Unacceptable

Consider tree structure:
```
USA → android → [banner, video] → wifi → 300x250
```

If we predict with `[USA, android, [banner, video], wifi, 300x250]` and fall back to parent at format level:
- We'd use predictions from: `USA → android`
- We'd LOSE signal from: `wifi → 300x250`

**These deeper nodes contain critical information**:
- Network type (wifi vs cellular)
- Ad size (300x250 vs 728x90)
- Other contextual features

**The real use case**: Predict at maximum depth while handling multi-value features correctly along the path.

## Current Status: Impasse

We have successfully solved the **training problem** (parent nodes don't double-count), but we're at an impasse for the **prediction problem**:

1. **Can't correctly aggregate co-occurring feature statistics** without tracking overlap
2. **Can't use parent fallback** because we lose deeper child signal
3. **Can't track overlap** due to memory/complexity explosion
4. **Can't ignore the problem** because predictions will be mathematically incorrect

## Open Questions

1. **Is approximate aggregation acceptable?**
   - Use weighted average knowing it's incorrect but "close enough"?
   - Document the limitation and accept the bias?

2. **Should multi-value predictions be discouraged?**
   - Only use multi-value for training?
   - Force users to predict single formats and aggregate outside the tree?

3. **Is there a bounded co-occurrence tracking approach?**
   - Track only recent N request IDs?
   - Use probabilistic data structures (Bloom filters)?
   - Accept approximate overlap calculations?

4. **Should the API make the limitation explicit?**
   - Return predictions with confidence intervals?
   - Include "these are not independent samples" warnings?
   - Provide utilities for manual aggregation?

5. **Is the user's use case actually solvable within this framework?**
   - Do they need exact aggregation, or is approximate acceptable?
   - Can they structure features differently to avoid the problem?
   - Should we recommend different tree structures for their use case?

## Recommendation Needed

We need direction on:
1. Whether approximate aggregation is acceptable for the traffic shaping use case
2. How to handle the trade-off between child signal and correct aggregation
3. Whether to document limitations or implement partial solutions
4. What level of error is acceptable in revenue/RPM predictions

The current implementation allows flexible prediction but produces mathematically incorrect results when aggregating co-occurring multi-value features. We need to decide if this is acceptable or if we need to restrict the feature set.
