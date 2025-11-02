# LogicTree

LogicTree is a concurrent decision tree for online learning. You supply the prediction handlers that live at each node. Handlers keep state, implement your learning and prediction logic, and can be as simple or as sophisticated as you need. Parents naturally see all updates for their descendants, so they accumulate broader—but more generalized—signal. In online environments, pruning can evict inactive nodes to prevent unbounded growth.

## Highlights

- **Concurrent:** `train` and `predict` can run at the same time; internals use `dashmap` for scalable access.
- **Online:** Incremental updates on every event; designed for always-on learning.
- **Fallback Control:** You decide prediction depth. If a deeper path doesn’t exist or you choose a shallower depth, the parent predicts using its broader data. If a deeper node’s handler returns `None` (insufficient data), the tree automatically searches upward to the nearest ancestor that returns `Some`.
- **Custom Handlers:** Implement `PredictionHandler<I, O>` to own state, define prediction logic, and decide when to prune.
- **Persistable:** Save/load entire trees via `serde` + `bincode`.
- **Fast:** Predictions typically take microseconds; ~1M predictions/second is attainable on modern CPUs (handler/depth dependent).
- **Online or Batch:** Fully online learning is supported; a traditional train → save → load flow works as well.

## How It Works

- **Feature Order:** Define the complete, fixed feature order at construction (e.g., `[country, device, browser]`). Prediction inputs must follow this order. You may provide a partial, contiguous prefix at prediction time (e.g., `[country]` or `[country, device]`), but not gaps.
- **Parent Usage:** If a deeper feature value is unknown (no child node) or you predict at a shallower depth, the parent makes the prediction. If a deeper node returns `None` (not enough data at that depth), the tree walks upward until an ancestor returns `Some`.
- **Pruning:** Your handler’s `should_prune()` can signal eviction for inactive nodes. You are responsible for calling `prune()` (e.g., on a schedule); it may be expensive for large trees.
- **Map Helpers:** Convenience methods accept a `HashMap<&str, Feature>` so you can feed data from sources like JSON → map and let the tree enforce ordering and contiguity for you.
- **Cleanup Hooks:** If your handler needs resource cleanup on eviction, it can implement `Drop` in addition to `PredictionHandler`.
- **Node Creation:** `new_instance()` must return a new, empty handler. The tree calls it when creating new child nodes for newly observed feature values.

## Step 1 — Define Your Prediction Handler

Start simple: a counter that learns online and predicts a sum.

```rust
use logictree::PredictionHandler;
use serde::{Serialize, Deserialize};
use std::sync::Mutex;

#[derive(Serialize, Deserialize)]
pub struct SimpleHandler { total: Mutex<u64> }

impl Clone for SimpleHandler {
    fn clone(&self) -> Self { Self { total: Mutex::new(*self.total.lock().unwrap()) } }
}

impl SimpleHandler { pub fn new() -> Self { Self { total: Mutex::new(0) } } }

impl PredictionHandler<u64, u64> for SimpleHandler {
    fn train(&self, input: &u64) { *self.total.lock().unwrap() += *input; }
    fn predict(&self) -> u64 { *self.total.lock().unwrap() }
    fn should_prune(&self) -> bool { false }
    fn new_instance(&self) -> Self { SimpleHandler::new() }
    fn resolve(&self, predictions: Vec<u64>) -> Option<u64> {
        // Simple passthrough for single predictions
        predictions.into_iter().next()
    }
}
```

Handlers can also manage rich state and work with real domain types. For example, an ad-quality model that ingests events and predicts multiple KPIs:

```rust
use logictree::PredictionHandler;
use serde::{Serialize, Deserialize};
use std::sync::Mutex;

#[derive(Serialize, Deserialize, Clone, Default)]
pub struct AdAgg {
    pub auctions: u64,
    pub impressions: u64,
    pub spend: f64,
    pub last_active_secs: u64,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct AdEvent {
    pub auctions: u32,
    pub impressions: u32,
    pub spend: f32,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct AdPrediction {
    pub auctions: u64,
    pub impressions: u64,
    pub spend: f64,
}

#[derive(Serialize, Deserialize)]
pub struct AdHandler { agg: Mutex<AdAgg> }

impl Clone for AdHandler {
    fn clone(&self) -> Self { Self { agg: Mutex::new(self.agg.lock().unwrap().clone()) } }
}

impl AdHandler { pub fn new() -> Self { Self { agg: Mutex::new(AdAgg::default()) } } }

fn now_secs() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs()
}

impl PredictionHandler<AdEvent, AdPrediction> for AdHandler {
    fn train(&self, e: &AdEvent) {
        let mut a = self.agg.lock().unwrap();
        a.auctions += e.auctions as u64;
        a.impressions += e.impressions as u64;
        a.spend += e.spend as f64;
        a.last_active_secs = now_secs();
    }

    fn predict(&self) -> AdPrediction {
        let a = self.agg.lock().unwrap();
        AdPrediction {
            auctions: a.auctions,
            impressions: a.impressions,
            spend: a.spend,
        }
    }

    // Example pruning policy for online usage: evict inactive nodes
    fn should_prune(&self) -> bool {
        let a = self.agg.lock().unwrap();
        let idle_secs = now_secs().saturating_sub(a.last_active_secs);
        idle_secs > 10 * 60 // e.g., inactive for > 10 minutes
    }

    fn new_instance(&self) -> Self { AdHandler::new() }

    fn resolve(&self, predictions: Vec<AdPrediction>) -> Option<AdPrediction> {
        let total_auctions: u64 = predictions.iter().map(|p| p.auctions).sum();
        // Sufficiency check: require enough data before predicting
        if total_auctions < 1000 { return None; }
        Some(AdPrediction {
            auctions: total_auctions,
            impressions: predictions.iter().map(|p| p.impressions).sum(),
            spend: predictions.iter().map(|p| p.spend).sum(),
        })
    }
}
```

The example above shows how handlers can keep rich state, compute multiple metrics, and use `resolve()` to decide sufficiency when aggregating predictions. If you allocate external resources, implement `Drop` for cleanup when nodes are pruned.

## Step 2 — Using the Tree

Create a tree with a fixed feature order, train online, predict with fallback, and persist when needed.

```rust
use logictree::{LogicTree, Feature};

// Example with the AdHandler above
let tree = LogicTree::new(vec!["country".into(), "device".into()], AdHandler::new());

// Online training with ordered, contiguous features
tree.train(vec![
    Feature::string("country", "US"),
    Feature::string("device", "mobile"),
], &AdEvent { auctions: 100, impressions: 60, spend: 25.0 })?;

// Predict at the deepest node; if the child is unseen or its handler returns None
// (e.g., below a threshold), the tree falls back to the parent
let pred = tree.predict(vec![
    Feature::string("country", "US"),
    Feature::string("device", "tablet"), // unseen child -> falls back to country-level
])?;

// Persist the whole tree (handlers must be Serialize/Deserialize)
tree.save("tree.bin")?;
let loaded: LogicTree<AdEvent, AdPrediction, AdHandler> = LogicTree::load("tree.bin")?;

// Maintenance utilities
let _leaf_nodes = loaded.size(true);
let _all_pruned = loaded.prune(); // consults handler.should_prune() at nodes
```

If your features arrive as maps (for example, parsed JSON), use the convenience helpers to let the tree order and validate them:

```rust
use std::collections::HashMap;

let mut m = HashMap::new();
m.insert("country", Feature::string("country", "US"));
m.insert("device", Feature::string("device", "mobile"));

// The helpers enforce contiguity and the constructor’s order
tree.train_map(m.clone(), &AdEvent { auctions: 10, impressions: 6, spend: 2.5 })?;
let _ = tree.predict_map(m)?;

// Partial feature depth prediction (broader node):
let country_only = tree.predict(vec![ Feature::string("country", "US") ])?; // uses country-level aggregation
```

## Multi-Value Features: Solving the Attribution Problem

### The Problem

In real-world traffic shaping and ad bidding systems, requests often contain **multi-valued features at any depth in the tree**. For example:
- An OpenRTB bid request might support both **banner and video** formats
- A CTV request might contain **10 identical video impression opportunities**
- A content page might belong to multiple categories (**sports and news**)
- Any combination: multiple countries, devices, AND formats in the same request

**Multi-value features can appear at ANY level** - not just leaf nodes. You can have multi-value features at the root, middle, or leaf of your feature hierarchy.

When manually handling this by making separate `train()` calls (one for "USA,Android,banner" and another for "USA,Android,video"), the parent nodes incorrectly **double-count** the auction:

```rust
// WRONG: Double-counts parent metrics
tree.train(vec![
    Feature::string("country", "USA"),
    Feature::string("device", "Android"),
    Feature::string("format", "banner"),
], &event)?;

tree.train(vec![
    Feature::string("country", "USA"),
    Feature::string("device", "Android"),
    Feature::string("format", "video"),
], &event)?;  // Same event trained twice!

// Result: USA->Android node sees 2 auctions when there was only 1
```

This breaks the tree's mathematical integrity:
- **Undercounting**: Recording only one format misses valuable signal
- **Overcounting**: Recording all formats separately inflates parent metrics
- **No accurate fallback**: Parent predictions become unreliable for traffic shaping

### The Solution: Multi-Value Features

LogicTree supports **multi-value features** that properly attribute metrics without double-counting:

```rust
// CORRECT: Single train call with multi-value feature
tree.train(vec![
    Feature::string("country", "USA"),
    Feature::string("device", "Android"),
    Feature::multi_string("format", vec!["banner", "video"]),
], &event)?;

// Result:
// - USA node sees 1 auction (correct)
// - USA->Android node sees 1 auction (correct)
// - Both formats are tracked without parent double-counting
```

### How It Works

**Training with multi-value features**:
- Single value `[banner]` → creates direct child "banner"
- Multi-value `[banner, video]` → creates **composite child "banner|video"** (not separate children)
- Parent nodes visited once, receive full training input (no double-counting)

**Prediction with multi-value features**:
```rust
// Historical training:
train([banner]) → 9000 times         // banner-only requests
train([video]) → 500 times           // video-only requests
train([banner, video]) → 1000 times  // co-occurring requests

// Prediction query:
predict([banner, video])

// Collects ALL independent samples:
// - "banner" child (9000 requests)
// - "video" child (500 requests)
// - "banner|video" composite (1000 requests)
//
// resolve() receives all 3 predictions and:
// 1. Aggregates: 10500 total samples with combined metrics
// 2. Decides sufficiency: if samples (10500) < min_samples (e.g., 1000), returns aggregated result
//    If < 1000, returns None to defer to parent
// No overlap! All samples are independent.
```

This maintains **mathematical correctness**: parent nodes aggregate properly, child nodes represent distinct sample sets, and resolve() combines non-overlapping statistics.

### Prediction with Multi-Value Features

When predicting with multi-value features, **peer predictions are aggregated** using your `resolve()` implementation:

```rust
impl PredictionHandler<AdEvent, AdPrediction> for AdHandler {
    // ... train, predict, etc.

    /// Aggregate predictions and decide if sufficient data exists
    fn resolve(&self, predictions: Vec<AdPrediction>) -> Option<AdPrediction> {
        // Sum raw counts across independent samples
        let total_auctions: u64 = predictions.iter().map(|p| p.auctions).sum();
        let total_impressions: u64 = predictions.iter().map(|p| p.impressions).sum();
        let total_spend: f64 = predictions.iter().map(|p| p.spend).sum();

        // Sufficiency check: need enough data to make reliable prediction
        if total_auctions < 1000 {
            return None;  // Defer to parent node
        }

        Some(AdPrediction {
            auctions: total_auctions,
            impressions: total_impressions,
            spend: total_spend,
        })
    }
}

// Predict with multi-value format
let prediction = tree.predict(vec![
    Feature::string("country", "USA"),
    Feature::string("device", "Android"),
    Feature::multi_string("format", vec!["banner", "video"]),
])?;
// Returns: Sum of independent samples from banner + video paths via resolve()
```

**Note**: `resolve()` is called for **ALL child predictions (1 or more)**. It's responsible for both aggregating multiple predictions AND deciding if the result is sufficient to return (Some) or should defer to parent (None).

### Multiple Multi-Value Features

Multi-value features can appear at ANY depth and multiple features can be multi-valued in the same path:

```rust
// Training with multiple multi-value features
tree.train(vec![
    Feature::multi_string("country", vec!["USA", "Canada"]),  // Multi at root
    Feature::string("device", "mobile"),                       // Single value
    Feature::multi_string("format", vec!["banner", "video"]),  // Multi at leaf
], &event)?;

// This trains multiple paths simultaneously:
// - Both USA and Canada paths receive the training data
// - Both banner and video formats are tracked at each location
// - Parent nodes see correct counts (no double-counting)

// Prediction aggregates across all matching paths
let result = tree.predict(vec![
    Feature::multi_string("country", vec!["USA"]),      // Query subset
    Feature::string("device", "mobile"),
    Feature::multi_string("format", vec!["banner"]),    // Query subset
])?;
// Aggregates data from all paths that match the query values
```

### Real-World Example: Ad Decisioning

When evaluating ad requests to determine bid values for buying partners, requests often support multiple formats (banner, video) and sizes. The tree learns from a continuous stream of bid outcomes, initially making predictions with limited data and improving as more training data arrives.

```rust
// Simple handler that tracks bid revenue
struct BidHandler {
    count: Mutex<u64>,
    revenue: Mutex<f64>,
}

impl PredictionHandler<BidEvent, BidValue> for BidHandler {
    fn train(&self, event: &BidEvent) {
        *self.count.lock().unwrap() += 1;
        *self.revenue.lock().unwrap() += event.cpm;
    }

    fn predict(&self) -> BidValue {
        BidValue {
            count: *self.count.lock().unwrap(),
            revenue: *self.revenue.lock().unwrap(),
        }
    }

    fn resolve(&self, predictions: Vec<BidValue>) -> Option<BidValue> {
        let total_count: u64 = predictions.iter().map(|p| p.count).sum();

        // Need sufficient data before making bid decisions
        if total_count < 1000 {
            return None;  // Defer to parent's broader data
        }

        Some(BidValue {
            count: total_count,
            revenue: predictions.iter().map(|p| p.revenue).sum(),
        })
    }

    // ... other required methods
}

// Create tree with geo, device, and format features
let tree = LogicTree::new(
    vec!["geo".into(), "device".into(), "format".into()],
    BidHandler::new()
);

// Train: Ad request supports both banner AND video formats
tree.train(vec![
    Feature::string("geo", "USA"),
    Feature::string("device", "mobile"),
    Feature::multi_string("format", vec!["banner", "video"]),
], &BidEvent { cpm: 2.50 })?;

// Predict: What bid value for USA mobile traffic supporting both formats?
let bid_value = tree.predict(vec![
    Feature::string("geo", "USA"),
    Feature::string("device", "mobile"),
    Feature::multi_string("format", vec!["banner", "video"]),
])?;

// Early predictions may return None (insufficient data), falling back to
// broader segments (e.g., all USA traffic) until enough data accumulates
```

### Key Benefits
- **Accurate parent metrics** - No double-counting in parent nodes
- **Rich child signal** - Each format accumulates its own statistics
- **Reliable fallbacks** - Parent predictions based on correct aggregations
- **Flexible aggregation** - Custom `resolve()` logic for your domain
- **No deduplication** - Caller must ensure no duplicate values are provided
- **Performance optimized** - Single-value features use SmallVec (zero allocation)

### Constructor Reference

```rust
// Single-value constructors (zero heap allocation via SmallVec)
Feature::string("country", "USA")
Feature::i32("age", 25)
Feature::boolean("premium", true)

// Multi-value constructors (caller must ensure no duplicates)
Feature::multi_string("format", vec!["banner", "video"])
Feature::multi_i32("categories", vec![1, 2, 3])
Feature::multi_boolean("flags", vec![true, false])
```

### Constraints

- **Empty values are invalid**: `Feature::multi_string("format", vec![])` will return an error during `train()`/`predict()`
- **resolve() must be implemented**: The default implementation panics with a descriptive error if you use multi-value features without implementing `resolve()`
- **No automatic deduplication**: Providing duplicate values will create duplicate tree paths
- **Guaranteed aggregation**: `resolve()` is called for ALL child predictions (1+) to handle aggregation and sufficiency

## Performance Notes

- Most predictions complete in microseconds; ~1M predictions/second is attainable on modern CPUs in typical scenarios.
- Real-world throughput depends on handler complexity, tree depth and branching, and concurrent load.
- `prune()` and `size()` traverse parts of the tree and can be expensive on very large trees; schedule accordingly.
- **Multi-value optimization**: Single-value features use SmallVec for zero-allocation storage

## License

TBD
