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
    fn predict(&self) -> Option<u64> { Some(*self.total.lock().unwrap()) }
    fn should_prune(&self) -> bool { false }
    fn new_instance(&self) -> Self { SimpleHandler::new() }
    fn fold(&self, predictions: Vec<u64>) -> Option<u64> {
        // Not using multi-value features in this example
        unreachable!("fold not used in simple example")
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
    pub avg_cpm: f32,   // 1000 * spend / impressions
    pub fill_rate: f32, // impressions / auctions
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

    fn predict(&self) -> Option<AdPrediction> {
        let a = self.agg.lock().unwrap();
        // Example summary logic: require enough data before predicting
        if a.auctions < 1000 || a.impressions == 0 { return None; }
        Some(AdPrediction {
            avg_cpm: (a.spend as f64 * 1000.0 / a.impressions as f64) as f32,
            fill_rate: (a.impressions as f64 / a.auctions as f64) as f32,
        })
    }

    // Example pruning policy for online usage: evict inactive nodes
    fn should_prune(&self) -> bool {
        let a = self.agg.lock().unwrap();
        let idle_secs = now_secs().saturating_sub(a.last_active_secs);
        idle_secs > 10 * 60 // e.g., inactive for > 10 minutes
    }

    fn new_instance(&self) -> Self { AdHandler::new() }

    fn fold(&self, predictions: Vec<AdPrediction>) -> Option<AdPrediction> {
        // Not using multi-value features in this example
        unreachable!("fold not used in this example")
    }
}
```

The example above shows how handlers can keep rich state, compute multiple metrics, and decide when to return `None` so the tree falls back to parent nodes. If you allocate external resources, implement `Drop` for cleanup when nodes are pruned.

## Step 2 — Use the Tree

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

In real-world traffic shaping and ad bidding systems, requests often contain **multi-valued features**. For example:
- An OpenRTB bid request might support both **banner and video** formats
- A CTV request might contain **10 identical video impression opportunities**
- A content page might belong to multiple categories (**sports and news**)

When manually handling this by making separate `train()` calls (one for "USA,Android,banner" and another for "USA,Android,video"), the parent nodes incorrectly **double-count** the auction:

```rust
// ❌ WRONG: Double-counts parent metrics
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

### The Solution: Native Multi-Value Support

LogicTree now supports **multi-value features** that properly attribute metrics across feature combinations:

```rust
// ✅ CORRECT: Single train call with multi-value feature
tree.train(vec![
    Feature::string("country", "USA"),
    Feature::string("device", "Android"),
    Feature::multi_string("format", vec!["banner", "video"]),  // Multi-value!
], &event)?;

// Result:
// - USA node sees 1 auction (correct!)
// - USA->Android node sees 1 auction (correct!)
// - USA->Android->banner sees 1 auction
// - USA->Android->video sees 1 auction
```

### How It Works

When training with multi-value features:
1. **Parent nodes are visited once** (USA, then Android)
2. **Each parent receives the full training input** (1 auction count)
3. **Child branches are created for each value** (banner AND video)
4. **Each child receives the same full input** (banner: 1 auction, video: 1 auction)

This maintains **mathematical correctness**: parent nodes aggregate properly while child nodes accumulate format-specific signal.

### Prediction with Multi-Value Features

When predicting with multi-value features, **peer predictions are aggregated** using your `fold()` implementation:

```rust
impl PredictionHandler<AdEvent, AdPrediction> for AdHandler {
    // ... train, predict, etc.

    /// Aggregate predictions from peer nodes (e.g., banner + video)
    fn fold(&self, predictions: Vec<AdPrediction>) -> Option<AdPrediction> {
        if predictions.is_empty() { return None; }

        // Average CPM across formats
        let avg_cpm = predictions.iter()
            .map(|p| p.avg_cpm)
            .sum::<f32>() / predictions.len() as f32;

        // Average fill rate
        let avg_fill = predictions.iter()
            .map(|p| p.fill_rate)
            .sum::<f32>() / predictions.len() as f32;

        Some(AdPrediction { avg_cpm, fill_rate: avg_fill })
    }
}

// Predict with multi-value format
let prediction = tree.predict(vec![
    Feature::string("country", "USA"),
    Feature::string("device", "Android"),
    Feature::multi_string("format", vec!["banner", "video"]),
])?;
// Returns: Average of banner and video predictions via fold()
```

**Note**: `fold()` is **only called when 2+ predictions exist**. Single predictions are returned directly without aggregation.

### Real-World Example: Traffic Shaping

```rust
use logictree::{LogicTree, Feature, PredictionHandler};

#[derive(Serialize, Deserialize)]
struct TrafficHandler {
    auctions: Mutex<u64>,
    revenue: Mutex<f64>,
}

impl PredictionHandler<BidEvent, f64> for TrafficHandler {
    fn train(&self, event: &BidEvent) {
        *self.auctions.lock().unwrap() += 1;
        *self.revenue.lock().unwrap() += event.cpm;
    }

    fn predict(&self) -> Option<f64> {
        let auctions = *self.auctions.lock().unwrap();
        let revenue = *self.revenue.lock().unwrap();
        if auctions < 100 { return None; }  // Need enough data
        Some(revenue / auctions as f64)  // Predict average CPM
    }

    fn fold(&self, predictions: Vec<f64>) -> Option<f64> {
        // Average CPM across formats for blended prediction
        Some(predictions.iter().sum::<f64>() / predictions.len() as f64)
    }

    fn should_prune(&self) -> bool { false }
    fn new_instance(&self) -> Self { TrafficHandler::new() }
}

// Create tree for traffic shaping
let tree = LogicTree::new(
    vec!["country".into(), "device".into(), "format".into()],
    TrafficHandler::new()
);

// Train with OpenRTB bid request supporting multiple formats
tree.train(vec![
    Feature::string("country", "USA"),
    Feature::string("device", "mobile"),
    Feature::multi_string("format", vec!["banner", "video"]),  // Multi-format support
], &BidEvent { cpm: 2.50 })?;

// Later: Make bidding decision
let predicted_cpm = tree.predict(vec![
    Feature::string("country", "USA"),
    Feature::string("device", "mobile"),
    Feature::multi_string("format", vec!["banner", "video"]),
])?;

// Result: Accurate blended CPM prediction for traffic shaping decisions
// Parent nodes (USA, USA->mobile) have correct auction counts for fallback predictions
```

### Key Benefits

✅ **Accurate parent metrics** - No double-counting in parent nodes
✅ **Rich child signal** - Each format accumulates its own statistics
✅ **Reliable fallbacks** - Parent predictions based on correct aggregations
✅ **Flexible aggregation** - Custom `fold()` logic for your domain
✅ **Automatic deduplication** - `multi_string(vec!["banner", "banner", "video"])` → `["banner", "video"]`
✅ **Performance optimized** - Single-value features use SmallVec (zero allocation)

### Constructor Reference

```rust
// Single-value constructors (zero heap allocation via SmallVec)
Feature::string("country", "USA")
Feature::i32("age", 25)
Feature::boolean("premium", true)

// Multi-value constructors (automatic deduplication)
Feature::multi_string("format", vec!["banner", "video"])
Feature::multi_i32("categories", vec![1, 2, 3])
Feature::multi_boolean("flags", vec![true, false])
```

### Constraints

- **Empty values are invalid**: `Feature::multi_string("format", vec![])` will return an error during `train()`/`predict()`
- **fold() must be implemented**: The default implementation panics with a descriptive error if you use multi-value features without implementing `fold()`
- **Values are deduplicated**: Duplicates are automatically removed using HashSet
- **Guaranteed aggregation**: `fold()` is only called when 2+ peer predictions exist

## Performance Notes

- Most predictions complete in microseconds; ~1M predictions/second is attainable on modern CPUs in typical scenarios.
- Real-world throughput depends on handler complexity, tree depth and branching, and concurrent load.
- `prune()` and `size()` traverse parts of the tree and can be expensive on very large trees; schedule accordingly.
- **Multi-value optimization**: Single-value features use SmallVec for zero-allocation storage; multi-value features deduplicate via HashSet

## License

TBD
