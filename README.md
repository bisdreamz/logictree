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

## Performance Notes

- Most predictions complete in microseconds; ~1M predictions/second is attainable on modern CPUs in typical scenarios.
- Real-world throughput depends on handler complexity, tree depth and branching, and concurrent load.
- `prune()` and `size()` traverse parts of the tree and can be expensive on very large trees; schedule accordingly.

## License

TBD
