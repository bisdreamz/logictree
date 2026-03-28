//! A high-performance decision tree library for online learning with multi-value feature support.
//!
//! # Overview
//!
//! LogicTree is a thread-safe, concurrent decision tree designed for:
//! - **Online learning**: Continuous training from streaming data
//! - **Custom prediction logic**: Implement your own handlers at each node
//! - **Multi-value features**: Properly handle multi-valued inputs without double-counting
//! - **Automatic fallback**: Parent nodes provide predictions when child data is insufficient
//! - **Tree pruning**: Evict inactive nodes to prevent unbounded growth
//!
//! # Multi-Value Features
//!
//! The key innovation is native support for **multi-value features**, crucial for scenarios like:
//! - **Ad serving**: Requests supporting multiple formats (banner + video)
//! - **Content classification**: Pages belonging to multiple categories
//! - **Multi-device targeting**: Campaigns targeting multiple device types
//!
//! Traditional approaches double-count parent metrics when training separate paths.
//! LogicTree solves this by visiting parent nodes once while creating multiple child branches.
//!
//! # Quick Start
//!
//! ```rust
//! use logictree::{LogicTree, PredictionHandler, Feature, PredictionResult};
//! use smallvec::SmallVec;
//! use std::sync::Mutex;
//! # use serde::{Serialize, Deserialize};
//!
//! // Define a handler that tracks a running total and selects the
//! // highest-value prediction when multiple children match.
//! #[derive(Serialize, Deserialize)]
//! struct MaxValueHandler {
//!     total: Mutex<u32>,
//! }
//!
//! # impl Clone for MaxValueHandler {
//! #     fn clone(&self) -> Self {
//! #         MaxValueHandler { total: Mutex::new(*self.total.lock().unwrap()) }
//! #     }
//! # }
//!
//! impl MaxValueHandler {
//!     fn new() -> Self {
//!         MaxValueHandler { total: Mutex::new(0) }
//!     }
//! }
//!
//! impl PredictionHandler<u32, u32> for MaxValueHandler {
//!     fn train(&self, input: &u32, _next: Option<&Feature>) {
//!         *self.total.lock().unwrap() += input;
//!     }
//!
//!     fn predict(&self) -> u32 {
//!         *self.total.lock().unwrap()
//!     }
//!
//!     fn resolve(&self, predictions: SmallVec<[(u32, usize); 1]>) -> Option<(u32, usize)> {
//!         // Select the prediction with the highest value
//!         predictions.into_iter().max_by_key(|(value, _)| *value)
//!     }
//!
//!     fn should_prune(&self) -> bool { false }
//!     fn new_instance(&self) -> Self { MaxValueHandler::new() }
//! }
//!
//! // Create tree with ordered features
//! let tree = LogicTree::new(
//!     vec!["country".to_string(), "format".to_string()],
//!     MaxValueHandler::new()
//! );
//!
//! // Train — parent (USA) trained once, children (banner, video) each trained once
//! tree.train(&vec![
//!     Feature::string("country", "USA"),
//!     Feature::multi_string("format", vec!["banner", "video"]),
//! ], &10).unwrap();
//!
//! // Predict — resolve picks the highest-value child
//! let prediction = tree.predict(&vec![
//!     Feature::string("country", "USA"),
//! ]).unwrap().unwrap();
//!
//! assert_eq!(prediction.depth, 1);
//! assert_eq!(prediction.full_depth, false);
//! ```
//!
//! # Performance
//!
//! - **Thread-safe**: Uses DashMap for lock-free concurrent access
//! - **Zero-allocation**: Single-value features use SmallVec (inline storage)
//! - **Fast predictions**: ~1M predictions/second on modern CPUs
//! - **Automatic deduplication**: Multi-value features deduplicate via HashSet
//!
//! # See Also
//!
//! - [`LogicTree`]: Main tree structure
//! - [`PredictionHandler`]: Trait for custom prediction logic
//! - [`Feature`]: Feature input types (single and multi-value)
//! - See README for detailed multi-value feature examples

mod feature;
mod handler;
mod logictree;
mod node;

/// Result from a tree prediction containing the value and depth information.
///
/// * `value` - The predicted value returned by the handler
/// * `depth` - The tree level where this prediction originated (0 = root, 1 = first feature, etc.)
/// * `full_depth` - Whether this prediction is from the deepest possible level (all features matched)
#[derive(Debug, Clone, PartialEq)]
pub struct PredictionResult<O> {
    pub value: O,
    pub depth: usize,
    pub full_depth: bool,
}

pub use feature::*;
pub use handler::*;
pub use logictree::*;
