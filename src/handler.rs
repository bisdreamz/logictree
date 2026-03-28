//! Custom prediction handlers for tree nodes.
//!
//! This module defines the [`PredictionHandler`] trait that enables custom
//! prediction logic at each node in the tree.
//!
//! Handlers manage state, implement learning and prediction logic, and can be
//! as simple or sophisticated as needed. They must be thread-safe (`Send + Sync`)
//! for concurrent training and prediction.

use crate::Feature;
use smallvec::SmallVec;

/// Trait defining the functional prediction handler at each node.
///
/// Handles training of new data and predictions. Each node in the tree has its own
/// handler instance that accumulates statistics and computes predictions.
///
/// # Thread Safety
///
/// Handlers must be thread-safe (`Send + Sync`) for concurrent training and prediction.
/// Use interior mutability (e.g., `Mutex`, `RwLock`) for mutable state.
///
/// # Resource Cleanup
///
/// If special cleanup is needed upon node pruning (e.g., releasing external resources),
/// implement `Drop` for your handler.
///
/// # Multi-Value Features
///
/// The `resolve()` method is responsible for aggregating peer feature predictions
/// and for implementing logic to determine if the aggregate prediction(s) at a
/// node depth are sufficient, or if the prediction responsibility should be passed
/// upwards to the next parent node.
pub trait PredictionHandler<I, O>: Send + Sync {
    /// Online update this prediction handler with a new training event,
    /// e.g. record new visit(s), sales, whatever.
    ///
    /// `next_feature` is the feature that will be consumed next in the tree traversal,
    /// allowing handlers to track which paths are taken from this node.
    fn train(&self, input: &I, next_feature: Option<&Feature>);

    /// Make a prediction for this associated node.
    /// Always returns a prediction value for internal state.
    fn predict(&self) -> O;

    /// Evaluation method which enables implementation of expiry
    /// logic for tree nodes. E.g. if a node has not seen activity
    /// recently. Applicable while using in an online learning
    /// environment to prevent unbounded tree growth.
    /// If true, will prune the associated node and all children.
    fn should_prune(&self) -> bool;

    /// Return a *new* empty instance of this prediction handler,
    /// invoked as new tree values are encountered and thus
    /// new nodes created
    fn new_instance(&self) -> Self
    where
        Self: Sized;

    /// Resolve predictions from child nodes into a final result.
    ///
    /// Called for every prediction path — single-value, multi-value, and fallback.
    /// Responsible for selecting/aggregating predictions AND deciding if the result
    /// is sufficient to return or should defer to the parent node.
    ///
    /// Each tuple is `(value, depth)` where depth is the tree level it originated from.
    /// Return the selected `(value, depth)` pair, or `None` to defer to parent.
    fn resolve(&self, predictions: SmallVec<[(O, usize); 1]>) -> Option<(O, usize)>;
}
