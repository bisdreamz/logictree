//! Custom prediction handlers for tree nodes.
//!
//! This module defines the [`PredictionHandler`] trait that enables custom
//! prediction logic at each node in the tree.
//!
//! Handlers manage state, implement learning and prediction logic, and can be
//! as simple or sophisticated as needed. They must be thread-safe (`Send + Sync`)
//! for concurrent training and prediction.

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
    /// e.g. record new visit(s), sales, whatever
    fn train(&self, input: &I);

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
    /// When a feature has multiple values (e.g., format=[banner, video]), this method
    /// combines the predictions from sibling nodes at the same tree depth and decides
    /// if the aggregate is sufficient to make a prediction.
    ///
    /// The `predictions` vector contains tuples of (value, depth) where depth indicates
    /// the tree level where each prediction came from (0=root, 1=after first feature, etc).
    /// All predictions in the vector will be at the same depth level.
    /// This method is responsible for both aggregating multiple predictions AND deciding
    /// if the aggregate (or single prediction) is sufficient to return (Some) or should
    /// defer to parent (None).
    ///
    /// # Example implementations
    /// - Average with threshold: `if total_samples >= min { Some(avg) } else { None }`
    /// - Maximum: `predictions.into_iter().map(|(v, _)| v).max()`
    /// - Sum with sufficiency: `if sufficient_data { Some(sum) } else { None }`
    fn resolve(&self, predictions: Vec<(O, usize)>) -> Option<O>;
}
