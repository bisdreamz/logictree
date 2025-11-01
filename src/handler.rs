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
/// The `fold()` method is required for all implementations. If you don't use multi-value
/// features, you can implement it with `unreachable!()` or `panic!()` as it will never be called.
pub trait PredictionHandler<I, O>: Send + Sync {
    /// Online update this prediction handler with a new training event,
    /// e.g. record new visit(s), sales, whatever
    fn train(&self, input: &I);

    /// Make a prediction for this associated node,
    /// or if insufficient activity exists at this node depth
    /// yet (e.g. too few visits yet) may return an empty
    /// option to pass the decision up to the broader parent
    fn predict(&self) -> Option<O>;

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

    /// Aggregate predictions from peer child nodes when using multi-valued features.
    ///
    /// When a feature has multiple values (e.g., format=[banner, video]), this method
    /// combines the predictions from sibling nodes at the same tree depth before
    /// returning the result up the hierarchy.
    ///
    /// The `predictions` vector is guaranteed to have **2 or more** elements when called.
    /// Single predictions are returned directly without calling fold.
    ///
    /// # Example implementations
    /// - Average: `Some(predictions.iter().sum() / predictions.len())`
    /// - Maximum: `predictions.into_iter().max()`
    /// - Sum: `Some(predictions.iter().sum())`
    /// - No-op (if not using multi-value): `unreachable!("not using multi-value features")`
    fn fold(&self, predictions: Vec<O>) -> Option<O>;
}
