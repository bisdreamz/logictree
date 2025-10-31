/// Trait defining the functional prediction handler at each node,
/// which handles training of new data and predictions.
/// If special handler cleanup is needed upon node pruning
/// (such as eviction due to inactivity) then it is suggested
/// that Drop be implemented for resource cleanup
pub trait PredictionHandler<I, O>: Send + Sync {
    /// Online update this prediction handler with a new training event,
    /// e.g. record new visit(s), sales, whatever
    fn train(&self, input: &I) -> ();

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
}
