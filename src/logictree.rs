use std::string::ToString;
use std::sync::Arc;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use serde::de::DeserializeOwned;
use crate::feature::Value;
use crate::node::Node;
use crate::{Feature, PredictionHandler};

/// A high-performance decision tree that accumulates outcomes at each node and
/// enables the user to define complex prediction logic, while supporting online learning.
///
/// The tree traverses features in a predefined order, with each node containing
/// a user-defined [`PredictionHandler`] that manages training and prediction logic.
/// Thread-safe for concurrent training and prediction operations
#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "H: Serialize, Node<I, O, H>: Serialize",
    deserialize = "H: DeserializeOwned, Node<I, O, H>: for<'a> Deserialize<'a>"
))]
pub struct LogicTree<I, O, H>
where H: PredictionHandler<I, O>
{
    features: Vec<String>,
    root: DashMap<Value, Node<I, O, H>>,
}

impl <I, O, H> LogicTree<I, O, H>
where H: PredictionHandler<I, O> + for<'de> Deserialize<'de> + Serialize + Clone
{
    fn root_value() -> Value {
        Value::String("root".to_string())
    }

    /// Construct a new LogicTree with the provided features.
    /// Features are immutable and ordering depicts the tree
    /// layout. For example, typically lower cardinality
    /// features may be placed first.
    pub fn new(features: Vec<String>, handler: H) -> LogicTree<I, O, H> {
        let map = DashMap::new();

        map.insert(Self::root_value(), Node::new(Arc::new(handler.new_instance())));

        LogicTree { features, root: map }
    }

    pub fn save(&self, path: &str) -> Result<(), String>
    where
        H: Serialize
    {
        let config = bincode::config::standard();
        let bytes = bincode::serde::encode_to_vec(self, config)
            .map_err(|e| e.to_string())?;
        std::fs::write(path, bytes).map_err(|e| e.to_string())
    }

    pub fn load(path: &str) -> Result<Self, String>
    where
        H: DeserializeOwned
    {
        let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
        let config = bincode::config::standard();
        let (tree, _) = bincode::serde::decode_from_slice(&bytes, config)
            .map_err(|e| e.to_string())?;
        Ok(tree)
    }

    /// Validate the provided feature name -> value pairs are in correct order.
    /// allows incomplete arrays as long as theyre in proper contiguous order
    /// since we can still make partial tree predictions
    fn validate(&self, features: &[Feature]) -> Result<(), String> {
        if features.len() > self.features.len() {
            return Err(format!("Tree only has {} features but received input of {}",
                       self.features.len(), features.len()));
        }

        for (i, feat) in features.iter().enumerate() {
            if !feat.key.eq(&self.features[i]) {
                return Err(format!("Found feat key `{}` but expected `{}`", feat.key, self.features[i]));
            }
        }

        Ok(())
    }

    /// Extract the feature order stack from a map input.
    /// Validates contiguous ordered values are provided.
    fn extract(&self, map: &std::collections::HashMap<&str, Feature>) -> Result<Vec<Feature>, String> {
        let mut features = Vec::with_capacity(map.len());

        for (_, key) in self.features.iter().enumerate() {
            let map_en = map.get(key.as_str());

            if map_en.is_none() {
                break // hit end of contiguous matches
            }

            features.push(map_en.unwrap().clone());
        }

        // must complain if map provided is missing interim values, this is NOT okay
        // if latter values are supplied e.g. [a,b, <missing>, d] is not okay because we
        // would improperly only make a prediction based on [a,b] if following valid vals
        if features.len() < map.len() {
            Err(format!("Break in contiguous values at missing feature `{}`",
                        features.get(features.len()).unwrap()))?
        }

        Ok(features)
    }

    /// Updates the tree, and all associated feature nodes
    /// with the associated update data, which will be
    /// passed to each node's prediction handler
    pub fn train(&self, features: Vec<Feature>, update: &I) -> Result<(), String> {
        self.validate(&features)?;

        self.root.get(&Self::root_value())
            .expect("should have root node!")
            .train(&features, update);

        Ok(())
    }

    pub fn train_map(&self, data: std::collections::HashMap<&str, Feature>, update: &I) -> Result<(), String> {
        self.train(self.extract(&data)?, update)
    }

    /// Make a prediction for the associated feature vector,
    /// which must be in the same order as the constructor.
    /// Will attempt to make a prediction at the deepest node
    /// first, and walk up the tree (parent feature nodes)
    /// returning the first non empty node result. This
    /// minimum prediction logic is handler impl specific.
    /// Contiguous values are required, but the full
    /// feature chain is not strictly required. E.g. for a
    /// tree of [a,b,c] a prediction pf [a,b] is acceptable
    /// however [a,c] is not.
    pub fn predict(&self, features: Vec<Feature>) -> Result<Option<O>, String> {
        self.validate(&features)?;


        let res = self.root.get(&Self::root_value())
            .expect("should have root node!")
            .predict(&features);

        Ok(res)
    }

    /// Convenience method to predict based on map input of key -> feature values, and the
    /// tree handles organizing feature order. See `predict` in which the same validation rules apply.
    pub fn predict_map(&self, data: std::collections::HashMap<&str, Feature>) -> Result<Option<O>, String> {
        self.predict(self.extract(&data)?)
    }

    /// Perform pruning of the tree per the handlers should_prune logic.
    /// If this returns true, then the whole map has been pruned (empty)
    pub fn prune(&self) -> bool {
        self.root.get(&Self::root_value())
            .expect("should have root node!")
            .should_prune()
    }

    /// Return the count of nodes in this map. Optionally return
    /// the count of only leaf nodes.
    pub fn size(&self, leaf_only: bool) -> u32 {
        self.root.get(&Self::root_value())
            .expect("should have root node!")
            .size(leaf_only)
    }

}