use crate::feature::Value;
use crate::node::Node;
use crate::{Feature, PredictionHandler};
use dashmap::DashMap;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::string::ToString;
use std::sync::Arc;

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
where
    H: PredictionHandler<I, O>,
{
    features: Vec<String>,
    root: DashMap<Value, Node<I, O, H>>,
}

impl<I, O, H> LogicTree<I, O, H>
where
    H: PredictionHandler<I, O> + for<'de> Deserialize<'de> + Serialize,
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

        map.insert(
            Self::root_value(),
            Node::new(Arc::new(handler.new_instance())),
        );

        LogicTree {
            features,
            root: map,
        }
    }

    pub fn save(&self, path: &str) -> Result<(), String>
    where
        H: Serialize,
    {
        let config = bincode::config::standard();
        let bytes = bincode::serde::encode_to_vec(self, config).map_err(|e| e.to_string())?;
        std::fs::write(path, bytes).map_err(|e| e.to_string())
    }

    pub fn load(path: &str) -> Result<Self, String>
    where
        H: DeserializeOwned,
    {
        let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
        let config = bincode::config::standard();
        let (tree, _) =
            bincode::serde::decode_from_slice(&bytes, config).map_err(|e| e.to_string())?;
        Ok(tree)
    }

    /// Validate the provided feature name -> value pairs are in correct order.
    /// allows incomplete arrays as long as theyre in proper contiguous order
    /// since we can still make partial tree predictions
    fn validate(&self, features: &[Feature]) -> Result<(), String> {
        if features.len() > self.features.len() {
            return Err(format!(
                "Tree only has {} features but received input of {}",
                self.features.len(),
                features.len()
            ));
        }

        for (i, feat) in features.iter().enumerate() {
            if !feat.key.eq(&self.features[i]) {
                return Err(format!(
                    "Found feat key `{}` but expected `{}`",
                    feat.key, self.features[i]
                ));
            }

            if feat.values.is_empty() {
                return Err(format!("Feature '{}' cannot have empty values", feat.key));
            }
        }

        Ok(())
    }

    /// Extract the feature order stack from a map input.
    /// Validates contiguous ordered values are provided.
    fn extract(
        &self,
        map: &std::collections::HashMap<&str, Feature>,
    ) -> Result<Vec<Feature>, String> {
        let mut features = Vec::with_capacity(map.len());

        for key in self.features.iter() {
            let map_en = map.get(key.as_str());

            if map_en.is_none() {
                break; // hit end of contiguous matches
            }

            features.push(map_en.unwrap().clone());
        }

        // must complain if map provided is missing interim values, this is NOT okay
        // if latter values are supplied e.g. [a,b, <missing>, d] is not okay because we
        // would improperly only make a prediction based on [a,b] if following valid vals
        if features.len() < map.len() {
            Err(format!(
                "Break in contiguous values at missing feature `{}`",
                self.features.get(features.len()).unwrap()
            ))?
        }

        Ok(features)
    }

    /// Updates the tree and all associated feature nodes with training data.
    ///
    /// For multi-value features, the same training input is applied to all corresponding
    /// child nodes. For example, training with `format=[banner, video]` creates/updates
    /// both the banner and video subtrees with the same input, while parent nodes
    /// (country, device) are visited only once, preventing double-counting.
    ///
    /// # Arguments
    /// * `features` - Feature vector in the same order as specified in constructor
    /// * `update` - Training data passed to each node's PredictionHandler
    ///
    /// # Errors
    /// Returns error if:
    /// - Features are not in the correct order
    /// - Feature keys don't match tree structure
    /// - Any feature has empty values
    ///
    /// # Example
    /// ```
    /// # use logictree::{LogicTree, Feature, PredictionHandler};
    /// # use std::sync::Mutex;
    /// # use serde::{Serialize, Deserialize};
    /// # #[derive(Serialize, Deserialize)] struct H { c: Mutex<u32> }
    /// # impl Clone for H { fn clone(&self) -> Self { H { c: Mutex::new(*self.c.lock().unwrap()) } } }
    /// # impl H { fn new() -> Self { H { c: Mutex::new(0) } } }
    /// # impl PredictionHandler<u32, u32> for H {
    /// #     fn train(&self, input: &u32) { *self.c.lock().unwrap() += input; }
    /// #     fn predict(&self) -> u32 { *self.c.lock().unwrap() }
    /// #     fn should_prune(&self) -> bool { false }
    /// #     fn new_instance(&self) -> Self { H::new() }
    /// #     fn resolve(&self, p: Vec<u32>) -> Option<u32> { Some(p.iter().sum()) }
    /// # }
    /// # let tree = LogicTree::new(vec!["country".into(), "format".into()], H::new());
    /// // Train with multi-value feature
    /// tree.train(vec![
    ///     Feature::string("country", "USA"),
    ///     Feature::multi_string("format", vec!["banner", "video"]),
    /// ], &10).unwrap();
    ///
    /// // Result: USA node sees 10, banner node sees 10, video node sees 10
    /// // (Not 20 for USA!)
    /// ```
    pub fn train(&self, features: Vec<Feature>, update: &I) -> Result<(), String> {
        self.validate(&features)?;

        self.root
            .get(&Self::root_value())
            .expect("should have root node!")
            .train(&features, update);

        Ok(())
    }

    pub fn train_map(
        &self,
        data: std::collections::HashMap<&str, Feature>,
        update: &I,
    ) -> Result<(), String> {
        self.train(self.extract(&data)?, update)
    }

    /// Make a prediction for the associated feature vector.
    ///
    /// Attempts to make a prediction at the deepest matching node first, walking up the tree
    /// to parent nodes and returning the first non-None result. For multi-value features,
    /// predictions from all matching child nodes are aggregated using the PredictionHandler's
    /// `resolve()` method. If `resolve()` returns None, falls back to the parent node's prediction.
    ///
    /// # Arguments
    /// * `features` - Feature vector in the same order as specified in constructor
    ///
    /// # Constraints
    /// - Features must be in the correct order
    /// - Contiguous values required (e.g., `[a,b]` valid, `[a,c]` invalid for tree `[a,b,c]`)
    /// - Full feature chain not required (partial paths acceptable)
    ///
    /// # Example
    /// ```
    /// # use logictree::{LogicTree, Feature, PredictionHandler};
    /// # use std::sync::Mutex;
    /// # use serde::{Serialize, Deserialize};
    /// # #[derive(Serialize, Deserialize)] struct H { c: Mutex<u32> }
    /// # impl Clone for H { fn clone(&self) -> Self { H { c: Mutex::new(*self.c.lock().unwrap()) } } }
    /// # impl H { fn new() -> Self { H { c: Mutex::new(0) } } }
    /// # impl PredictionHandler<u32, u32> for H {
    /// #     fn train(&self, input: &u32) { *self.c.lock().unwrap() += input; }
    /// #     fn predict(&self) -> u32 { *self.c.lock().unwrap() }
    /// #     fn should_prune(&self) -> bool { false }
    /// #     fn new_instance(&self) -> Self { H::new() }
    /// #     fn resolve(&self, p: Vec<u32>) -> Option<u32> { Some(p.iter().sum()) }
    /// # }
    /// # let tree = LogicTree::new(vec!["country".into(), "format".into()], H::new());
    /// # tree.train(vec![Feature::string("country", "USA"), Feature::string("format", "banner")], &100).unwrap();
    /// # tree.train(vec![Feature::string("country", "USA"), Feature::string("format", "video")], &200).unwrap();
    /// // Predict with multi-value feature (aggregates banner + video via resolve)
    /// let result = tree.predict(vec![
    ///     Feature::string("country", "USA"),
    ///     Feature::multi_string("format", vec!["banner", "video"]),
    /// ]).unwrap();
    ///
    /// // Returns: Sum of banner (100) and video (200) = 300
    /// ```
    pub fn predict(&self, features: Vec<Feature>) -> Result<Option<O>, String> {
        self.validate(&features)?;

        let res = self
            .root
            .get(&Self::root_value())
            .expect("should have root node!")
            .predict(&features);

        Ok(res)
    }

    /// Convenience method to predict based on map input of key -> feature values, and the
    /// tree handles organizing feature order. See `predict` in which the same validation rules apply.
    pub fn predict_map(
        &self,
        data: std::collections::HashMap<&str, Feature>,
    ) -> Result<Option<O>, String> {
        self.predict(self.extract(&data)?)
    }

    /// Perform pruning of the tree per the handlers should_prune logic.
    /// If this returns true, then the whole map has been pruned (empty)
    pub fn prune(&self) -> bool {
        self.root
            .get(&Self::root_value())
            .expect("should have root node!")
            .should_prune()
    }

    /// Return the count of nodes in this map. Optionally return
    /// the count of only leaf nodes.
    pub fn size(&self, leaf_only: bool) -> u32 {
        self.root
            .get(&Self::root_value())
            .expect("should have root node!")
            .size(leaf_only)
    }
}
