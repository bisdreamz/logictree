use crate::feature::Value;
use crate::node::Node;
use crate::{Feature, PredictionHandler, PredictionResult};
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

    /// Serialize this tree to a binary file at the given path using bincode.
    pub fn save(&self, path: &str) -> Result<(), String>
    where
        H: Serialize,
    {
        let config = bincode::config::standard();
        let bytes = bincode::serde::encode_to_vec(self, config).map_err(|e| e.to_string())?;
        std::fs::write(path, bytes).map_err(|e| e.to_string())
    }

    /// Deserialize a tree from a binary file previously written by [`save()`](Self::save).
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
    /// ```rust
    /// # use logictree::{LogicTree, PredictionHandler, Feature};
    /// # use smallvec::SmallVec;
    /// # use serde::{Serialize, Deserialize};
    /// # use std::sync::Mutex;
    /// # #[derive(Serialize, Deserialize)]
    /// # struct H { total: Mutex<u32> }
    /// # impl Clone for H { fn clone(&self) -> Self { H { total: Mutex::new(*self.total.lock().unwrap()) } } }
    /// # impl PredictionHandler<u32, u32> for H {
    /// #     fn train(&self, input: &u32, _: Option<&Feature>) { *self.total.lock().unwrap() += input; }
    /// #     fn predict(&self) -> u32 { *self.total.lock().unwrap() }
    /// #     fn should_prune(&self) -> bool { false }
    /// #     fn new_instance(&self) -> Self { H { total: Mutex::new(0) } }
    /// #     fn resolve(&self, p: SmallVec<[(u32, usize); 1]>) -> Option<(u32, usize)> { p.into_iter().max_by_key(|(v,_)| *v) }
    /// # }
    /// # let tree = LogicTree::new(vec!["country".into(), "format".into()], H { total: Mutex::new(0) });
    /// tree.train(&vec![
    ///     Feature::string("country", "USA"),
    ///     Feature::multi_string("format", vec!["banner", "video"]),
    /// ], &10).unwrap();
    /// // USA node sees 10, banner node sees 10, video node sees 10
    /// // (Not 20 for USA — parent trained once, children trained individually)
    /// ```
    pub fn train(&self, features: &Vec<Feature>, update: &I) -> Result<(), String> {
        self.validate(&features)?;

        self.root
            .get(&Self::root_value())
            .expect("should have root node!")
            .train(features, update);

        Ok(())
    }

    /// Convenience method to train from a map of key -> feature values.
    /// The tree handles organizing features into the correct order.
    /// See [`train()`](Self::train) for validation rules and error details.
    pub fn train_map(
        &self,
        data: std::collections::HashMap<&str, Feature>,
        update: &I,
    ) -> Result<(), String> {
        self.train(&self.extract(&data)?, update)
    }

    /// Make a prediction for the associated feature vector.
    ///
    /// Traverses the tree from root to the deepest matching node. At each level, the node's
    /// `resolve()` decides whether the child predictions are sufficient or should defer to
    /// the parent. For multi-value features, predictions from all matching children are
    /// passed to `resolve()` for aggregation/selection.
    ///
    /// # Arguments
    /// * `features` - Feature vector in the same order as specified in constructor
    ///
    /// # Returns
    /// * `Ok(Some(PredictionResult))` - Prediction found with value, depth, and full_depth flag
    /// * `Ok(None)` - No prediction available (e.g., all untrained paths or handlers returned None)
    /// * `Err(String)` - Invalid input (wrong feature order, non-contiguous features, etc.)
    ///
    /// # Constraints
    /// - Features must be in the correct order
    /// - Contiguous values required (e.g., `[a,b]` valid, `[a,c]` invalid for tree `[a,b,c]`)
    /// - Full feature chain not required (partial paths acceptable)
    ///
    /// # Example
    /// ```rust
    /// # use logictree::{LogicTree, PredictionHandler, Feature};
    /// # use smallvec::SmallVec;
    /// # use serde::{Serialize, Deserialize};
    /// # use std::sync::Mutex;
    /// # #[derive(Serialize, Deserialize)]
    /// # struct H { total: Mutex<u32> }
    /// # impl Clone for H { fn clone(&self) -> Self { H { total: Mutex::new(*self.total.lock().unwrap()) } } }
    /// # impl PredictionHandler<u32, u32> for H {
    /// #     fn train(&self, input: &u32, _: Option<&Feature>) { *self.total.lock().unwrap() += input; }
    /// #     fn predict(&self) -> u32 { *self.total.lock().unwrap() }
    /// #     fn should_prune(&self) -> bool { false }
    /// #     fn new_instance(&self) -> Self { H { total: Mutex::new(0) } }
    /// #     fn resolve(&self, p: SmallVec<[(u32, usize); 1]>) -> Option<(u32, usize)> { p.into_iter().max_by_key(|(v,_)| *v) }
    /// # }
    /// # let tree = LogicTree::new(vec!["country".into(), "format".into()], H { total: Mutex::new(0) });
    /// # tree.train(&vec![Feature::string("country", "USA"), Feature::string("format", "banner")], &10).unwrap();
    /// let result = tree.predict(&vec![
    ///     Feature::string("country", "USA"),
    ///     Feature::multi_string("format", vec!["banner", "video"]),
    /// ]).unwrap();
    ///
    /// if let Some(pred) = result {
    ///     println!("value={} depth={} full={}", pred.value, pred.depth, pred.full_depth);
    /// }
    /// ```
    pub fn predict(&self, features: &Vec<Feature>) -> Result<Option<PredictionResult<O>>, String> {
        self.validate(&features)?;

        let (value, depth) = self
            .root
            .get(&Self::root_value())
            .expect("should have root node!")
            .predict(features, 0);

        Ok(value.map(|v| PredictionResult {
            value: v,
            depth,
            full_depth: depth == self.features.len(),
        }))
    }

    /// Convenience method to predict based on map input of key -> feature values, and the
    /// tree handles organizing feature order. Returns the same `PredictionResult` as `predict()`.
    /// See `predict` for validation rules and return value details.
    pub fn predict_map(
        &self,
        data: std::collections::HashMap<&str, Feature>,
    ) -> Result<Option<PredictionResult<O>>, String> {
        self.predict(&self.extract(&data)?)
    }

    /// Perform pruning of the tree per each handler's `should_prune()` logic.
    /// Returns true if the root node itself was pruned (entire tree empty).
    pub fn prune(&self) -> bool {
        self.root
            .get(&Self::root_value())
            .expect("should have root node!")
            .should_prune()
    }

    /// Access the handler at a specific node in the tree.
    ///
    /// Traverses the feature path to reach the target node and calls `f`
    /// with a reference to its handler. Creates the target node if it
    /// doesn't exist (with `new_instance()` handler). Returns `Err` if
    /// any intermediate (parent) node along the path is missing — callers
    /// must create nodes in depth order (root first, then deeper).
    ///
    /// Use an empty feature slice to access the root handler.
    pub fn with_node<F>(&self, features: &[Feature], f: &F) -> Result<(), String>
    where
        F: Fn(&H),
    {
        self.validate(features)?;

        self.root
            .get(&Self::root_value())
            .expect("should have root node!")
            .with_node(features, f, 0)
    }

    /// Walk every node in the tree, calling `f` with the accumulated
    /// feature value path and a reference to the handler. Root receives
    /// an empty path, depth-1 nodes receive `[value]`, etc.
    ///
    /// Path is accumulated on the stack during traversal — zero
    /// per-node storage overhead.
    pub fn for_each<F>(&self, f: &F)
    where
        F: Fn(&[Value], &H),
    {
        self.root
            .get(&Self::root_value())
            .expect("should have root node!")
            .for_each(&mut Vec::new(), f);
    }

    /// Return the count of nodes in this tree.
    /// If `leaf_only` is true, counts only leaf nodes (no children).
    pub fn size(&self, leaf_only: bool) -> u32 {
        self.root
            .get(&Self::root_value())
            .expect("should have root node!")
            .size(leaf_only)
    }
}
