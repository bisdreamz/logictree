use crate::handler::PredictionHandler;
use crate::{Feature, Value};
use hashbrown::{HashMap, HashSet};
use parking_lot::RwLock;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;
use std::sync::Arc;

const COMPOSITE_DELIMITER: &str = "\x00";

#[derive(Serialize, Deserialize)]
#[serde(bound(serialize = "H: Serialize", deserialize = "H: DeserializeOwned"))]
pub(crate) struct Node<I, O, H>
where
    H: PredictionHandler<I, O>,
{
    pub handler: Arc<H>,
    pub children: RwLock<HashMap<Value, Arc<Node<I, O, H>>>>,
    /// Maps individual feature values to composite keys containing them.
    /// Lazily populated only when multi-value features are used.
    composite_index: RwLock<Option<HashMap<Value, HashSet<Value>>>>,
    _phantom: PhantomData<(I, O)>,
}

impl<I, O, H> Node<I, O, H>
where
    H: PredictionHandler<I, O>,
{
    pub fn new(handler: Arc<H>) -> Node<I, O, H> {
        Node {
            handler,
            children: RwLock::new(HashMap::new()),
            composite_index: RwLock::new(None),
            _phantom: PhantomData,
        }
    }

    /// Creates a deterministic composite key from multiple values.
    /// Uses null byte delimiter to prevent collisions with user values.
    fn create_composite_key(values: &[Value]) -> Value {
        let mut sorted: Vec<String> = values
            .iter()
            .map(|v| match v {
                Value::String(s) => s.clone(),
                Value::Boolean(b) => b.to_string(),
                Value::I32(i) => i.to_string(),
                Value::I64(i) => i.to_string(),
                Value::U16(u) => u.to_string(),
                Value::U32(u) => u.to_string(),
                Value::U64(u) => u.to_string(),
            })
            .collect();
        sorted.sort();
        Value::String(sorted.join(COMPOSITE_DELIMITER))
    }

    pub fn train(&self, stack: &[Feature], input: &I) {
        self.handler.train(input, stack.first());

        if stack.is_empty() {
            return;
        }

        let feat = &stack[0];

        if feat.values.len() == 1 {
            let value = &feat.values[0];
            let child = {
                let mut children = self.children.write();
                children
                    .entry(value.clone())
                    .or_insert_with(|| Arc::new(Node::new(Arc::new(self.handler.new_instance()))))
                    .clone()
            };
            child.train(&stack[1..], input);
        } else {
            let composite_key = Self::create_composite_key(&feat.values);

            // Initialize composite_index lazily on first multi-value feature
            {
                let mut idx = self.composite_index.write();
                let map = idx.get_or_insert_with(HashMap::new);
                for value in &feat.values {
                    map.entry(value.clone())
                        .or_insert_with(HashSet::new)
                        .insert(composite_key.clone());
                }
            }

            let child = {
                let mut children = self.children.write();
                children
                    .entry(composite_key)
                    .or_insert_with(|| Arc::new(Node::new(Arc::new(self.handler.new_instance()))))
                    .clone()
            };
            child.train(&stack[1..], input);
        }
    }

    fn collect_direct_predictions(
        &self,
        values: &[Value],
        stack: &[Feature],
        depth: usize,
    ) -> Vec<(O, usize)> {
        let children = self.children.read();
        values
            .iter()
            .filter_map(|value| {
                children.get(value).map(|child| {
                    let (result, d) = child.predict(stack, depth);
                    result.map(|v| (v, d))
                })
            })
            .flatten()
            .collect()
    }

    fn collect_composite_predictions(
        &self,
        values: &[Value],
        stack: &[Feature],
        depth: usize,
    ) -> Vec<(O, usize)> {
        let idx = self.composite_index.read();
        let Some(ref index) = *idx else {
            return Vec::new();
        };

        let children = self.children.read();
        let mut seen = HashSet::new();

        values
            .iter()
            .filter_map(|value| index.get(value).cloned())
            .flatten()
            .filter(|key| seen.insert(key.clone()))
            .filter_map(|key| {
                children.get(&key).map(|child| {
                    let (result, d) = child.predict(stack, depth);
                    result.map(|v| (v, d))
                })
            })
            .flatten()
            .collect()
    }

    /// Predicts by collecting predictions from all matching paths using union semantics.
    pub fn predict(&self, stack: &[Feature], depth: usize) -> (Option<O>, usize) {
        if !stack.is_empty() {
            let feat = &stack[0];
            let next_stack = &stack[1..];
            let next_depth = depth + 1;

            let mut predictions =
                self.collect_direct_predictions(&feat.values, next_stack, next_depth);
            predictions.extend(self.collect_composite_predictions(
                &feat.values,
                next_stack,
                next_depth,
            ));

            if !predictions.is_empty() {
                let actual_depth = predictions.iter().map(|(_, d)| *d).max().unwrap_or(depth);
                return (self.handler.resolve(predictions), actual_depth);
            }
        }

        let value = self.handler.predict();
        let resolved = self.handler.resolve(vec![(value, depth)]);
        (resolved, depth)
    }

    pub(crate) fn should_prune(&self) -> bool {
        if self.handler.should_prune() {
            return true;
        }

        let children = self.children.read();
        if children.is_empty() {
            return false;
        }
        drop(children);

        let mut children = self.children.write();
        let pruned_keys: Vec<Value> = children
            .iter()
            .filter(|(_, node)| node.should_prune())
            .map(|(key, _)| key.clone())
            .collect();

        for key in &pruned_keys {
            children.remove(key);
        }
        drop(children);

        {
            let mut idx = self.composite_index.write();
            if let Some(ref mut index) = *idx {
                index.retain(|_, composites| {
                    for pruned_key in &pruned_keys {
                        composites.remove(pruned_key);
                    }
                    !composites.is_empty()
                });
            }
        }

        false
    }

    /// Returns the count of total nodes if leaf_only is false,
    /// otherwise the count of leafs
    pub(crate) fn size(&self, leaf_only: bool) -> u32 {
        let children = self.children.read();

        if leaf_only && children.is_empty() {
            return 1;
        }

        let child_sz: u32 = children.values().map(|node| node.size(leaf_only)).sum();

        // return child size plus self if not leaf only
        child_sz + if !leaf_only { 1 } else { 0 }
    }
}
