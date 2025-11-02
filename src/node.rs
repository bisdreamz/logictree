use crate::handler::PredictionHandler;
use crate::{Feature, Value};
use dashmap::DashMap;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
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
    pub children: DashMap<Value, Node<I, O, H>>,
    /// Maps individual feature values to composite keys containing them.
    /// Used for efficient lookup of multi-value feature combinations during prediction
    composite_index: DashMap<Value, HashSet<Value>>,
    _phantom: PhantomData<(I, O)>,
}

impl<I, O, H> Node<I, O, H>
where
    H: PredictionHandler<I, O>,
{
    pub fn new(handler: Arc<H>) -> Node<I, O, H> {
        Node {
            handler,
            children: DashMap::with_capacity(0),
            composite_index: DashMap::new(),
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
                Value::U32(u) => u.to_string(),
                Value::U64(u) => u.to_string(),
            })
            .collect();
        sorted.sort();
        Value::String(sorted.join(COMPOSITE_DELIMITER))
    }

    pub fn train(&self, stack: &[Feature], input: &I) {
        self.handler.train(input);

        if stack.is_empty() {
            return;
        }

        let feat = &stack[0];

        if feat.values.len() == 1 {
            let value = &feat.values[0];
            self.children
                .entry(value.clone())
                .or_insert_with(|| Node::new(Arc::new(self.handler.new_instance())))
                .train(&stack[1..], input);
        } else {
            let composite_key = Self::create_composite_key(&feat.values);

            for value in &feat.values {
                self.composite_index
                    .entry(value.clone())
                    .or_insert_with(HashSet::new)
                    .insert(composite_key.clone());
            }

            self.children
                .entry(composite_key)
                .or_insert_with(|| Node::new(Arc::new(self.handler.new_instance())))
                .train(&stack[1..], input);
        }
    }

    fn collect_direct_predictions(&self, values: &[Value], stack: &[Feature]) -> Vec<O> {
        values
            .iter()
            .filter_map(|value| {
                self.children
                    .get(value)
                    .and_then(|child| child.predict(stack))
            })
            .collect()
    }

    fn collect_composite_predictions(&self, values: &[Value], stack: &[Feature]) -> Vec<O> {
        let mut seen = HashSet::new();
        let mut predictions = Vec::new();

        for value in values {
            let composites = match self.composite_index.get(value) {
                Some(composites) => composites,
                None => continue,
            };

            for composite_key in composites.iter() {
                if seen.insert(composite_key.clone()) {
                    if let Some(prediction) = self
                        .children
                        .get(composite_key)
                        .and_then(|child| child.predict(stack))
                    {
                        predictions.push(prediction);
                    }
                }
            }
        }

        predictions
    }

    fn aggregate_predictions(&self, predictions: Vec<O>) -> Option<O> {
        match predictions.len() {
            0 => Some(self.handler.predict()), // no children, as the parent we will predict
            _ => self.handler.resolve(predictions), // resolve and return aggregate predictions
        }
    }

    /// Predicts by collecting predictions from all matching paths using union semantics.
    ///
    /// For multi-value features, this collects predictions from:
    /// - Direct children matching query values (e.g., "banner", "video")
    /// - Composite children containing ANY query value (e.g., "banner|video", "banner|native")
    ///
    /// All collected predictions are aggregated via `resolve()`. This union approach ensures
    /// that queries like `[banner]` include both banner-only stats AND co-occurrence stats
    /// from composites like "banner|video".
    ///
    /// # Example
    /// ```text
    /// Query: [banner]
    /// Matches:
    ///   - Direct child "banner" (banner-only requests)
    ///   - Composite "banner|video" (banner+video requests)
    ///   - Composite "banner|native" (banner+native requests)
    ///
    /// Result: resolve([banner, banner|video, banner|native])
    /// ```
    pub fn predict(&self, stack: &[Feature]) -> Option<O> {
        if stack.is_empty() {
            return Some(self.handler.predict());
        }

        let feat = &stack[0];
        let next_stack = &stack[1..];

        let mut predictions = self.collect_direct_predictions(&feat.values, next_stack);
        predictions.extend(self.collect_composite_predictions(&feat.values, next_stack));

        self.aggregate_predictions(predictions)
    }

    pub(crate) fn should_prune(&self) -> bool {
        if self.handler.should_prune() {
            return true;
        }

        if self.children.is_empty() {
            return false;
        }

        let mut pruned_keys = Vec::new();
        self.children.retain(|key, node| {
            if node.should_prune() {
                pruned_keys.push(key.clone());
                false
            } else {
                true
            }
        });

        for pruned_key in pruned_keys {
            self.composite_index.retain(|_, composites| {
                composites.remove(&pruned_key);
                !composites.is_empty()
            });
        }

        false
    }

    /// Returns the count of total nodes if leaf_only is false,
    /// otherwise the count of leafs
    pub(crate) fn size(&self, leaf_only: bool) -> u32 {
        if leaf_only && self.children.is_empty() {
            return 1;
        }

        let child_sz: u32 = self.children.iter().map(|node| node.size(leaf_only)).sum();

        // return child size plus self if not leaf only
        child_sz + if !leaf_only { 1 } else { 0 }
    }
}
