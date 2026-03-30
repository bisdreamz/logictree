use crate::handler::PredictionHandler;
use crate::{Feature, Value};
use hashbrown::HashMap;
use parking_lot::RwLock;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use smallvec::{SmallVec, smallvec};
use std::marker::PhantomData;
use std::sync::Arc;

#[derive(Serialize, Deserialize)]
#[serde(bound(serialize = "H: Serialize", deserialize = "H: DeserializeOwned"))]
pub(crate) struct Node<I, O, H>
where
    H: PredictionHandler<I, O>,
{
    pub handler: Arc<H>,
    pub children: RwLock<HashMap<Value, Arc<Node<I, O, H>>>>,
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
            _phantom: PhantomData,
        }
    }

    pub fn train(&self, stack: &[Feature], input: &I) {
        self.handler.train(input, stack.first());

        if stack.is_empty() {
            return;
        }

        let feat = &stack[0];

        let children_to_train: SmallVec<[Arc<Node<I, O, H>>; 1]> = {
            let mut children = self.children.write();
            feat.values
                .iter()
                .map(|value| {
                    children
                        .entry(value.clone())
                        .or_insert_with(|| {
                            Arc::new(Node::new(Arc::new(self.handler.new_instance())))
                        })
                        .clone()
                })
                .collect()
        };

        for child in children_to_train {
            child.train(&stack[1..], input);
        }
    }

    fn collect_direct_predictions(
        &self,
        values: &[Value],
        stack: &[Feature],
        depth: usize,
    ) -> SmallVec<[(O, usize); 1]> {
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

    pub fn predict(&self, stack: &[Feature], depth: usize) -> (Option<O>, usize) {
        if !stack.is_empty() {
            let feat = &stack[0];
            let next_stack = &stack[1..];
            let next_depth = depth + 1;

            let predictions = self.collect_direct_predictions(&feat.values, next_stack, next_depth);

            if !predictions.is_empty() {
                return match self.handler.resolve(predictions) {
                    Some((value, d)) => (Some(value), d),
                    None => (None, depth),
                };
            }
        }

        let value = self.handler.predict();
        match self.handler.resolve(smallvec![(value, depth)]) {
            Some((value, d)) => (Some(value), d),
            None => (None, depth),
        }
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
        children.retain(|_, node| !node.should_prune());

        false
    }

    pub(crate) fn size(&self, leaf_only: bool) -> u32 {
        let children = self.children.read();

        if leaf_only && children.is_empty() {
            return 1;
        }

        let child_sz: u32 = children.values().map(|node| node.size(leaf_only)).sum();

        child_sz + if !leaf_only { 1 } else { 0 }
    }
}
