use std::marker::PhantomData;
use std::sync::Arc;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use serde::de::DeserializeOwned;
use crate::{Feature, Value};
use crate::handler::PredictionHandler;

#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "H: Serialize",
    deserialize = "H: DeserializeOwned"
))]
pub(crate) struct Node<I, O, H>
where H: PredictionHandler<I, O> {
    pub handler: Arc<H>,
    pub children: DashMap<Value, Node<I, O, H>>,
    _phantom: PhantomData<(I, O)>,
}

impl<I, O, H> Node<I, O, H>
where H: PredictionHandler<I, O> {
    pub fn new(handler: Arc<H>) -> Node<I, O, H> {
        Node { handler, children: DashMap::with_capacity(0), _phantom: PhantomData }
    }

    pub fn train(&self, stack: &[Feature], input: &I) -> () {
        self.handler.train(input);

        if stack.is_empty() {
            return;
        }

        let feat = &stack[0];

        self.children.entry(feat.value.clone())
            .or_insert_with(|| { Node::new(Arc::new(self.handler.new_instance())) })
            .train(&stack[1..], input);
    }

    pub fn predict(&self, stack: &[Feature]) -> Option<O> {
        if stack.is_empty() {
            return self.handler.predict()
        }

        let feat = &stack[0];

        match self.children.get(&feat.value) {
            Some(child) => child.predict(&stack[1..]),
            None => self.handler.predict()
        }
    }

    pub(crate) fn should_prune(&self) -> bool {
        // check if we can short circuit sweeping prune ourselves and all children
        // if the implementer needs they can cleanup handlers via drop. e.g. if all
        // children have been inactive then this node has been too - dont even
        // checking all the child nodes
        if self.handler.should_prune() {
            return true;
        }

        // we are an active leaf node
        if self.children.is_empty() {
            return false
        }

        // remove any individual nodes we should prune
        self.children.retain(|_, node| {
            !node.should_prune()
        });

        false
    }

    /// Returns the count of total nodes if leaf_only is false,
    /// otherwise the count of leafs
    pub(crate) fn size(&self, leaf_only: bool) -> u32 {
        if leaf_only && self.children.is_empty() {
            return 1;
        }

        let child_sz: u32 = self.children.iter()
            .map(|node| node.size(leaf_only))
            .sum();

        // return child size plus self if not leaf only
        child_sz + if !leaf_only { 1 } else { 0 }
    }

}