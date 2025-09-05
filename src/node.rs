use std::sync::Arc;
use papaya::HashMap;
use crate::{Feature, Value};
use crate::handler::PredictionHandler;

pub(crate) struct Node<I, O, H>
where H: PredictionHandler<I, O> {
    pub handler: Arc<H>,
    pub children: HashMap<Value, Node<I, O, H>>
}

impl<I, O, H> Node<I, O, H>
where H: PredictionHandler<I, O> {
    pub fn new(handler: Arc<H>) -> Node<I, O, H> {
        Node { handler, children: HashMap::with_capacity(0) }
    }

    pub(crate) fn new_with_children(handler: Arc<H>, children: HashMap<Value, Node<I, O, H>>) -> Self {
        Node { handler, children }
    }

    pub fn train(&self, stack: &[Feature], input: &I) -> () {
        self.handler.train(input);

        if stack.is_empty() {
            return;
        }

        let feat = &stack[0];

        let child_map = self.children.pin();
        child_map.get_or_insert_with(feat.value.clone(),
                                     || Node::new(Arc::new(self.handler.new_instance()))
        ).train(&stack[1..], input);
    }

    pub fn predict(&self, stack: &[Feature]) -> Option<O> {
        if stack.is_empty() {
            return self.handler.predict()
        }

        let feat = &stack[0];

        let child_map = self.children.pin();
        match child_map.get(&feat.value) {
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
        self.children.pin().retain(|_, node| {
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

        let child_sz: u32 = self.children.pin().
            values().into_iter()
            .map(|node| node.size(leaf_only))
            .sum();

        // return child size plus self if not leaf only
        child_sz + if !leaf_only { 1 } else { 0 }
    }

}