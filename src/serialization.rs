use std::sync::Arc;
use serde::{Deserialize, Serialize};
use std::collections::HashMap as StdHashMap;
use papaya::HashMap;
use crate::{Value, PredictionHandler};
use crate::node::Node;

#[derive(Serialize, Deserialize)]
/// A serializable node state which adapts between std hashmap since
/// papaya doesnt impl serializable
pub(crate) struct SerializableNode<H> {
    handler: Arc<H>,
    children: StdHashMap<Value, SerializableNode<H>>,
}

impl<H> SerializableNode<H> {
    pub fn from_node<I, O>(node: &Node<I, O, H>) -> Self
    where
        H: PredictionHandler<I, O> + Clone
    {
        let guard = node.children.guard();
        let mut children = StdHashMap::new();

        for (key, child) in node.children.iter(&guard) {
            children.insert(key.clone(), SerializableNode::from_node(child));
        }

        SerializableNode {
            handler: node.handler.clone(),
            children,
        }
    }

    pub fn to_node<I, O>(self) -> Node<I, O, H>
    where
        H: PredictionHandler<I, O>
    {
        let children = HashMap::with_capacity(0);

        {
            let guard = children.guard();

            for (key, child) in self.children {
                children.insert(key, child.to_node(), &guard);
            }
        }

        Node::new_with_children(self.handler, children)
    }
}

#[derive(Serialize, Deserialize)]
pub(crate) struct SerializableTree<H> {
    pub features: Vec<String>,
    pub root: StdHashMap<Value, SerializableNode<H>>,
}

impl<H> SerializableTree<H> {
    pub fn save_to_file(&self, path: &str) -> Result<(), String>
    where
        H: Serialize
    {
        let config = bincode::config::standard();
        let bytes = bincode::serde::encode_to_vec(self, config)
            .map_err(|e| e.to_string())?;
        std::fs::write(path, bytes).map_err(|e| e.to_string())
    }

    pub fn load_from_file(path: &str) -> Result<Self, String>
    where
        H: for<'de> Deserialize<'de>
    {
        let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
        let config = bincode::config::standard();
        let (tree, _) = bincode::serde::decode_from_slice(&bytes, config)
            .map_err(|e| e.to_string())?;
        Ok(tree)
    }
}