use logictree::{LogicTree, Feature, PredictionHandler};
use std::sync::Mutex;
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
struct TestHandler {
    count: Mutex<u32>,
}

impl TestHandler {
    fn new() -> Self {
        TestHandler {
            count: Mutex::new(0),
        }
    }
}

impl Clone for TestHandler {
    fn clone(&self) -> Self {
        TestHandler {
            count: Mutex::new(*self.count.lock().unwrap()),
        }
    }
}

impl PredictionHandler<u32, u32> for TestHandler {
    fn train(&self, input: &u32) {
        *self.count.lock().unwrap() += input;
    }

    fn predict(&self) -> u32 {
        *self.count.lock().unwrap()
    }

    fn should_prune(&self) -> bool {
        false
    }

    fn new_instance(&self) -> Self {
        TestHandler::new()
    }

    fn resolve(&self, predictions: Vec<(u32, usize)>) -> Option<u32> {
        println!("resolve called with {} predictions:", predictions.len());
        for (value, depth) in &predictions {
            println!("  value: {}, depth: {}", value, depth);
        }
        predictions.into_iter().map(|(v, _)| v).next()
    }
}

fn main() {
    let tree = LogicTree::new(
        vec!["country".to_string(), "device".to_string(), "format".to_string()],
        TestHandler::new(),
    );

    // Train at different depths
    tree.train(&vec![
        Feature::string("country", "usa"),
    ], &10).unwrap();

    tree.train(&vec![
        Feature::string("country", "usa"),
        Feature::string("device", "ios"),
    ], &20).unwrap();

    tree.train(&vec![
        Feature::string("country", "usa"),
        Feature::string("device", "ios"),
        Feature::string("format", "banner"),
    ], &30).unwrap();

    // Test predictions at different depths
    println!("\nPredicting at root (0 features):");
    let res = tree.predict(&vec![]).unwrap();
    if let Some(pred) = res {
        println!("Result: value={}, depth={}, full_depth={}",
                 pred.value, pred.depth, pred.full_depth);
    }

    println!("\nPredicting at depth 1 (country=usa):");
    let res = tree.predict(&vec![
        Feature::string("country", "usa"),
    ]).unwrap();
    if let Some(pred) = res {
        println!("Result: value={}, depth={}, full_depth={}",
                 pred.value, pred.depth, pred.full_depth);
    }

    println!("\nPredicting at depth 2 (country=usa, device=ios):");
    let res = tree.predict(&vec![
        Feature::string("country", "usa"),
        Feature::string("device", "ios"),
    ]).unwrap();
    if let Some(pred) = res {
        println!("Result: value={}, depth={}, full_depth={}",
                 pred.value, pred.depth, pred.full_depth);
    }

    println!("\nPredicting at depth 3 (full path):");
    let res = tree.predict(&vec![
        Feature::string("country", "usa"),
        Feature::string("device", "ios"),
        Feature::string("format", "banner"),
    ]).unwrap();
    if let Some(pred) = res {
        println!("Result: value={}, depth={}, full_depth={}",
                 pred.value, pred.depth, pred.full_depth);
    }

    println!("\nPredicting unknown path (falls back to parent):");
    let res = tree.predict(&vec![
        Feature::string("country", "usa"),
        Feature::string("device", "android"),  // not trained
    ]).unwrap();
    if let Some(pred) = res {
        println!("Result: value={}, depth={}, full_depth={}",
                 pred.value, pred.depth, pred.full_depth);
        println!("  (Note: fell back to parent at depth 1)");
    }
}