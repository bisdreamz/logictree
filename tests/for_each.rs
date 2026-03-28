mod common;

use common::MaxValueHandler;
use logictree::{Feature, LogicTree, Value};
use std::cell::RefCell;

fn make_tree() -> LogicTree<u32, u32, MaxValueHandler> {
    LogicTree::new(
        vec!["country".into(), "city".into()],
        MaxValueHandler::new(),
    )
}

#[test]
fn visits_root_on_empty_tree() {
    let tree = make_tree();
    let visited = RefCell::new(Vec::new());

    tree.for_each(&|path, handler| {
        visited.borrow_mut().push((path.to_vec(), handler.get()));
    });

    let v = visited.borrow();
    assert_eq!(v.len(), 1, "should visit root only");
    assert!(v[0].0.is_empty(), "root path should be empty");
    assert_eq!(v[0].1, 0, "root handler should be untrained");
}

#[test]
fn visits_all_nodes() {
    let tree = make_tree();
    tree.train(
        &vec![
            Feature::string("country", "US"),
            Feature::string("city", "Austin"),
        ],
        &10,
    )
    .unwrap();
    tree.train(
        &vec![
            Feature::string("country", "US"),
            Feature::string("city", "Dallas"),
        ],
        &5,
    )
    .unwrap();

    let visited = RefCell::new(Vec::new());
    tree.for_each(&|path, handler| {
        visited.borrow_mut().push((path.to_vec(), handler.get()));
    });

    let v = visited.borrow();
    // root + US + Austin + Dallas = 4 nodes
    assert_eq!(v.len(), 4, "should visit 4 nodes, got {}", v.len());
}

#[test]
fn root_has_empty_path() {
    let tree = make_tree();
    tree.train(
        &vec![
            Feature::string("country", "US"),
            Feature::string("city", "Austin"),
        ],
        &10,
    )
    .unwrap();

    let root_paths = RefCell::new(Vec::new());
    tree.for_each(&|path, _| {
        if path.is_empty() {
            root_paths.borrow_mut().push(true);
        }
    });

    assert_eq!(root_paths.borrow().len(), 1, "exactly one root node");
}

#[test]
fn intermediate_node_has_correct_path() {
    let tree = make_tree();
    tree.train(
        &vec![
            Feature::string("country", "US"),
            Feature::string("city", "Austin"),
        ],
        &10,
    )
    .unwrap();

    let found = RefCell::new(false);
    tree.for_each(&|path, handler| {
        if path.len() == 1 && path[0] == Value::String("US".into()) {
            assert_eq!(handler.get(), 10, "US node should have 10");
            *found.borrow_mut() = true;
        }
    });

    assert!(*found.borrow(), "should find US intermediate node");
}

#[test]
fn leaf_has_full_path() {
    let tree = make_tree();
    tree.train(
        &vec![
            Feature::string("country", "US"),
            Feature::string("city", "Austin"),
        ],
        &10,
    )
    .unwrap();

    let found = RefCell::new(false);
    tree.for_each(&|path, handler| {
        if path.len() == 2
            && path[0] == Value::String("US".into())
            && path[1] == Value::String("Austin".into())
        {
            assert_eq!(handler.get(), 10, "leaf should have 10");
            *found.borrow_mut() = true;
        }
    });

    assert!(*found.borrow(), "should find US/Austin leaf node");
}

#[test]
fn handler_values_match_training() {
    let tree = make_tree();
    tree.train(
        &vec![
            Feature::string("country", "US"),
            Feature::string("city", "Austin"),
        ],
        &10,
    )
    .unwrap();
    tree.train(
        &vec![
            Feature::string("country", "US"),
            Feature::string("city", "Dallas"),
        ],
        &5,
    )
    .unwrap();
    tree.train(
        &vec![
            Feature::string("country", "UK"),
            Feature::string("city", "London"),
        ],
        &3,
    )
    .unwrap();

    let visited = RefCell::new(std::collections::HashMap::new());
    tree.for_each(&|path, handler| {
        let key: Vec<String> = path
            .iter()
            .map(|v| match v {
                Value::String(s) => s.clone(),
                _ => format!("{v:?}"),
            })
            .collect();
        visited.borrow_mut().insert(key, handler.get());
    });

    let v = visited.borrow();
    assert_eq!(v[&vec![] as &Vec<String>], 18, "root = 10+5+3");
    assert_eq!(v[&vec!["US".to_string()]], 15, "US = 10+5");
    assert_eq!(v[&vec!["UK".to_string()]], 3, "UK = 3");
    assert_eq!(
        v[&vec!["US".to_string(), "Austin".to_string()]],
        10,
        "Austin = 10"
    );
    assert_eq!(
        v[&vec!["US".to_string(), "Dallas".to_string()]],
        5,
        "Dallas = 5"
    );
    assert_eq!(
        v[&vec!["UK".to_string(), "London".to_string()]],
        3,
        "London = 3"
    );
}

#[test]
fn mutating_handler_via_for_each() {
    let tree = make_tree();
    tree.train(
        &vec![
            Feature::string("country", "US"),
            Feature::string("city", "Austin"),
        ],
        &10,
    )
    .unwrap();

    // Set all handlers to 999
    tree.for_each(&|_, handler| {
        handler.set(999);
    });

    // Verify via predict
    let result = tree
        .predict(&vec![
            Feature::string("country", "US"),
            Feature::string("city", "Austin"),
        ])
        .unwrap()
        .unwrap();

    assert_eq!(result.value, 999);
}
