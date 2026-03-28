mod common;

use common::MaxValueHandler;
use logictree::{Feature, LogicTree};
use std::cell::Cell;

fn make_tree() -> LogicTree<u32, u32, MaxValueHandler> {
    LogicTree::new(
        vec!["country".into(), "city".into()],
        MaxValueHandler::new(),
    )
}

#[test]
fn accesses_root() {
    let tree = make_tree();
    tree.train(
        &vec![
            Feature::string("country", "US"),
            Feature::string("city", "Austin"),
        ],
        &10,
    )
    .unwrap();

    let value = Cell::new(0u32);
    tree.with_node(&[], &|h| {
        value.set(h.get());
    })
    .unwrap();

    assert_eq!(value.get(), 10);
}

#[test]
fn accesses_intermediate() {
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

    let value = Cell::new(0u32);
    tree.with_node(&[Feature::string("country", "US")], &|h| {
        value.set(h.get());
    })
    .unwrap();

    // US node sees both Austin(10) + Dallas(5)
    assert_eq!(value.get(), 15);
}

#[test]
fn accesses_leaf() {
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

    let austin = Cell::new(0u32);
    tree.with_node(
        &[
            Feature::string("country", "US"),
            Feature::string("city", "Austin"),
        ],
        &|h| {
            austin.set(h.get());
        },
    )
    .unwrap();

    let dallas = Cell::new(0u32);
    tree.with_node(
        &[
            Feature::string("country", "US"),
            Feature::string("city", "Dallas"),
        ],
        &|h| {
            dallas.set(h.get());
        },
    )
    .unwrap();

    assert_eq!(austin.get(), 10);
    assert_eq!(dallas.get(), 5);
}

#[test]
fn sets_handler_state() {
    let tree = make_tree();
    tree.train(
        &vec![
            Feature::string("country", "US"),
            Feature::string("city", "Austin"),
        ],
        &10,
    )
    .unwrap();

    // Set state on the leaf via with_node
    tree.with_node(
        &[
            Feature::string("country", "US"),
            Feature::string("city", "Austin"),
        ],
        &|h| {
            h.set(999);
        },
    )
    .unwrap();

    // predict() should reflect new value
    let result = tree
        .predict(&vec![
            Feature::string("country", "US"),
            Feature::string("city", "Austin"),
        ])
        .unwrap()
        .unwrap();

    assert_eq!(result.value, 999);
}

#[test]
fn does_not_affect_siblings() {
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

    tree.with_node(
        &[
            Feature::string("country", "US"),
            Feature::string("city", "Austin"),
        ],
        &|h| {
            h.set(999);
        },
    )
    .unwrap();

    let dallas = Cell::new(0u32);
    tree.with_node(
        &[
            Feature::string("country", "US"),
            Feature::string("city", "Dallas"),
        ],
        &|h| {
            dallas.set(h.get());
        },
    )
    .unwrap();

    assert_eq!(dallas.get(), 5);
}

#[test]
fn does_not_affect_parent() {
    let tree = make_tree();
    tree.train(
        &vec![
            Feature::string("country", "US"),
            Feature::string("city", "Austin"),
        ],
        &10,
    )
    .unwrap();

    tree.with_node(
        &[
            Feature::string("country", "US"),
            Feature::string("city", "Austin"),
        ],
        &|h| {
            h.set(999);
        },
    )
    .unwrap();

    let parent = Cell::new(0u32);
    tree.with_node(&[Feature::string("country", "US")], &|h| {
        parent.set(h.get());
    })
    .unwrap();

    assert_eq!(parent.get(), 10);
}

#[test]
fn creates_target_if_missing() {
    let tree = make_tree();
    // Train US/Austin so country=US parent exists
    tree.train(
        &vec![
            Feature::string("country", "US"),
            Feature::string("city", "Austin"),
        ],
        &10,
    )
    .unwrap();

    // Dallas doesn't exist yet — with_node should create it
    tree.with_node(
        &[
            Feature::string("country", "US"),
            Feature::string("city", "Dallas"),
        ],
        &|h| {
            h.set(42);
        },
    )
    .unwrap();

    let result = tree
        .predict(&vec![
            Feature::string("country", "US"),
            Feature::string("city", "Dallas"),
        ])
        .unwrap()
        .unwrap();

    assert_eq!(result.value, 42);
}

#[test]
fn errors_if_parent_missing() {
    let tree = make_tree();
    // Only root exists — country=UK parent is missing

    let result = tree.with_node(
        &[
            Feature::string("country", "UK"),
            Feature::string("city", "London"),
        ],
        &|_| {},
    );

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        err.contains("parent node missing"),
        "expected parent missing error, got: {err}"
    );
}

#[test]
fn root_accessible_on_empty_tree() {
    let tree = make_tree();

    let value = Cell::new(0u32);
    tree.with_node(&[], &|h| {
        value.set(h.get());
    })
    .unwrap();

    assert_eq!(value.get(), 0);
}
