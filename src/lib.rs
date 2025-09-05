mod logictree;
mod feature;
mod node;
mod handler;
mod serialization;

pub use logictree::*;
pub use feature::*;
pub use handler::*;

#[cfg(test)]
mod tests {
    use std::fs;
    use std::sync::Mutex;
    use serde::{Deserialize, Serialize};
    use crate::{Feature, LogicTree, PredictionHandler};

    #[derive(Serialize, Deserialize)]
    struct TestHandler {
        pub total: Mutex<usize>,
    }

    impl Clone for TestHandler {
        fn clone(&self) -> Self {
            TestHandler {
                total: Mutex::new(*self.total.lock().unwrap())
            }
        }
    }

    impl TestHandler {
        pub fn new() -> TestHandler {
            TestHandler { total: Mutex::new(0) }
        }
    }

    // TODO handler examples with complex i/o types
    impl PredictionHandler<u32, u32> for TestHandler {
        fn train(&self, input: &u32) -> () {
            *self.total.lock().unwrap() += *input as usize;
        }

        fn predict(&self) -> Option<u32> {
            Some(*self.total.lock().unwrap() as u32)
        }

        fn should_prune(&self) -> bool {
            true
        }

        fn new_instance(&self) -> Self {
            TestHandler::new()
        }
    }

    fn test_map_assertions(tree: &LogicTree<u32, u32, TestHandler>) {
        let res = tree.predict(
            vec![
                Feature::string("one", "foo")
            ]
        );
        assert_eq!(res, Ok(Some(20)), "Foo parent prediction should equal sum of parent and children");

        let res = tree.predict(
            vec![
                Feature::string("one", "foo"),
                Feature::string("two", "bar")
            ]
        );
        assert_eq!(res, Ok(Some(10)), "Foo.bar prediction should equal node specific training value");

        let res = tree.predict(
            vec![
                Feature::string("one", "foo"),
                Feature::string("two", "dang")
            ]
        );
        assert_eq!(res, Ok(Some(20)), "Non existent foo.dang should return parent summary of foo");

        // prediction features in wrong order
        tree.predict(vec![
            Feature::string("two", "bar"),
            Feature::string("one", "foo"),
        ]).expect_err("Should fail on invalid feature order");

        let res = tree.size(true);
        assert_eq!(res, 3, "Should count leaf nodes correctly (bar baz bum)");

        let res = tree.size(false);
        assert_eq!(res, 5, "Should count total nodes correctly");

        assert_eq!(tree.prune(), true, "Pruning when all handlers should_prune should empty the tree");
    }

    #[test]
    fn it_works() {
        let tree = LogicTree::new(vec!["one".to_string(), "two".to_string()],
                                  TestHandler::new());

        tree.train(vec![
            Feature::string("one", "foo"),
            Feature::string("two", "bar")
        ], &10).unwrap();

        tree.train(vec![
            Feature::string("one", "foo"),
            Feature::string("two", "baz")
        ], &5).unwrap();

        tree.train(vec![
            Feature::string("one", "foo"),
            Feature::string("two", "bum")
        ], &5).unwrap();

        test_map_assertions(&tree);

        let file = "tree.bin";

        tree.save(file).expect("Should serialize tree to bin file");

        let tree: LogicTree<u32, u32, TestHandler> = LogicTree::load(file)
            .expect("Should have succeeded deserializing tree from bin file");

        test_map_assertions(&tree);

        fs::remove_file(file)
            .expect("Should have deleted the test bin file");
    }

}
