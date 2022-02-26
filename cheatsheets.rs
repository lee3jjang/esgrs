use std::fs::File;
use std::io::prelude::Read;
use yaml_rust::YamlLoader;
use std::thread;
use log::{debug, error, info, trace, warn};
use serde::Serialize;
use std::io;

// [dependencies]
// csv = "1.1"
// log = "0.4"
// yaml-rust = "0.4"
// env_logger = "0.9"
// simple_logger = "2.0"
// log4rs = "0.12.0"
// serde = { version = "1", features = ["derive"] }

fn main() {
    // 1. yaml 사용
    let mut file = File::open("test.yaml").unwrap();
    let mut config_str = String::new();
    file.read_to_string(&mut config_str).unwrap();
    let config = &YamlLoader::load_from_str(&config_str).unwrap()[0];
    println!("{:?}", config["baseYymm"].as_str().unwrap());
    println!("{:?}", config["learningRate"].as_f64());

    // 2. csv 읽기
    let file = File::open("uspop.csv").unwrap();
    let mut rdr = csv::Reader::from_reader(file);
    for result in rdr.records() {
        let record = &result.unwrap()[0];
        println!("{:?}", record);
    }

    // 3. 멀티쓰레딩 & 클로저
    let i = 4;
    thread::spawn(move || {
        println!("hi number {}", i);
    });

    // 4. 로깅
    log4rs::init_file("log4rs.yaml", Default::default()).unwrap();
    debug!("BOOM!!");
    error!("에러!!");

    // 5. csv 쓰기
    #[derive(Serialize)]
    struct Record<'a> {
        name: &'a str,
        place: &'a str,
        id: u64,
    }
    let mut wtr = csv::Writer::from_path("foo.csv").unwrap();
    let rec1 = Record { name: "Mark", place: "Melbourne", id: 56};
    let rec2 = Record { name: "Ashley", place: "Sydney", id: 64};
    let rec3 = Record { name: "Akshat", place: "Delhi", id: 98};
    wtr.serialize(rec1).unwrap();
    wtr.serialize(rec2).unwrap();
    wtr.serialize(rec3).unwrap();
    wtr.flush().unwrap();
}
    
