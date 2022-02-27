use std::thread;
use std::sync::mpsc;

use esgrs::hw::model::learning;
use esgrs::ts::{TermStructure, smith_wilson_ytm};

use std::fs::File;
use std::io::prelude::Read;
use yaml_rust::YamlLoader;
use log::{debug, error, info, trace, warn};


fn main() {

    // 1. Config & Logging
    let mut config_file = File::open("config.yaml").unwrap();
    let mut config_str = String::new();
    config_file.read_to_string(&mut config_str).unwrap();
    let config = &YamlLoader::load_from_str(&config_str).unwrap()[0];
    let lr = config["learningRate"].as_f64().unwrap();
    let tol: f64 = config["tolerance"].as_f64().unwrap();
    log4rs::init_file("log4rs.yaml", Default::default()).unwrap();

    let base_yymm = "202101";
    
    // 2. Yield
    let yield_file = File::open("yield.csv").unwrap();
    let ytm = get_ytm(base_yymm, yield_file);
    let tenor = vec![0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0];
    let ytm = Vec::from(ytm);
    let ltfr = ytm[ytm.len()-1];
    let alpha0 = 0.1;
    let freq = 2.0;

    let ts = smith_wilson_ytm(ltfr, alpha0, tenor, ytm, freq);

    // 3. Swaption
    let swaption_file = File::open("swaption.csv").unwrap();
    let swaption_vol_mkt = get_swaption_vol(base_yymm, swaption_file);

    // 4. Calibraiton
    let (tx, rx) = mpsc::channel();

    let mut handles = vec![];

    let handle = thread::spawn(move || {
        let ts = ts;
        let swaption_vol_mkt = swaption_vol_mkt;
        let p0 = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01];
        let lr = lr;
        let tol = tol;
        let p = learning(ts, swaption_vol_mkt, p0, lr, tol);
        tx.send(p).unwrap();
    });
    handles.push(handle);

    for handle in handles {
        handle.join().unwrap();  
    }

    for received in rx {
        println!("{:?}", received);
    }

}


fn get_swaption_vol(base_yymm: &str, file: File) -> [f64; 36] {
    let mut rdr = csv::Reader::from_reader(file);
    let mut swaption_vol_mkt = [0.0; 36];
    for result in rdr.records() {
        let record = &result.unwrap();
        if &record[0] == base_yymm {
            for i in 0..36 {
                swaption_vol_mkt[i] = record[i+1].parse::<f64>().unwrap();
            }
        }
    }
    swaption_vol_mkt
}

fn get_ytm(base_yymm: &str, file: File) -> Vec<f64> {
    let mut rdr = csv::Reader::from_reader(file);
    let mut ytm = Vec::new();
    for result in rdr.records() {
        let record = &result.unwrap();
        let n = record.len();
        if &record[0] == base_yymm {
            for i in 1..n {
                ytm.push(record[i].parse::<f64>().unwrap_or_default());
            }
        }
    }
    ytm
}