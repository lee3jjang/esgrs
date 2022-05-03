use std::thread;

use esgrs::ts::smith_wilson_ytm;

use std::fs::File;
use std::io::prelude::Read;
use yaml_rust::YamlLoader;
use log::info;

fn main() {

    // 0. Config & Logging
    let mut config_file = File::open("config.yaml").unwrap();
    let mut config_str = String::new();
    config_file.read_to_string(&mut config_str).unwrap();
    let config = &YamlLoader::load_from_str(&config_str).unwrap()[0];
    let lr: f64 = config["learningRate"].as_f64().unwrap();
    let tol: f64 = config["tolerance"].as_f64().unwrap();
    let typ: i64 = config["type"].as_i64().unwrap();
    let swaption_vol_var: f64 = config["swaptionVolVar"].as_f64().unwrap();
    let int_rate_var: f64 = config["intRateVar"].as_f64().unwrap();
    let base_yymm_all: Vec<String> = config["baseYymm"].as_vec().unwrap().iter().map(|x| String::from(x.as_str().unwrap())).collect();
    let p0_raw: Vec<f64> = config["p0"].as_vec().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
    let mut p0_typ1: [f64; 7] = [0.0; 7];
    let mut p0_typ2: [f64; 8] = [0.0; 8];

    // 0-2. Check Parameter Length
    let n_p0: i64 = p0_raw.iter().len().try_into().unwrap();
    if !((n_p0 == 7 && typ == 1) || (n_p0 == 8 && typ == 2)) { panic!("p0 length error!"); }
    if typ == 1 {
        for i in 0..7 { p0_typ1[i] = p0_raw[i]; }
    } else if typ == 2 {
        for i in 0..8 { p0_typ2[i] = p0_raw[i]; }
    } else {
        panic!("type error!");
    }
    
    log4rs::init_file("log4rs.yaml", Default::default()).unwrap();
    
    let mut handles = vec![];
    for base_yymm in base_yymm_all {

        let handle = thread::spawn(move || {

            // 1. Load File
            let yield_file = File::open("data/yield.csv").unwrap();
            let swaption_file = File::open("data/swaption.csv").unwrap();

            // 2. YTM, Tenor
            let tenor = if base_yymm.as_str() >= "201910" {
                vec![0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0]
            } else if base_yymm.as_str() >= "201610" {
                vec![0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0, 20.0, 30.0, 50.0]
            } else if base_yymm.as_str() >= "201310" {
                vec![0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0, 20.0, 30.0]
            } else if base_yymm.as_str() >= "201209" {
                vec![0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0, 10.0, 20.0, 30.0]
            } else if base_yymm.as_str() >= "200709" {
                vec![0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0, 10.0, 20.0]
            } else {
                panic!("base_yymm error!");
            };
            let ytm: Vec<f64> = get_ytm(&base_yymm, yield_file);
            let ytm: Vec<f64> = Vec::from(ytm).iter().filter(|x| x > &&0.0).map(|x| {x + int_rate_var}).collect();
            let ltfr = ytm[ytm.len()-1];
            let alpha0 = 0.1;
            let freq = 2.0;
            
            // 3. Swaption
            let mut swaption_vol_mkt = get_swaption_vol(&base_yymm, swaption_file);
            for i in 0..swaption_vol_mkt.len() {
                swaption_vol_mkt[i] += swaption_vol_var;
            }
            
            // 4. Calibraiton
            let ts = smith_wilson_ytm(ltfr, alpha0, tenor, ytm, freq);
            let lr = lr;
            let tol = tol;
            
            if typ == 1 {
                use esgrs::hw::model_type1::learning;
                let (p, err, grad_norm) = learning(ts, swaption_vol_mkt, p0_typ1, lr, tol);
                info!("(Type 1) {:?}, {:?}, {:?}, {:?}, {:?}, {:?}, {:?}, {:?}, {:?}, {:?}, {:?}, {:?}, {:?}, {:?}", base_yymm, int_rate_var, swaption_vol_var, p[0], p[1], p[2], p[3], p[4], p[5], p[6], err, grad_norm, lr, tol);
            } else if typ == 2 {
                use esgrs::hw::model_type2::learning;
                let (p, err, grad_norm) = learning(ts, swaption_vol_mkt, p0_typ2, lr, tol);
                info!("(Type 2) {:?}, {:?}, {:?}, {:?}, {:?}, {:?}, {:?}, {:?}, {:?}, {:?}, {:?}, {:?}, {:?}, {:?}, {:?}", base_yymm, int_rate_var, swaption_vol_var, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], err, grad_norm, lr, tol);
            } else {
                panic!("type error!");
            };
            
        });

        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();  
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