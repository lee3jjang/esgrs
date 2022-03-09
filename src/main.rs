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
    let lr = config["learningRate"].as_f64().unwrap();
    let tol: f64 = config["tolerance"].as_f64().unwrap();
    let typ: i64 = config["type"].as_i64().unwrap();
    log4rs::init_file("log4rs.yaml", Default::default()).unwrap();
    
    let mut handles = vec![];
    for base_yymm in [
        "201201", "201202", "201203", "201204", "201205", "201206", "201207", "201208", "201209", "201210", "201211", "201212",
        "201301", "201302", "201303", "201304", "201305", "201306", "201307", "201308", "201309", "201310", "201311", "201312",
        "201401", "201402", "201403", "201404", "201405", "201406", "201407", "201408", "201409", "201410", "201411", "201412",
        "201501", "201502", "201503", "201504", "201505", "201506", "201507", "201508", "201509", "201510", "201511", "201512",
        "201601", "201602", "201603", "201604", "201605", "201606", "201607", "201608", "201609", "201610", "201611", "201612",
        "201701", "201702", "201703", "201704", "201705", "201706", "201707", "201708", "201709", "201710", "201711", "201712",
        "201801", "201802", "201803", "201804", "201805", "201806", "201807", "201808", "201809", "201810", "201811", "201812",
        "201901", "201902", "201903", "201904", "201905", "201906", "201907", "201908", "201909", "201910", "201911", "201912",
        "202001", "202002", "202003", "202004", "202005", "202006", "202007", "202008", "202009", "202010", "202011", "202012",
        "202101", "202102", "202103", "202104", "202105", "202106", "202107", "202108", "202109", "202110", "202111", "202112",
    ].iter() {

        let handle = thread::spawn(move || {

            // 1. Load File
            let yield_file = File::open("yield.csv").unwrap();
            let swaption_file = File::open("swaption.csv").unwrap();

            // 2. YTM, Tenor
            let tenor = if base_yymm.to_owned() >= "201910" {
                vec![0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0]
            } else if base_yymm.to_owned() >= "201610" {
                vec![0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0, 20.0, 30.0, 50.0]
            } else if base_yymm.to_owned() >= "201310" {
                vec![0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0, 20.0, 30.0]
            } else if base_yymm.to_owned() >= "201209" {
                vec![0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0, 10.0, 20.0, 30.0]
            } else if base_yymm.to_owned() >= "201201" {
                vec![0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0, 10.0, 20.0]
            } else {
                panic!("base_yymm error!");
            };
            let ytm: Vec<f64> = get_ytm(base_yymm, yield_file);
            let ytm: Vec<f64> = Vec::from(ytm).iter().filter(|x| x > &&0.0).map(|x| {x + 0.0000}).collect();
            let ltfr = ytm[ytm.len()-1];
            let alpha0 = 0.1;
            let freq = 2.0;
            
            // 3. Swaption
            let mut swaption_vol_mkt = get_swaption_vol(base_yymm, swaption_file);
            for i in 0..swaption_vol_mkt.len() {
                swaption_vol_mkt[i] += 0.0000;
            }
            
            // 4. Calibraiton
            let ts = smith_wilson_ytm(ltfr, alpha0, tenor, ytm, freq);
            let lr = lr;
            let tol = tol;
            if typ == 1 {
                use esgrs::hw::model_type1::learning;
                let p0 = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01];
                let (p, err, grad_norm) = learning(ts, swaption_vol_mkt, p0, lr, tol);
                info!("(Type 1) {:?}, {:?}, {:?}, {:?}, {:?}, {:?}, {:?}, {:?}, {:?}, {:?}, {:?}, {:?}", base_yymm, p[0], p[1], p[2], p[3], p[4], p[5], p[6], err, grad_norm, lr, tol);
            } else if typ == 2 {
                use esgrs::hw::model_type2::learning;
                let p0 = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01];
                let (p, err, grad_norm) = learning(ts, swaption_vol_mkt, p0, lr, tol);
                info!("(Type 2) {:?}, {:?}, {:?}, {:?}, {:?}, {:?}, {:?}, {:?}, {:?}, {:?}, {:?}, {:?}, {:?}", base_yymm, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], err, grad_norm, lr, tol);
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