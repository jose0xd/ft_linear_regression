use plotters::prelude::*;
use std::{error::Error, process};

const DATA_FILE: &str = "data.csv";

fn read_file() -> Result<Vec<(i32, i32)>, Box<dyn Error>> {
    let mut rdr = csv::Reader::from_path(DATA_FILE)?;
    let mut records: Vec<(i32, i32)> = vec![];
    for result in rdr.records() {
        let record = result?;
        if record.len() < 2 {
            eprintln!("Error: records should have 2 fields");
            process::exit(1);
        }
        let first: i32 = record.get(0).unwrap().parse()?;
        let second: i32 = record.get(1).unwrap().parse()?;
        records.push((first, second));
    }
    Ok(records)
}

#[allow(dead_code)]
fn plot(records: &Vec<(i32, i32)>) -> Result<(), Box<dyn Error>> {
    let root_area = BitMapBackend::new("plot.png", (900, 600)).into_drawing_area();
    root_area.fill(&WHITE)?;

    let mut ctx = ChartBuilder::on(&root_area)
        .y_label_area_size(60)
        .x_label_area_size(60)
        .margin(20)
        .caption(DATA_FILE, ("sans-serif", 40))
        .build_cartesian_2d(10000..300000, 2000..10000)?;

    ctx.configure_mesh()
        .y_desc("Price")
        .x_desc("Km")
        .axis_desc_style(("sans-serif", 25))
        .draw()?;

    ctx.draw_series(records.iter().map(|point| Circle::new(*point, 3, &BLUE)))?;
    Ok(())
}

fn plot_line(records: &Vec<(i32, i32)>, theta0: f32, theta1: f32) -> Result<(), Box<dyn Error>> {
    let root_area = BitMapBackend::new("plot.png", (900, 600)).into_drawing_area();
    root_area.fill(&WHITE)?;

    let mut ctx = ChartBuilder::on(&root_area)
        .y_label_area_size(60)
        .x_label_area_size(60)
        .margin(20)
        .caption(DATA_FILE, ("sans-serif", 40))
        .build_cartesian_2d(10000..300000, 2000..10000)?;

    ctx.configure_mesh()
        .y_desc("Price")
        .x_desc("Km")
        .axis_desc_style(("sans-serif", 25))
        .draw()?;

    ctx.draw_series(records.iter().map(|point| Circle::new(*point, 3, &BLUE)))?;
    ctx.draw_series(LineSeries::new((10000..300000).map(|x| (x, (theta0 + theta1 * x as f32) as i32)), &RED))?;
    Ok(())
}

fn estimate_price(x: i32, theta0: f32, theta1: f32) -> f32 {
    theta0 + (theta1 * x as f32)
}

fn train_model(records: &Vec<(i32, i32)>, learning_rate: f32, iterations: i32) -> (f32, f32) {
    let mut theta0: f32 = 0.;
    let mut theta1: f32 = 0.;

    for _ in 0..=iterations {
        let next_tetha0 = learning_rate * records.iter().fold(0., |acc: f32, (x, y)| acc + estimate_price(*x, theta0, theta1) - *y as f32) / records.len() as f32;
        let next_tetha1 = learning_rate * records.iter().fold(0., |acc: f32, (x, y)| acc + (estimate_price(*x, theta0, theta1) - *y as f32) * *x as f32) / records.len() as f32;
        theta0 -= next_tetha0;
        theta1 -= next_tetha1;
        println!("theta0: {theta0}, theta1: {theta1}");
    }
    (theta0, theta1)
}

/// Using the least-squares approach: a line that minimizes the sum of squared residuals.
/// https://en.wikipedia.org/wiki/Simple_linear_regression
#[allow(dead_code)]
fn least_squares(records: &Vec<(i32, i32)>) -> (f32, f32) {
    let average_x = records.iter().fold(0, |acc, (x, _)| acc + *x) as f32 / records.len() as f32;
    let average_y = records.iter().fold(0, |acc, (_, y)| acc + *y) as f32 / records.len() as f32;

    let theta1 = records.iter().fold(0., |acc, (x, y)| acc + (*x as f32 - average_x) * (*y as f32 - average_y)) /
        records.iter().fold(0., |acc, (x, _)| acc + (*x as f32 - average_x).powf(2.0));
    let theta0 = average_y - theta1 * average_x;
    (theta0, theta1)
}

fn main() {
    let records: Vec<(i32, i32)>;
    match read_file() {
        Ok(result) => records = result,
        Err(err) => {
            eprintln!("Error: {}", err);
            process::exit(1);
        }
    }
    // if let Err(err) = plot(&records) {
    //     println!("Error: {}", err);
    //     process::exit(1);
    // }
    // for result in records {
    //     println!("km: {}, price: {}", result.0, result.1);
    // }
    let model = train_model(&records, 0.00000000001, 40 );
    let (a, b) = least_squares(&records);
    println!("theta0: {a}, theta1: {b}");
    if let Err(err) = plot_line(&records, model.0, model.1) {
        println!("Error: {}", err);
        process::exit(1);
    }
}
