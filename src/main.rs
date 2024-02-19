extern crate nalgebra as na;
use na::DMatrix;
use rand::prelude::*;
use rand_distr::{num_traits::ToPrimitive, Normal};
use std::{error::Error, os::unix::process::parent_id, usize};


type MatrixVec = Vec<DMatrix<f64>>;

const SAMPLES: f64 = 40000.0;

fn generate_random_number() -> f64 {
    let mut rng = thread_rng();
    let normal = Normal::new(0.0, 1.0).unwrap();

    rng.sample(normal)
}

fn read_csv(filepath: &str, num_samples: &i32) -> Result<Vec<Vec<f64>>, Box<dyn Error>> {
    let mut reader = csv::Reader::from_path(filepath)?;
    let mut rows = Vec::new();
    let mut count = 0;
    for result in reader.records() {
        if count == *num_samples {
            return Ok(rows);
        }
        let record = result?;
        let row: Result<Vec<f64>, _> = record.iter().map(|field| field.parse::<f64>()).collect();
        rows.push(row?);
        count += 1;
    }
    Ok(rows)
}

fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + f64::exp(-z))
}

fn sigmoid_prime(z: f64) -> f64 {
    sigmoid(z) * (1.0 - sigmoid(z))
}

fn init(weights: &mut MatrixVec, biases: &mut MatrixVec, sizes: &[usize]) {
    for i in 1..(sizes).len() {
        weights.push(DMatrix::from_fn(sizes[i], sizes[i - 1], |_, _| {
            generate_random_number()
        }));
        biases.push(DMatrix::from_fn(sizes[i], 1, |_, _| {
            generate_random_number()
        }));
    }
}

fn feed_forward(weights: &MatrixVec,biases: &MatrixVec, x_input: &DMatrix<f64> ) -> (MatrixVec, MatrixVec){
    let mut activations: MatrixVec = Vec::new();
    let mut inputs: MatrixVec = Vec::new();
    activations.push(x_input.clone());

    for (w, b) in weights.iter().zip(biases.iter()) {
        // println!("w shape {:?} b shape{:?}", w.shape(), b.shape());
        let  mut z = w * activations.last().expect("could not get last from feed_forward");
        // add biases, need to do col by cols as z is shape (depth, samples) and b is shape (depth, 1)
        for col_index in 0..z.ncols() {
                let col = DMatrix::from_column_slice(z.nrows(), 1, z.column(col_index).as_slice());
                // let mut col = z.column_mut(col_index);
                // let col = z.column(col_index).as_slice();
                for i in (0..col.len()-1) {
                     z[(i, col_index)] += col[i] ;
                }
        }
        activations.push(z.map(|x| sigmoid(x)));
        inputs.push(z);
    }
    // println!("activations len {:?} inputs len {:?}", activations.len(), inputs.len());
    (activations, inputs)
}

fn one_hot(input: &DMatrix<f64>) -> DMatrix<f64> {
    let mut output: DMatrix<f64> = DMatrix::repeat(10, input.len(), 0.0);
    // println!("input size {:?}", input.len());
    for col in 0..input.len() {
        // println!("col {col}");
        let val = input[col] as usize;
        output[(val,col)] = 1.0;
        // println!("target {} output one hot {:?}", input[col], output.column(col).as_slice());
    }
    output

}


fn back_prop(activations: &MatrixVec, inputs: MatrixVec,weights: &MatrixVec, targets: &DMatrix<f64>) -> (MatrixVec, MatrixVec) {
    let mut dW: MatrixVec = Vec::new();
    let mut dB: MatrixVec = Vec::new();
    let mut dZ: MatrixVec = Vec::new();
    let one_hot = one_hot(targets);

    let size = weights.len();
    let z = activations.last().expect("should not be empty") - one_hot;
    dZ.push(z);
    // println!("activations len {:?}", activations.len());

    for i in (0..size).rev() {
        // println!("size {} at {}", size, i);
        // println!("activations {} shape {:?}", i, activations[i].shape());
        let dz = dZ.last().expect("backprop 1");
        // println!("weights shape {:?} dz shape {:?}", weights[i].shape(), dz.shape());
        let dw =(dz * activations[i].transpose()) * 1.0/SAMPLES;
        // println!("dw {} shape {:?}", i, dw.shape());
        dW.insert(0, dw);
        let row_sums = dz.row_iter().map(|row| row.sum()).collect::<Vec<f64>>();
        let db = DMatrix::from_column_slice(row_sums.len(), 1, &row_sums);
        let db_scaled = db * 1.0/SAMPLES;
        dB.insert(0, db_scaled);

        if i > 0 {
           let dz = weights[i].transpose() * dz;
           let sigmoids = inputs[i-1].map(|x| sigmoid_prime(x));
           // println!("sigmoids shape {:?} dz shape {:?}", sigmoids.shape(), dz.shape());
           dZ.push(dz.component_mul(&sigmoids));
        }

        // println!("{:?}", dZ.last().expect("").shape());
    }
    (dW, dB)
}

fn get_accurracy(activations:&MatrixVec, targets:&DMatrix<f64>) {
    // let one_hot = one_hot(targets);
    // println!("accurracy one hot shape {:?}", one_hot.shape());
    let mut output = activations.last().expect("no activations").clone();
    output = output.transpose();
    // println!("output {:?} targets {:?}", output.shape(), targets.shape());
    let mut total = 0;

    // Iterate over each row of the output matrix
    for (i, row) in output.row_iter().enumerate() {
        let max_index = row.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(index, _)| index);

        let target_value = targets[(i, 0)] as usize;
        if let Some(max_index) = max_index {
            if max_index == target_value {
                total += 1;
            }
        }
    }
        println!("accuracy: {}", (total as f64)/SAMPLES);
}

fn SGD(x_input: DMatrix<f64>, targets: DMatrix<f64>) {
    let sizes = [784, 64, 10];
    let mut weights: Vec<DMatrix<f64>> = Vec::new();
    let mut biases: Vec<DMatrix<f64>> = Vec::new();
    let l_r:f64 = 3.0;
    init(&mut weights, &mut biases, &sizes);
    let epochs = 500;
    for i in 0..epochs {
        let (activations, inputs) = feed_forward(&weights, &biases, &x_input);
        let (dW, dB) = back_prop(&activations, inputs, &weights, &targets);
        // println!("big dW shape {:?}", dW[0].shape());
        //update
        for i in 0..(sizes.len()-1) {
            // println!("{:?} {:?}", weights[i].shape(), dW[i].shape());
            weights[i] = &weights[i] - (&dW[i] * (l_r));
            biases[i] = &biases[i] - (&dB[i] * (l_r))
        }
        if i % 10 == 0 {
            get_accurracy(&activations, &targets);
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let sizes = [784, 30, 10];
    // let mut weights: Vec<DMatrix<f64>> = Vec::new();
    // let mut biases: Vec<DMatrix<f64>> = Vec::new();
    let num_samples = SAMPLES as i32;
    let mut data = read_csv("/Users/shaun/Dev/python/NN/train.csv", &num_samples)
        .expect("couldnt load or somthing");
    // init(&mut weights, &mut biases, &sizes);

    let mut y_train:Vec<f64> = Vec::new();

    for row in data.iter_mut() {
        y_train.push(row.remove(0))
    }

    let flattened: Vec<f64> = (0..data[0].len()).flat_map(|i| data.iter().map(move |col| col[i])).collect();

    let x_train = DMatrix::from_column_slice(data[0].len(), data.len(), &flattened);

    println!("Data loaded. Begining training!");
    println!("X_train lenght {}", x_train.len()/784);

    SGD(x_train/255.0, DMatrix::from_vec(y_train.len(), 1, y_train));

   // let (activations, _inputs) = feed_forward(&mut weights, &mut biases, &x_train);

   // for x in activations.last().expect("").column(0).iter() {
   //     println!("{:?}", x);
   // }

    Ok(())
}
